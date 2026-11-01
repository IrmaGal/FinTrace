"""Full-batch two-layer GraphSAGE with chronological edge filtering.

Targets are transaction-level illicit labels, not ground-truth group labels.
Uses all observed unlabeled nodes as graph context, never as negative labels.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .data import load_elliptic
from .metrics import choose_f1_threshold, evaluate
from .splits import edges_observed_until, make_time_splits


def run(
    data_dir: str,
    output_dir: str,
    seed: int = 42,
    epochs: int = 60,
    hidden_dim: int = 64,
) -> dict:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import to_undirected

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    graph = load_elliptic(data_dir)
    splits = make_time_splits(graph)

    # Fit preprocessing on *past* nodes, including unknown ones. We do not use
    # any future feature distribution for the imputer/scaler.
    observed_train = graph.times <= splits.train_end
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(graph.features[observed_train])
    scaler = StandardScaler()
    scaler.fit(imputer.transform(graph.features[observed_train]))
    x_np = scaler.transform(imputer.transform(graph.features)).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(graph.labels).long().to(device)
    train_mask = torch.from_numpy(splits.train).bool().to(device)

    def edge_tensor(until: int):
        # SAGE needs messages from both predecessor and successor transactions;
        # however, no future node/edge is available in earlier periods.
        e = torch.from_numpy(edges_observed_until(graph, until)).long().to(device)
        return to_undirected(e, num_nodes=graph.num_nodes)

    train_edges = edge_tensor(splits.train_end)
    # Evaluation is strictly time-aware: a transaction at step t can only
    # receive messages through nodes/edges already observed by t, never t+1.
    eval_times = sorted(set(graph.times[splits.valid | splits.test].tolist()))
    eval_edges = {int(t): edge_tensor(int(t)) for t in eval_times}

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_dim: int, hidden: int):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, 2)

        def forward(self, features, edges):
            hidden = F.relu(self.conv1(features, edges))
            hidden = F.dropout(hidden, p=0.3, training=self.training)
            return self.conv2(hidden, edges)

    model = GraphSAGE(x.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    train_y = graph.labels[splits.train]
    n_neg, n_pos = int((train_y == 0).sum()), int((train_y == 1).sum())
    weights = torch.tensor([1.0, n_neg / n_pos], device=device)

    def predict_chronologically(mask: np.ndarray) -> np.ndarray:
        """Predict each evaluation time slice using only its past graph."""
        selected = np.flatnonzero(mask)
        scores = np.empty(selected.size, dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for step in sorted(set(graph.times[selected].tolist())):
                positions = np.flatnonzero(graph.times[selected] == step)
                node_indices = torch.as_tensor(selected[positions], device=device)
                probs = torch.softmax(model(x, eval_edges[int(step)]), dim=1)[:, 1]
                scores[positions] = probs[node_indices].cpu().numpy()
        return scores

    best_pr_auc = -1.0
    best_state = None
    best_epoch = 0
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, train_edges)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        model.eval()
        val_scores = predict_chronologically(splits.valid)
        ap = float(average_precision_score(graph.labels[splits.valid], val_scores))
        print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val PR-AUC={ap:.4f}")

        if ap > best_pr_auc + 1e-5:
            best_pr_auc = ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    p_valid = predict_chronologically(splits.valid)
    p_test = predict_chronologically(splits.test)
    threshold = choose_f1_threshold(graph.labels[splits.valid], p_valid)

    result = {
        "model": "GraphSAGE-2", "seed": seed, "device": str(device),
        "best_epoch": best_epoch, "hidden_dim": hidden_dim,
        "train_end": splits.train_end, "valid_end": splits.valid_end,
        "validation": evaluate(graph.labels[splits.valid], p_valid, threshold),
        "test": evaluate(graph.labels[splits.test], p_test, threshold),
    }
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dest / "graphsage_state.pt")
    (dest / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({
        "txId": graph.tx_ids[splits.test],
        "time_step": graph.times[splits.test],
        "label": graph.labels[splits.test],
        "probability_illicit": p_test,
    }).to_csv(dest / "test_predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/graphsage")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir, args.seed, args.epochs, args.hidden_dim)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
