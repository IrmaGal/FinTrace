"""MLP control experiment on Elliptic++ with the same 182 inputs as GraphSAGE.

No explicit graph edges are used. This separates the contribution of a neural
network from the contribution of graph message passing. Hyperparameter
selection and the decision threshold use validation only; test is evaluated
once after the best checkpoint is selected.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .data import load_elliptic
from .metrics import choose_f1_threshold, evaluate
from .splits import make_time_splits


def run(
    data_dir: str,
    output_dir: str,
    seed: int = 42,
    epochs: int = 150,
    hidden_dim: int = 128,
    lr: float = 0.001,
    dropout: float = 0.3,
    patience: int = 20,
) -> dict:
    import torch
    from torch import nn

    if epochs < 1 or hidden_dim < 4 or patience < 1 or not (0 <= dropout < 1) or lr <= 0:
        raise ValueError("Neispravne postavke MLP-a")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    graph = load_elliptic(data_dir)
    splits = make_time_splits(graph)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    observed_training_period = graph.times <= splits.train_end
    imputer.fit(graph.features[observed_training_period])
    scaler = StandardScaler()
    scaler.fit(imputer.transform(graph.features[observed_training_period]))

    def transformed(mask: np.ndarray) -> np.ndarray:
        return scaler.transform(imputer.transform(graph.features[mask])).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.as_tensor(transformed(splits.train), device=device)
    y_train_np = graph.labels[splits.train]
    y_train = torch.as_tensor(y_train_np, dtype=torch.long, device=device)
    x_valid = torch.as_tensor(transformed(splits.valid), device=device)
    y_valid_np = graph.labels[splits.valid]
    x_test = torch.as_tensor(transformed(splits.test), device=device)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, 2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n_pos, n_neg = int((y_train_np == 1).sum()), int((y_train_np == 0).sum())
    weights = torch.tensor([1.0, n_neg / n_pos], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_score = -1.0
    best_epoch = 0
    best_state = None
    without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_probs = torch.softmax(model(x_valid), dim=1)[:, 1].cpu().numpy()
        valid_ap = float(average_precision_score(y_valid_np, valid_probs))
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": valid_ap})
        print(f"Epoch {epoch:03d} | train loss={loss.item():.4f} | val PR-AUC={valid_ap:.4f}")
        if valid_ap > best_score + 1e-5:
            best_score = valid_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= patience:
                print(f"Early stopping: {patience} epoha bez napretka.")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_valid = torch.softmax(model(x_valid), dim=1)[:, 1].cpu().numpy()
        p_test = torch.softmax(model(x_test), dim=1)[:, 1].cpu().numpy()
    threshold = choose_f1_threshold(y_valid_np, p_valid)

    result = {
        "model": "MLP", "seed": seed, "device": str(device),
        "best_epoch": best_epoch, "max_epochs": epochs, "hidden_dim": hidden_dim,
        "learning_rate": lr, "dropout": dropout, "patience": patience,
        "input_features": int(x_train.shape[1]),
        "train_end": splits.train_end, "valid_end": splits.valid_end,
        "validation": evaluate(y_valid_np, p_valid, threshold),
        "test": evaluate(graph.labels[splits.test], p_test, threshold),
        "protocol_note": (
            "Test scores are for reporting only. Do not tune new hyperparameters "
            "or thresholds on test results."
        ),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": int(x_train.shape[1]),
        "hidden_dim": hidden_dim, "dropout": dropout,
        "seed": seed, "best_epoch": best_epoch,
    }, target / "mlp_checkpoint.pt")
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_names": graph.feature_names},
                target / "preprocessing.joblib")
    (target / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(target / "training_history.csv", index=False)
    for split_name, mask, probabilities in (
        ("validation", splits.valid, p_valid),
        ("test", splits.test, p_test),
    ):
        pd.DataFrame({
            "txId": graph.tx_ids[mask],
            "time_step": graph.times[mask],
            "label": graph.labels[mask],
            "probability_illicit": probabilities,
        }).to_csv(target / f"{split_name}_predictions.csv", index=False)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output-dir", default="outputs/mlp_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=20)
    args = p.parse_args()
    result = run(args.data_dir, args.output_dir, args.seed, args.epochs,
                 args.hidden_dim, args.lr, args.dropout, args.patience)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
