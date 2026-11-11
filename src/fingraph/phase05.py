"""Phase 05: chronological feature ablations, preprocessing comparisons and tuning.

No held-out test is read for candidate selection. All fitted transforms see only
nodes from the training period (unknown labels allowed, never targets).
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer

FOLDS = {"A": (20, 27), "B": (27, 34), "C": (34, 41)}
FEATURE_VARIANTS = ("all", "without_aggregate", "local_only", "aggregate_only", "named_only", "all_pca95")
PREPROCESS_VARIANTS = ("median_robust", "median_log_robust", "median_quantile", "median_power")
SUPPORTED_PREPROCESS = ("median_standard",) + PREPROCESS_VARIANTS
BASE_PREPROCESS = "median_standard"
BASE_CONFIG = {
    "XGBoost": {"n_estimators": 350, "max_depth": 5, "learning_rate": .05, "subsample": .85,
                "colsample_bytree": .85, "min_child_weight": 1, "reg_lambda": 1., "weight_multiplier": 1.},
    "MLP": {"hidden_dim": 128, "dropout": .3, "learning_rate": .001, "weight_decay": 1e-4,
            "epochs": 150, "patience": 20, "weight_multiplier": 1.},
    "GraphSAGE": {"hidden_dim": 64, "dropout": .3, "learning_rate": .001, "weight_decay": 1e-4,
                  "epochs": 60, "patience": 10, "weight_multiplier": 1., "edge_mode": "real", "num_layers": 2},
}
# Small and hypothesis-driven, not a large grid search.
TUNING_CONFIGS = {
    "XGBoost": [
        {"max_depth": 3, "min_child_weight": 5, "reg_lambda": 5.},
        {"max_depth": 3, "learning_rate": .03, "n_estimators": 650, "reg_lambda": 10.},
        {"max_depth": 5, "min_child_weight": 10, "reg_lambda": 10.},
        {"max_depth": 5, "weight_multiplier": .5},
    ],
    "MLP": [
        {"hidden_dim": 64, "dropout": .5, "weight_decay": .001},
        {"hidden_dim": 64, "dropout": .4, "learning_rate": .0003},
        {"hidden_dim": 128, "dropout": .5, "weight_decay": .001},
        {"hidden_dim": 128, "weight_multiplier": .5},
    ],
    "GraphSAGE": [
        {"num_layers": 1, "epochs": 90, "patience": 15},
        {"num_layers": 3, "epochs": 90, "patience": 15},
        {"hidden_dim": 32, "dropout": .5, "weight_decay": .001, "epochs": 120, "patience": 20},
        {"hidden_dim": 64, "dropout": .4, "epochs": 120, "patience": 20},
        {"hidden_dim": 128, "dropout": .4, "learning_rate": .0005, "epochs": 120, "patience": 20},
        {"hidden_dim": 64, "epochs": 120, "patience": 20, "weight_multiplier": .5},
    ],
}


def feature_indices(names, variant: str) -> np.ndarray:
    """Select exact columns from the loader's actual names, never from fixed positions."""
    names = [str(n) for n in names]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names")
    local = np.array([i for i, n in enumerate(names) if n.startswith("Local_feature_")], dtype=int)
    aggregate = np.array([i for i, n in enumerate(names) if n.startswith("Aggregate_feature_")], dtype=int)
    named = np.array([i for i, n in enumerate(names)
                      if not n.startswith(("Local_feature_", "Aggregate_feature_"))], dtype=int)
    if variant in ("all", "all_pca95"):
        result = np.arange(len(names), dtype=int)
    elif variant == "without_aggregate":
        result = np.array([i for i in range(len(names)) if i not in set(aggregate)], dtype=int)
    elif variant == "local_only":
        result = local
    elif variant == "aggregate_only":
        result = aggregate
    elif variant == "named_only":
        result = named
    else:
        raise ValueError(f"Unknown feature variant: {variant}")
    if not len(result):
        raise ValueError(f"No features selected for {variant}")
    return result


@dataclass
class FeatureProcessor:
    columns: np.ndarray
    imputer: SimpleImputer
    scaler: object
    log_columns: np.ndarray
    pca: PCA | None
    feature_names: tuple[str, ...]
    fit_count: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        a = self.imputer.transform(np.asarray(x)[:, self.columns]).astype(np.float64)
        if len(self.log_columns):
            a[:, self.log_columns] = np.log1p(np.maximum(a[:, self.log_columns], 0))
        a = self.scaler.transform(a)
        if self.pca is not None:
            a = self.pca.transform(a)
        return a.astype(np.float32)

    @property
    def n_features(self) -> int:
        return int(self.pca.n_components_) if self.pca is not None else len(self.columns)


class SafePowerTransformer:
    """Yeo-Johnson on variable columns, pass constant columns unchanged."""
    def fit_transform(self, a):
        a = np.asarray(a, dtype=np.float64)
        self.variable_ = np.flatnonzero(np.nanstd(a, axis=0) > 1e-12)
        self.constant_ = np.setdiff1d(np.arange(a.shape[1]), self.variable_)
        self.n_features_in_ = a.shape[1]
        self.pt_ = PowerTransformer(method="yeo-johnson", standardize=True) if len(self.variable_) else None
        output = np.zeros_like(a)
        if self.pt_ is not None:
            output[:, self.variable_] = self.pt_.fit_transform(a[:, self.variable_])
        return output

    def transform(self, a):
        a = np.asarray(a, dtype=np.float64)
        output = np.zeros_like(a)
        if self.pt_ is not None:
            output[:, self.variable_] = self.pt_.transform(a[:, self.variable_])
        return output


def fit_processor(graph, train_end: int, feature_variant: str = "all", preprocess: str = BASE_PREPROCESS) -> FeatureProcessor:
    """Train-only fit, no test labels or validation feature distribution."""
    if preprocess not in SUPPORTED_PREPROCESS:
        raise ValueError(preprocess)
    cols = feature_indices(graph.feature_names, feature_variant)
    observed = graph.times <= train_end
    raw = graph.features[observed][:, cols]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    a = imputer.fit_transform(raw).astype(np.float64)
    log_cols = np.empty(0, dtype=int)
    if preprocess == "median_log_robust":
        nonnegative = (np.min(a, axis=0) >= 0)
        asymmetry = np.abs(skew(a, axis=0, nan_policy="omit", bias=False))
        asymmetry = np.nan_to_num(asymmetry, nan=0, posinf=0, neginf=0)
        log_cols = np.flatnonzero(nonnegative & (asymmetry > 2) & (np.max(a, axis=0) > 0))
        if len(log_cols):
            a[:, log_cols] = np.log1p(a[:, log_cols])
    if preprocess == "median_standard":
        scaler = StandardScaler()
    elif preprocess in ("median_robust", "median_log_robust"):
        scaler = RobustScaler()
    elif preprocess == "median_quantile":
        scaler = QuantileTransformer(
            n_quantiles=min(1000, len(a)), output_distribution="normal", random_state=42
        )
    elif preprocess == "median_power":
        # Yeo-Johnson supports zero and negative inputs. Constant columns are handled
        # separately because PowerTransformer cannot fit them.
        scaler = SafePowerTransformer()
    a = scaler.fit_transform(a)
    pca = None
    if feature_variant == "all_pca95":
        # PCA uses standardized values, fit exclusively on historical inputs.
        pca = PCA(n_components=.95, svd_solver="full")
        pca.fit(a)
    return FeatureProcessor(cols, imputer, scaler, log_cols, pca,
                            tuple(graph.feature_names[int(j)] for j in cols), int(observed.sum()))


def masks(graph, train_end: int, valid_end: int):
    known = graph.labels >= 0
    train = known & (graph.times <= train_end)
    valid = known & (graph.times > train_end) & (graph.times <= valid_end)
    for which, mask in (("train", train), ("valid", valid)):
        if np.unique(graph.labels[mask]).size != 2:
            raise ValueError(f"{which} requires both classes")
    return train, valid


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _fit_xgboost(graph, processor, train, valid, cfg, seed):
    from xgboost import XGBClassifier
    yy = graph.labels[train]
    ratio = float(np.sum(yy == 0) / np.sum(yy == 1))
    model = XGBClassifier(
        n_estimators=int(cfg['n_estimators']), max_depth=int(cfg['max_depth']),
        learning_rate=float(cfg['learning_rate']), subsample=float(cfg['subsample']),
        colsample_bytree=float(cfg['colsample_bytree']),
        min_child_weight=float(cfg['min_child_weight']), reg_lambda=float(cfg['reg_lambda']),
        scale_pos_weight=ratio * float(cfg['weight_multiplier']),
        tree_method='hist', objective='binary:logistic', eval_metric='logloss',
        random_state=seed, n_jobs=2)
    model.fit(processor.transform(graph.features[train]), yy)
    pred = model.predict_proba(processor.transform(graph.features[valid]))[:, 1]
    gain = model.feature_importances_.tolist()
    return pred, [], {"feature_importance": gain, "best_epoch": None}


def _fit_mlp(graph, processor, train, valid, cfg, seed):
    import torch
    from torch import nn
    _seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xtr = torch.as_tensor(processor.transform(graph.features[train]), device=device)
    xva = torch.as_tensor(processor.transform(graph.features[valid]), device=device)
    ytr = torch.as_tensor(graph.labels[train], dtype=torch.long, device=device)
    yva = graph.labels[valid]
    hidden = int(cfg['hidden_dim'])
    model = nn.Sequential(nn.Linear(xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(float(cfg['dropout'])),
                          nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(float(cfg['dropout'])),
                          nn.Linear(hidden // 2, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
    weight = torch.tensor([1., float((graph.labels[train] == 0).sum() / (graph.labels[train] == 1).sum()) * float(cfg['weight_multiplier'])], device=device)
    crit = nn.CrossEntropyLoss(weight=weight)
    best, state, best_ep, bad, history = -1., None, 0, 0, []
    for ep in range(1, int(cfg['epochs']) + 1):
        model.train(); opt.zero_grad(); loss = crit(model(xtr), ytr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(xva), dim=1)[:, 1].cpu().numpy()
        ap = float(average_precision_score(yva, p))
        history.append({'epoch': ep, 'train_loss': float(loss.item()), 'val_pr_auc': ap})
        if ap > best + 1e-5:
            best, state, best_ep, bad = ap, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= int(cfg['patience']):
                break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        p = torch.softmax(model(xva), dim=1)[:, 1].cpu().numpy()
    return p, history, {'best_epoch': best_ep, 'device': str(device)}


def _fit_graphsage(graph, processor, train, valid, cfg, seed, train_end, valid_end):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import to_undirected
    _seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Remap to only nodes observed by valid_end, never future nodes.
    observed = graph.times <= valid_end
    old = np.flatnonzero(observed)
    remap = np.full(graph.num_nodes, -1, dtype=np.int64)
    remap[old] = np.arange(len(old))
    x = torch.as_tensor(processor.transform(graph.features[observed]), device=device)
    train_nodes = torch.as_tensor(remap[np.flatnonzero(train)], device=device)
    valid_nodes = torch.as_tensor(remap[np.flatnonzero(valid)], device=device)
    yy = torch.as_tensor(graph.labels[observed], dtype=torch.long, device=device)
    e = graph.edge_index
    def until(t):
        keep = (graph.times[e[0]] <= t) & (graph.times[e[1]] <= t)
        edge = torch.as_tensor(remap[e[:, keep]], dtype=torch.long, device=device)
        return to_undirected(edge, num_nodes=len(old))
    etrain, evalid = until(train_end), until(valid_end)
    if cfg.get('edge_mode') == 'self_only':
        etrain = torch.empty((2, 0), dtype=torch.long, device=device)
        evalid = etrain
    hidden = int(cfg['hidden_dim']); dropout = float(cfg['dropout'])
    num_layers = int(cfg.get("num_layers", 2))
    if not 1 <= num_layers <= 4:
        raise ValueError("GraphSAGE num_layers must be between 1 and 4")

    class ConfigurableSAGE(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            dimensions = [dim] + [hidden] * (num_layers - 1) + [2]
            self.convs = torch.nn.ModuleList([
                SAGEConv(dimensions[i], dimensions[i + 1])
                for i in range(num_layers)
            ])

        def forward(self, xx, edges):
            for conv in self.convs[:-1]:
                xx = F.relu(conv(xx, edges))
                xx = F.dropout(xx, p=dropout, training=self.training)
            return self.convs[-1](xx, edges)

    model = ConfigurableSAGE(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
    label_train = graph.labels[train]
    w = torch.tensor([1., float((label_train == 0).sum() / (label_train == 1).sum()) * float(cfg['weight_multiplier'])], device=device)
    best, state, best_ep, bad, history = -1., None, 0, 0, []
    for ep in range(1, int(cfg['epochs']) + 1):
        model.train(); opt.zero_grad(); z = model(x, etrain)
        loss = F.cross_entropy(z[train_nodes], yy[train_nodes], weight=w)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(x, evalid)[valid_nodes], dim=1)[:, 1].cpu().numpy()
        ap = float(average_precision_score(graph.labels[valid], p))
        history.append({'epoch': ep, 'train_loss': float(loss.item()), 'val_pr_auc': ap})
        if ap > best + 1e-5:
            best, state, best_ep, bad = ap, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= int(cfg['patience']):
                break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x, evalid)[valid_nodes], dim=1)[:, 1].cpu().numpy()
    return p, history, {'best_epoch': best_ep, 'device': str(device), 'edge_mode': cfg.get('edge_mode', 'real')}


def run_one(graph, out_dir: str | Path, *, stage: str, model_name: str,
            fold_name: str, feature_variant: str = 'all', preprocess: str = BASE_PREPROCESS,
            overrides: dict | None = None, seed: int = 42, skip_existing: bool = True) -> dict:
    """Run one candidate on ONE development fold, never against original held-out test."""
    if fold_name not in FOLDS or model_name not in BASE_CONFIG:
        raise ValueError('Unknown model/fold')
    train_end, valid_end = FOLDS[fold_name]
    cfg = {**BASE_CONFIG[model_name], **(overrides or {})}
    if model_name == 'XGBoost' and feature_variant == 'all_pca95':
        # Allowed for empirical comparison, but often hurts tree splits.
        pass
    base = Path(out_dir) / 'runs' / stage / model_name.lower() / f'fold_{fold_name}' / f'{feature_variant}__{preprocess}__{candidate_id(cfg, BASE_CONFIG[model_name])}'
    base.mkdir(parents=True, exist_ok=True)
    metrics_path = base / 'phase05_metrics.json'
    if skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding='utf-8'))
    _seed(seed)
    train, valid = masks(graph, train_end, valid_end)
    processor = fit_processor(graph, train_end, feature_variant, preprocess)
    tic = time.time()
    if model_name == 'XGBoost':
        p, history, extra = _fit_xgboost(graph, processor, train, valid, cfg, seed)
    elif model_name == 'MLP':
        p, history, extra = _fit_mlp(graph, processor, train, valid, cfg, seed)
    else:
        p, history, extra = _fit_graphsage(graph, processor, train, valid, cfg, seed, train_end, valid_end)
    from fingraph.metrics import choose_f1_threshold, evaluate
    # Development-only threshold, reported on SAME valid fold: optimistic for thresholded metrics.
    threshold = choose_f1_threshold(graph.labels[valid], p)
    stats = evaluate(graph.labels[valid], p, threshold)
    result = dict(stage=stage, model=model_name, fold=fold_name, train_end=train_end,
                  valid_end=valid_end, feature_variant=feature_variant, preprocess=preprocess,
                  n_inputs=processor.n_features, fit_observed_nodes=processor.fit_count,
                  seed=seed, config=cfg, metrics=stats, pr_auc=stats['pr_auc'],
                  ap_lift=stats['pr_auc'] / (stats['positives'] / stats['n']),
                  runtime_sec=round(time.time() - tic, 2),
                  best_epoch=extra.get('best_epoch'),
                  validation_only=True,
                  note='No test selection. Threshold selected on this validation fold; precision/recall/F1 descriptive and optimistic.')
    if processor.pca is not None:
        result['pca_explained_variance'] = float(np.sum(processor.pca.explained_variance_ratio_))
    if model_name == 'XGBoost':
        pd.DataFrame({'feature': [f'PC{i+1}' for i in range(processor.n_features)] if processor.pca else processor.feature_names,
                      'importance': extra['feature_importance']}).sort_values('importance', ascending=False).to_csv(base / 'phase05_xgboost_importance.csv', index=False)
    if len(history):
        pd.DataFrame(history).to_csv(base / 'phase05_history.csv', index=False)
    pd.DataFrame({'txId': graph.tx_ids[valid], 'time_step': graph.times[valid],
                  'label': graph.labels[valid], 'probability_illicit': p}).to_csv(base / 'phase05_validation_predictions.csv', index=False)
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def candidate_id(cfg: dict, baseline: dict) -> str:
    import hashlib
    raw = json.dumps(cfg, sort_keys=True)
    if cfg == baseline:
        return 'baseline'
    return 'cfg_' + hashlib.sha256(raw.encode()).hexdigest()[:10]


def summary_table(root: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted((Path(root) / 'runs').rglob('phase05_metrics.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        rows.append({key: r.get(key) for key in ('stage', 'model', 'fold', 'feature_variant', 'preprocess', 'n_inputs', 'pr_auc', 'ap_lift', 'runtime_sec', 'best_epoch')} |
                    {'config_id': candidate_id(r['config'], BASE_CONFIG[r['model']]),
                     'precision': r['metrics']['precision'], 'recall': r['metrics']['recall'],
                     'f1': r['metrics']['f1']})
    return pd.DataFrame(rows)


def selection_table(summary: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Mean of folds for development selection; not an unbiased test result."""
    x = summary.loc[summary.stage == stage].copy()
    if x.empty:
        return pd.DataFrame()
    keys = ['model', 'feature_variant', 'preprocess', 'config_id']
    return (x.groupby(keys, dropna=False)
             .agg(folds=('fold', 'nunique'), mean_pr_auc=('pr_auc', 'mean'),
                  worst_pr_auc=('pr_auc', 'min'), std_pr_auc=('pr_auc', 'std'),
                  mean_ap_lift=('ap_lift', 'mean'), mean_f1=('f1', 'mean'))
             .reset_index().sort_values(['model', 'mean_pr_auc'], ascending=[True, False]))
