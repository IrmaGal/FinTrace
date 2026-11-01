"""Read the Elliptic++ transaction graph with strict identifier/label checks.

Labels: 1=illicit, 2=licit, 3=unknown. Internal targets: 1, 0, -1.
Only train/validation/test labels 0 and 1. Unknown nodes may be graph context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransactionGraph:
    tx_ids: np.ndarray
    times: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    edge_index: np.ndarray  # Shape [2, E], original direction txId1 -> txId2.

    @property
    def num_nodes(self) -> int:
        return len(self.tx_ids)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def load_elliptic(data_dir: str | Path) -> TransactionGraph:
    folder = Path(data_dir)
    paths = {name: folder / name for name in (
        "txs_features.csv", "txs_classes.csv", "txs_edgelist.csv"
    )}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Nedostaju CSV datoteke: {missing}. Preuzmi pune podatke s linka autora."
        )

    features = pd.read_csv(paths["txs_features.csv"])
    classes = pd.read_csv(paths["txs_classes.csv"])
    edges = pd.read_csv(paths["txs_edgelist.csv"])

    expected = [
        (features, {"txId", "Time step"}, "features"),
        (classes, {"txId", "class"}, "classes"),
        (edges, {"txId1", "txId2"}, "edgelist"),
    ]
    for frame, required, name in expected:
        absent = required - set(frame.columns)
        if absent:
            raise ValueError(f"{name}: nedostaju kolone {sorted(absent)}")

    for frame, name in ((features, "features"), (classes, "classes")):
        if frame["txId"].isna().any() or frame["txId"].duplicated().any():
            raise ValueError(f"{name}: txId mora biti jedinstven i neprazan")

    # Some third-party notebooks insert class into the feature frame. Never
    # trust or expose that column as a model feature; classes.csv is canonical.
    features = features.drop(columns=["class", "label", "raw_label"], errors="ignore")
    merged = features.merge(
        classes[["txId", "class"]], on="txId", how="left", validate="one_to_one",
        sort=False,
    )
    if merged["class"].isna().any():
        raise ValueError("Postoje transakcije bez oznake u txs_classes.csv")
    raw_labels = pd.to_numeric(merged["class"], errors="raise").to_numpy()
    if not np.isin(raw_labels, [1, 2, 3]).all():
        raise ValueError("Očekivane oznake su 1 (illicit), 2 (licit), 3 (unknown)")
    labels = np.where(raw_labels == 1, 1, np.where(raw_labels == 2, 0, -1))

    times = pd.to_numeric(merged["Time step"], errors="raise")
    if times.isna().any() or (times < 1).any():
        raise ValueError("Time step mora biti pozitivan i bez praznih vrijednosti")
    times_np = times.to_numpy(dtype=np.int64)

    non_features = {"txId", "Time step", "class", "label", "raw_label"}
    feature_names = tuple(str(col) for col in merged.columns if col not in non_features)
    if not feature_names:
        raise ValueError("Nisu pronađene numeričke karakteristike transakcija")
    feature_matrix = merged[list(feature_names)].apply(pd.to_numeric, errors="raise")
    values = feature_matrix.to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = np.nan

    # Map tx identifiers to positions in features. Never assume IDs are 0..N-1
    ids = merged["txId"].to_numpy()
    id_to_row = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    left = edges["txId1"].map(id_to_row)
    right = edges["txId2"].map(id_to_row)
    if left.isna().any() or right.isna().any():
        invalid = int((left.isna() | right.isna()).sum())
        raise ValueError(f"Edgelist sadrži {invalid} veza prema nepoznatim txId")
    edge_index = np.stack((left.to_numpy(dtype=np.int64), right.to_numpy(dtype=np.int64)))

    return TransactionGraph(
        tx_ids=ids, times=times_np, labels=labels.astype(np.int64),
        features=values, feature_names=feature_names, edge_index=edge_index,
    )
