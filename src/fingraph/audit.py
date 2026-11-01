"""First command to run after downloading Elliptic++ CSV files."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .data import load_elliptic


def describe(data_dir: str) -> dict:
    graph = load_elliptic(data_dir)
    counts = {
        "illicit": int((graph.labels == 1).sum()),
        "licit": int((graph.labels == 0).sum()),
        "unknown": int((graph.labels == -1).sum()),
    }
    by_time = (
        pd.DataFrame({"time": graph.times, "label": graph.labels})
        .groupby(["time", "label"]).size().unstack(fill_value=0)
    )
    by_time = by_time.rename(columns={-1: "unknown", 0: "licit", 1: "illicit"})
    by_time = by_time.reindex(columns=["illicit", "licit", "unknown"], fill_value=0)

    return {
        "nodes": graph.num_nodes,
        "directed_edges": graph.num_edges,
        "features": graph.features.shape[1],
        "missing_feature_values": int(np.isnan(graph.features).sum()),
        "time_steps": int(np.unique(graph.times).size),
        "labels": counts,
        "class_distribution_by_time": {
            str(i): row for i, row in by_time.to_dict(orient="index").items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    print(json.dumps(describe(args.data_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
