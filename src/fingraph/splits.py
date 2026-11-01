"""Chronological partitions for Elliptic++ experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TransactionGraph


@dataclass(frozen=True)
class TimeSplits:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    train_end: int
    valid_end: int


def make_time_splits(
    graph: TransactionGraph, train_end: int = 34, valid_end: int = 41
) -> TimeSplits:
    if not 0 < train_end < valid_end < graph.times.max():
        raise ValueError("Granice podjele moraju biti strogo rastuće i prije zadnjeg koraka")
    known = graph.labels >= 0
    masks = (
        known & (graph.times <= train_end),
        known & (graph.times > train_end) & (graph.times <= valid_end),
        known & (graph.times > valid_end),
    )
    for name, mask in zip(("train", "valid", "test"), masks):
        if not mask.any() or np.unique(graph.labels[mask]).size != 2:
            raise ValueError(
                f"{name}: nedovoljno poznatih pozitivnih/negativnih oznaka; "
                "pregledaj distribuciju po vremenskim koracima prije izbora protokola"
            )
    return TimeSplits(*masks, train_end=train_end, valid_end=valid_end)


def edges_observed_until(graph: TransactionGraph, time_step: int) -> np.ndarray:
    """Keep only links between nodes present by the specified time step."""
    src, dst = graph.edge_index
    keep = (graph.times[src] <= time_step) & (graph.times[dst] <= time_step)
    return graph.edge_index[:, keep]
