
"""Phase C: combined preprocessing, seed stability and tuning.

Uses existing phase05.py without changing results from phases A/B.
No original held-out test data is used for candidate selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fingraph.phase05 import (
    BASE_CONFIG,
    FOLDS,
    run_one,
)


SEEDS = (42, 123, 2026)


# Model, feature variant, preprocessing.
# Names also serve as identifiers in results.

CANDIDATES = {
    "xgb_all_standard": (
        "XGBoost", "all", "median_standard"
    ),
    "xgb_all_power": (
        "XGBoost", "all", "median_power"
    ),
    "xgb_local_standard": (
        "XGBoost", "local_only", "median_standard"
    ),

    "mlp_all_quantile": (
        "MLP", "all", "median_quantile"
    ),
    "mlp_noaggregate_quantile": (
        "MLP", "without_aggregate", "median_quantile"
    ),
    "mlp_local_quantile": (
        "MLP", "local_only", "median_quantile"
    ),

    "sage_all_quantile": (
        "GraphSAGE", "all", "median_quantile"
    ),
    "sage_noaggregate_quantile": (
        "GraphSAGE", "without_aggregate", "median_quantile"
    ),
    "sage_local_quantile": (
        "GraphSAGE", "local_only", "median_quantile"
    ),
}



# Ciljane konfiguracije za C.3
# Sve promjene se primjenjuju na izabrani BASE_CONFIG.

TUNING = {
    "XGBoost": {
        "shallower_regularized": {
            "max_depth": 3,
            "min_child_weight": 5,
            "reg_lambda": 5.0,
        },
        "shallower_more_trees": {
            "max_depth": 3,
            "learning_rate": 0.03,
            "n_estimators": 650,
            "reg_lambda": 10.0,
        },
        "stronger_leaf_regularization": {
            "min_child_weight": 10,
            "reg_lambda": 10.0,
        },
        "reduced_positive_weight": {
            "weight_multiplier": 0.5,
        },
    },

    "MLP": {
        # Provjera da li 150 epoha ograničava učenje
        "longer_training": {
            "epochs": 240,
            "patience": 30,
        },

        # Manja mreža i jača regularizacija
        "smaller_regularized": {
            "hidden_dim": 64,
            "dropout": 0.4,
            "weight_decay": 0.001,
            "epochs": 240,
            "patience": 30,
        },

        # Niža stopa učenja
        "lower_learning_rate": {
            "learning_rate": 0.0005,
            "epochs": 240,
            "patience": 30,
        },

        # Provjera utjecaja težine pozitivne klase
        "reduced_positive_weight": {
            "weight_multiplier": 0.5,
            "epochs": 240,
            "patience": 30,
        },
    },

    "GraphSAGE": {
        # Kontrolna konfiguracija: 2 sloja, duži trening
        "longer_training": {
            "num_layers": 2,
            "epochs": 120,
            "patience": 20,
        },

        # Izolovana promjena broja slojeva:
        # isti epochs i patience kao kontrolna verzija
        "one_layer": {
            "num_layers": 1,
            "epochs": 120,
            "patience": 20,
        },

        "three_layers": {
            "num_layers": 3,
            "epochs": 120,
            "patience": 20,
        },

        # Jača regularizacija dvoslojnog modela
        "stronger_regularization": {
            "num_layers": 2,
            "dropout": 0.4,
            "weight_decay": 0.001,
            "epochs": 120,
            "patience": 20,
        },
    },
}

def _registry_path(out):
    path = (
        Path(out)
        / "tables"
        / "phase05_c_registry.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _record(out, phase, candidate, result):
    """Save progress after every run, including cached runs."""

    path = _registry_path(out)

    row = {
        "phase": phase,
        "candidate": candidate,
        "model": result["model"],
        "fold": result["fold"],
        "seed": int(result["seed"]),
        "feature_variant": result["feature_variant"],
        "preprocess": result["preprocess"],
        "pr_auc": float(result["pr_auc"]),
        "ap_lift": float(result["ap_lift"]),
        "best_epoch": result.get("best_epoch"),
        "config_json": json.dumps(
            result["config"],
            sort_keys=True,
        ),
    }

    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True,
    )

    df = df.drop_duplicates(
        [
            "phase",
            "candidate",
            "model",
            "fold",
            "seed",
        ],
        keep="last",
    )

    df.to_csv(path, index=False)

    return row


def run_screening(
    graph,
    out,
    *,
    seeds=SEEDS,
    folds=tuple(FOLDS),
):
    """Compare selected A/B candidates across folds and seeds."""

    for candidate, (
        model,
        features,
        preprocessing,
    ) in CANDIDATES.items():

        for fold in folds:
            for seed in seeds:

                print(
                    "SCREENING",
                    candidate,
                    fold,
                    seed,
                    flush=True,
                )

                result = run_one(
                    graph,
                    out,
                    stage=f"phaseC_screening_seed_{seed}",
                    model_name=model,
                    fold_name=fold,
                    feature_variant=features,
                    preprocess=preprocessing,
                    seed=int(seed),
                    skip_existing=True,
                )

                _record(
                    out,
                    "screening",
                    candidate,
                    result,
                )

                print(
                    "PR-AUC:",
                    round(float(result["pr_auc"]), 4),
                    flush=True,
                )


def run_tuning(
    graph,
    out,
    selected_base: dict[str, str],
    *,
    seeds=(42,),
    folds=tuple(FOLDS),
):
    """Tune each model using its selected screening candidate."""

    for model, base_name in selected_base.items():

        if base_name not in CANDIDATES:
            raise ValueError(
                f"Unknown candidate: {base_name}"
            )

        (
            actual_model,
            features,
            preprocessing,
        ) = CANDIDATES[base_name]

        if model != actual_model:
            raise ValueError(
                f"{base_name} belongs to {actual_model}"
            )

        for setting, overrides in TUNING[model].items():

            for fold in folds:
                for seed in seeds:

                    print(
                        "TUNING",
                        model,
                        setting,
                        fold,
                        seed,
                        flush=True,
                    )

                    result = run_one(
                        graph,
                        out,
                        stage=f"phaseC_tuning_seed_{seed}",
                        model_name=model,
                        fold_name=fold,
                        feature_variant=features,
                        preprocess=preprocessing,
                        overrides=overrides,
                        seed=int(seed),
                        skip_existing=True,
                    )

                    _record(
                        out,
                        "tuning",
                        f"{base_name}__{setting}",
                        result,
                    )

                    print(
                        "PR-AUC:",
                        round(float(result["pr_auc"]), 4),
                        flush=True,
                    )


def run_confirmation(
    graph,
    out,
    selected_base: dict[str, str],
    chosen_tuning: dict[str, str],
    *,
    seeds=(123, 2026),
    folds=tuple(FOLDS),
):
    """Repeat selected tuned candidates on additional seeds."""

    for model, setting in chosen_tuning.items():

        base_name = selected_base[model]

        (
            actual_model,
            features,
            preprocessing,
        ) = CANDIDATES[base_name]

        if (
            model != actual_model
            or setting not in TUNING[model]
        ):
            raise ValueError(
                f"Invalid selection: {model}, {setting}"
            )

        for fold in folds:
            for seed in seeds:

                print(
                    "CONFIRM",
                    model,
                    setting,
                    fold,
                    seed,
                    flush=True,
                )

                result = run_one(
                    graph,
                    out,
                    stage=f"phaseC_confirmation_seed_{seed}",
                    model_name=model,
                    fold_name=fold,
                    feature_variant=features,
                    preprocess=preprocessing,
                    overrides=TUNING[model][setting],
                    seed=int(seed),
                    skip_existing=True,
                )

                _record(
                    out,
                    "confirmation",
                    f"{base_name}__{setting}",
                    result,
                )


def report(out):
    """Create per-fold and overall comparisons for phase C."""

    path = _registry_path(out)

    if not path.exists():
        print("No phase C results yet.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    tables = path.parent

    df.to_csv(
        tables / "phase05_c_all_runs.csv",
        index=False,
    )

    # First average over seeds within the same fold.
    by_fold = (
        df.groupby(
            ["phase", "model", "candidate", "fold"]
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_pr_auc=("pr_auc", "mean"),
            sd_seeds=("pr_auc", "std"),
            min_pr_auc=("pr_auc", "min"),
        )
        .reset_index()
    )

    by_fold.to_csv(
        tables / "phase05_c_by_fold.csv",
        index=False,
    )

    # Then summarize the different chronological folds.
    overall = (
        by_fold.groupby(
            ["phase", "model", "candidate"]
        )
        .agg(
            n_folds=("fold", "nunique"),
            mean_pr_auc=("mean_pr_auc", "mean"),
            worst_fold_mean=("mean_pr_auc", "min"),
            sd_between_folds=("mean_pr_auc", "std"),
            mean_sd_between_seeds=("sd_seeds", "mean"),
        )
        .reset_index()
        .sort_values(
            ["phase", "model", "mean_pr_auc"],
            ascending=[True, True, False],
        )
    )

    overall.to_csv(
        tables / "phase05_c_comparison.csv",
        index=False,
    )

    print(
        "Completed runs:",
        df.groupby("phase").size().to_dict(),
    )

    return overall


def run_baseline_confirmation(
    graph,
    out,
    selected_base: dict[str, str],
    chosen_tuning: dict[str, str],
    *,
    seeds=(77, 31415),
    folds=tuple(FOLDS),
):
    """Run untuned baselines on the same new seeds as tuned models.

    Allows paired comparisons between tuned and untuned
    configurations. Results from A/B/C.1/C.2 are unchanged.
    """

    for model in chosen_tuning:

        base_name = selected_base[model]

        (
            actual_model,
            features,
            preprocessing,
        ) = CANDIDATES[base_name]

        if actual_model != model:
            raise ValueError(
                f"Candidate {base_name} does not match {model}"
            )

        for fold in folds:
            for seed in seeds:

                print(
                    "BASELINE CONFIRMATION:",
                    model,
                    fold,
                    seed,
                    flush=True,
                )

                result = run_one(
                    graph,
                    out,
                    stage=(
                        f"phaseC_baseline_confirmation_seed_{seed}"
                    ),
                    model_name=model,
                    fold_name=fold,
                    feature_variant=features,
                    preprocess=preprocessing,
                    overrides=None,
                    seed=int(seed),
                    skip_existing=True,
                )

                _record(
                    out,
                    "confirmation_baseline",
                    base_name,
                    result,
                )

                print(
                    "PR-AUC:",
                    round(float(result["pr_auc"]), 4),
                    flush=True,
                )
