#!/usr/bin/env python3

"""
Threshold-selected pairwise cross-relay analysis.

This module performs both:

1. Source-validation threshold evaluation
2. Threshold result analysis and plotting

Protocol
--------
For each source relay:

1. Load the already-trained source model.
2. Predict on SOURCE validation data.
3. Select one threshold using Youden's J:
       J = TPR - FPR
4. Freeze that threshold.
5. Evaluate the source model on the COMMON TEST simulations
   for every target relay using the same threshold.

Important
---------
- No target test data is used for threshold selection.
- No retraining is performed.
- AUC is threshold-independent and uses raw logits.
- The common simulation-level split is reconstructed using seed 42.
- No experiment name or timestamp is hard-coded.
- All output is stored below:

    <run_dir>/analysis/threshold/
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Project root
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# Project imports
# ============================================================

from scripts.utils.dataset import load_dataset
from scripts.datasets.relay_dataset import RelayDataset
from scripts.models.compact_mlp import CompactMLP_FD


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 1024

NUM_WORKERS = 4

THRESHOLD_METHOD = "youden_j"

MODEL_NAME = "compact_mlp"

RELAY_ORDER = [
    "Bus1_Ln12A",
    "Bus1_Ln12B",
    "Bus2_Ln12A",
    "Bus2_Ln12B",
    "Bus2_Ln23A",
    "Bus2_Ln23B",
    "Bus3_Ln23A",
    "Bus3_Ln23B",
]

RELAY_NAME_MAP = {
    "Bus1_Ln12A": "Bus_1_Line_1-2A",
    "Bus1_Ln12B": "Bus_1_Line_1-2B",
    "Bus2_Ln12A": "Bus_2_Line_1-2A",
    "Bus2_Ln12B": "Bus_2_Line_1-2B",
    "Bus2_Ln23A": "Bus_2_Line_2-3A",
    "Bus2_Ln23B": "Bus_2_Line_2-3B",
    "Bus3_Ln23A": "Bus_3_Line_2-3A",
    "Bus3_Ln23B": "Bus_3_Line_2-3B",
}


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Model loading
# ============================================================

def find_model_file(
    source_relay: str,
    training_run: Path,
) -> Path:
    """
    Find the saved model for a source relay.
    """

    candidates = [
        training_run / f"{source_relay}.pt",
        training_run / "models" / f"{source_relay}.pt",
        training_run / f"model_{source_relay}.pt",
        training_run / "models" / f"model_{source_relay}.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(
        training_run.rglob(
            f"*{source_relay}*.pt"
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple model files found for {source_relay}:\n"
            + "\n".join(str(x) for x in matches)
        )

    raise FileNotFoundError(
        f"Could not find saved model for {source_relay} "
        f"under {training_run}"
    )


def load_model(
    source_relay: str,
    training_run: Path,
):
    """
    Load one trained CompactMLP_FD model.
    """

    model_path = find_model_file(
        source_relay,
        training_run,
    )

    print(
        f"Loading model: {model_path}"
    )

    model = CompactMLP_FD(
        input_dim=480 * 6,
        h1=256,
        h2=128,
    )

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    return model, model_path


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_dataset(
    model,
    dataset,
):
    """
    Return raw logits and labels.

    Raw logits are used for:
    - threshold selection
    - classification
    - ROC AUC

    This avoids sigmoid saturation.
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    all_logits = []
    all_labels = []

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            X_batch
        )

        all_logits.append(
            logits.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_labels.append(
            y_batch.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    logits = np.concatenate(
        all_logits
    )

    labels = np.concatenate(
        all_labels
    )

    return logits, labels


# ============================================================
# Threshold selection
# ============================================================

def select_youden_threshold(
    y_true,
    logits,
):
    """
    Select the threshold maximizing Youden's J.

        J = TPR - FPR

    Threshold is selected only on source validation data.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        logits,
    )

    j_scores = tpr - fpr

    finite = np.isfinite(
        thresholds
    )

    valid_indices = np.where(
        finite
    )[0]

    if len(valid_indices) == 0:
        raise RuntimeError(
            "No finite thresholds returned by roc_curve."
        )

    best_local = valid_indices[
        np.argmax(
            j_scores[valid_indices]
        )
    ]

    threshold = float(
        thresholds[best_local]
    )

    return {
        "threshold": threshold,
        "youden_j": float(
            j_scores[best_local]
        ),
        "validation_tpr": float(
            tpr[best_local]
        ),
        "validation_fpr": float(
            fpr[best_local]
        ),
        "validation_fnr": float(
            1.0 - tpr[best_local]
        ),
    }


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true,
    logits,
    threshold,
):
    """
    Calculate metrics using a frozen logit threshold.

    AUC is calculated from raw logits and is therefore
    threshold-independent.
    """

    predictions = (
        logits >= threshold
    ).astype(np.int64)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_true,
        logits,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


# ============================================================
# Dataset preparation
# ============================================================

def prepare_datasets():
    """
    Reconstruct the common simulation-level split used by
    the corrected pairwise experiment.

    The split is:

        15% test
        15% validation overall
        70% train

    Grouping variable:
        sample_id

    X remains lazy and is never materialized.
    """

    print()
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    X, y = load_dataset()

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    # --------------------------------------------------------
    # Keep only fault simulations.
    # --------------------------------------------------------

    fault_mask = (
        y["event_family"].to_numpy()
        != "switching"
    )

    fault_indices = np.flatnonzero(
        fault_mask
    )

    y_fault = (
        y.loc[fault_mask]
        .reset_index(drop=True)
    )

    print(
        f"Fault samples: {len(fault_indices)}"
    )

    print(
        f"Fault simulations: "
        f"{y_fault['sample_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Simulation IDs
    # --------------------------------------------------------

    simulation_ids = (
        y_fault["sample_id"]
        .drop_duplicates()
        .to_numpy()
    )

    print(
        f"Unique fault simulations: "
        f"{len(simulation_ids)}"
    )

    groups = (
        y_fault["sample_id"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # First split:
    # 85% train+validation
    # 15% test
    # --------------------------------------------------------

    splitter_test = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEED,
    )

    train_val_fault_idx, test_fault_idx = next(
        splitter_test.split(
            np.zeros(len(y_fault)),
            groups=groups,
        )
    )

    # --------------------------------------------------------
    # Second split:
    #
    # 15% validation overall:
    #
    # 0.15 / 0.85
    # = 0.17647058823529413
    # --------------------------------------------------------

    splitter_val = GroupShuffleSplit(
        n_splits=1,
        test_size=0.17647058823529413,
        random_state=SEED,
    )

    train_fault_idx, val_fault_idx = next(
        splitter_val.split(
            np.zeros(
                len(train_val_fault_idx)
            ),
            groups=groups[
                train_val_fault_idx
            ],
        )
    )

    train_fault_idx = (
        train_val_fault_idx[
            train_fault_idx
        ]
    )

    val_fault_idx = (
        train_val_fault_idx[
            val_fault_idx
        ]
    )

    # --------------------------------------------------------
    # Convert back to ORIGINAL X indices.
    # --------------------------------------------------------

    train_idx = fault_indices[
        train_fault_idx
    ]

    val_idx = fault_indices[
        val_fault_idx
    ]

    test_idx = fault_indices[
        test_fault_idx
    ]

    # --------------------------------------------------------
    # Split statistics
    # --------------------------------------------------------

    train_simulations = (
        y_fault.iloc[
            train_fault_idx
        ]["sample_id"]
        .nunique()
    )

    val_simulations = (
        y_fault.iloc[
            val_fault_idx
        ]["sample_id"]
        .nunique()
    )

    test_simulations = (
        y_fault.iloc[
            test_fault_idx
        ]["sample_id"]
        .nunique()
    )

    print()
    print(
        f"Train simulations: "
        f"{train_simulations}"
    )

    print(
        f"Validation simulations: "
        f"{val_simulations}"
    )

    print(
        f"Test simulations: "
        f"{test_simulations}"
    )

    print()
    print(
        f"Train samples: "
        f"{len(train_idx)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_idx)}"
    )

    print(
        f"Test samples: "
        f"{len(test_idx)}"
    )

    # --------------------------------------------------------
    # Verify simulation-level disjointness.
    # --------------------------------------------------------

    train_sims = set(
        y_fault.iloc[
            train_fault_idx
        ]["sample_id"]
        .unique()
    )

    val_sims = set(
        y_fault.iloc[
            val_fault_idx
        ]["sample_id"]
        .unique()
    )

    test_sims = set(
        y_fault.iloc[
            test_fault_idx
        ]["sample_id"]
        .unique()
    )

    assert train_sims.isdisjoint(
        val_sims
    )

    assert train_sims.isdisjoint(
        test_sims
    )

    assert val_sims.isdisjoint(
        test_sims
    )

    print()
    print(
        "Simulation-level split check: PASSED"
    )

    # --------------------------------------------------------
    # Labels aligned with ORIGINAL X.
    # --------------------------------------------------------

    y_fault_present = (
        y["y_fault_present"]
        .to_numpy()
    )

    return (
        X,
        y_fault_present,
        y,
        {
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx,
        },
    )


# ============================================================
# Relay-specific indices
# ============================================================

def get_relay_indices(
    y,
    relay,
    split_indices,
):
    """
    Get original-X indices for one canonical relay.
    """

    dataset_relay_name = (
        RELAY_NAME_MAP[relay]
    )

    relay_mask = (
        y["sensor_name"].to_numpy()
        == dataset_relay_name
    )

    indices = {}

    for split_name, split_idx in split_indices.items():

        indices[split_name] = split_idx[
            relay_mask[split_idx]
        ]

    return indices


# ============================================================
# Matrix helper
# ============================================================

def metric_matrix(
    results: dict,
    metric: str,
) -> pd.DataFrame:

    matrix = pd.DataFrame(
        index=RELAY_ORDER,
        columns=RELAY_ORDER,
        dtype=float,
    )

    for source in RELAY_ORDER:
        for target in RELAY_ORDER:
            matrix.loc[
                source,
                target,
            ] = results[
                source
            ][
                target
            ][metric]

    return matrix


# ============================================================
# Heatmap
# ============================================================

def save_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_file: Path,
    fmt: str = ".3f",
):
    # --------------------------------------------------------
    # Display labels
    # --------------------------------------------------------

    relay_display_map = {
        "Bus1_Ln12A": "R1",
        "Bus1_Ln12B": "R2",
        "Bus2_Ln12A": "R3",
        "Bus2_Ln12B": "R4",
        "Bus2_Ln23A": "R5",
        "Bus2_Ln23B": "R6",
        "Bus3_Ln23A": "R7",
        "Bus3_Ln23B": "R8",
    }

    display_labels = [
        relay_display_map.get(
            relay,
            relay,
        )
        for relay in matrix.columns
    ]

    plt.figure(
        figsize=(11, 8),
    )

    ax = sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=fmt,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "label": title,
        },
    )

    # --------------------------------------------------------
    # X-axis: Target relay
    # --------------------------------------------------------

    ax.set_xlabel(
        "Target relay"
    )

    ax.set_xticklabels(
        display_labels,
        rotation=0,
    )

    # --------------------------------------------------------
    # Y-axis: Source relay
    # --------------------------------------------------------

    ax.set_ylabel(
        "Source relay"
    )

    ax.set_yticklabels(
        display_labels,
        rotation=0,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

# ============================================================
# Self / cross summary
# ============================================================

def calculate_self_cross_summary(
    results: dict,
) -> pd.DataFrame:

    metrics = [
        "accuracy",
        "f1",
        "auc",
        "fpr",
        "fnr",
    ]

    rows = []

    for metric in metrics:

        self_values = []

        cross_values = []

        for source in RELAY_ORDER:

            for target in RELAY_ORDER:

                value = results[
                    source
                ][
                    target
                ][metric]

                if source == target:
                    self_values.append(
                        value
                    )
                else:
                    cross_values.append(
                        value
                    )

        self_mean = float(
            np.mean(self_values)
        )

        cross_mean = float(
            np.mean(cross_values)
        )

        rows.append(
            {
                "metric": metric,
                "self_mean": self_mean,
                "cross_mean": cross_mean,
                "gap_self_minus_cross":
                    self_mean - cross_mean,
                "worst_cross":
                    (
                        max(cross_values)
                        if metric in ["fpr", "fnr"]
                        else min(cross_values)
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Threshold Table I
# ============================================================

def save_table_I(
    summary: pd.DataFrame,
    output_file: Path,
):
    """
    Save a compact LaTeX self/cross threshold table.
    """

    table = summary.copy()

    table["metric"] = (
        table["metric"]
        .str.upper()
    )

    table = table.rename(
        columns={
            "metric": "Metric",
            "self_mean": "Self",
            "cross_mean": "Cross",
            "gap_self_minus_cross": "Gap",
            "worst_cross": "Worst Cross",
        }
    )

    latex = table.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        escape=True,
        caption=(
            "Self- and cross-relay performance using "
            "source-validation-selected thresholds."
        ),
        label="tab:threshold_self_cross",
    )

    output_file.write_text(
        latex
    )


# ============================================================
# Analysis outputs
# ============================================================

def generate_analysis_outputs(
    results: dict,
    thresholds: list,
    results_dir: Path,
):

    print()
    print("=" * 70)
    print("GENERATING THRESHOLD ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Threshold table
    # --------------------------------------------------------

    thresholds_df = pd.DataFrame(
        thresholds
    )

    thresholds_df.to_csv(
        results_dir / "thresholds.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metric matrices
    # --------------------------------------------------------

    matrices = {}

    for metric in [
        "accuracy",
        "f1",
        "auc",
        "fpr",
        "fnr",
    ]:

        matrix = metric_matrix(
            results,
            metric,
        )

        matrices[metric] = matrix

        matrix.to_csv(
            results_dir
            / f"{metric}_matrix.csv"
        )

    # --------------------------------------------------------
    # Heatmaps
    # --------------------------------------------------------

    save_heatmap(
        matrices["fpr"],
        "Threshold-based FPR",
        results_dir / "fpr_heatmap.png",
    )

    save_heatmap(
        matrices["fnr"],
        "Threshold-based FNR",
        results_dir / "fnr_heatmap.png",
    )

    # --------------------------------------------------------
    # Self/cross summary
    # --------------------------------------------------------

    summary = calculate_self_cross_summary(
        results
    )

    summary.to_csv(
        results_dir / "self_cross_summary.csv",
        index=False,
    )

    save_table_I(
        summary,
        results_dir / "table_I_threshold.tex",
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"Threshold analysis outputs: "
        f"{results_dir}"
    )


# ============================================================
# Main analysis function
# ============================================================

def run_threshold_analysis(
    run_dir: Path,
) -> Path:
    """
    Run the complete threshold evaluation and analysis
    for an existing canonical training run.

    Parameters
    ----------
    run_dir:
        Existing canonical experiment run directory.

    Returns
    -------
    Path
        Threshold analysis output directory.
    """

    run_dir = Path(
        run_dir
    ).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory does not exist:\n"
            f"{run_dir}"
        )

    results_dir = (
        run_dir
        / "analysis"
        / "threshold"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        "SOURCE-VALIDATION THRESHOLD "
        "PAIRWISE EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Training run: {run_dir}"
    )

    print(
        f"Results directory: {results_dir}"
    )

    print(
        f"Threshold method: "
        f"{THRESHOLD_METHOD}"
    )

    # --------------------------------------------------------
    # Load dataset and reconstruct common split
    # --------------------------------------------------------

    X, y_fault_present, y, split_indices = (
        prepare_datasets()
    )

    # --------------------------------------------------------
    # Validate relay mapping
    # --------------------------------------------------------

    available_relays = sorted(
        y["sensor_name"]
        .unique()
        .tolist()
    )

    print()
    print(
        "Available dataset relays:"
    )

    for relay in available_relays:
        print(
            f"  {relay}"
        )

    print()
    print(
        "Relay mapping:"
    )

    for canonical_name in RELAY_ORDER:

        dataset_name = (
            RELAY_NAME_MAP[
                canonical_name
            ]
        )

        if dataset_name not in available_relays:
            raise ValueError(
                f"Dataset relay '{dataset_name}' "
                f"for canonical relay "
                f"'{canonical_name}' "
                f"was not found."
            )

        print(
            f"  {canonical_name:<18} "
            f"-> {dataset_name}"
        )

    # --------------------------------------------------------
    # Build relay-specific indices
    # --------------------------------------------------------

    relay_indices = {}

    for relay in RELAY_ORDER:

        relay_indices[relay] = (
            get_relay_indices(
                y,
                relay,
                split_indices,
            )
        )

        print(
            f"{relay}: "
            f"train="
            f"{len(relay_indices[relay]['train'])}, "
            f"val="
            f"{len(relay_indices[relay]['validation'])}, "
            f"test="
            f"{len(relay_indices[relay]['test'])}"
        )

    print()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_results = {}

    thresholds = []

    # --------------------------------------------------------
    # Evaluate every source relay
    # --------------------------------------------------------

    for source_relay in RELAY_ORDER:

        print()
        print("=" * 70)
        print(
            f"SOURCE RELAY: "
            f"{source_relay}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Load source model
        # ----------------------------------------------------

        model, model_path = load_model(
            source_relay,
            run_dir,
        )

        # ----------------------------------------------------
        # Source validation
        # ----------------------------------------------------

        validation_dataset = RelayDataset(
            X,
            y_fault_present,
            relay_indices[
                source_relay
            ]["validation"],
        )

        validation_logits, validation_labels = (
            predict_dataset(
                model,
                validation_dataset,
            )
        )

        # ----------------------------------------------------
        # Select threshold
        # ----------------------------------------------------

        threshold_info = (
            select_youden_threshold(
                validation_labels,
                validation_logits,
            )
        )

        threshold = (
            threshold_info["threshold"]
        )

        print()
        print(
            f"Selected threshold: "
            f"{threshold:.6f}"
        )

        print(
            f"Youden J: "
            f"{threshold_info['youden_j']:.6f}"
        )

        print(
            f"Validation TPR: "
            f"{threshold_info['validation_tpr']:.6f}"
        )

        print(
            f"Validation FPR: "
            f"{threshold_info['validation_fpr']:.6f}"
        )

        print(
            f"Validation FNR: "
            f"{threshold_info['validation_fnr']:.6f}"
        )

        thresholds.append(
            {
                "source_relay":
                    source_relay,
                "logit_threshold":
                    threshold,
                "youden_j":
                    threshold_info[
                        "youden_j"
                    ],
                "validation_tpr":
                    threshold_info[
                        "validation_tpr"
                    ],
                "validation_fpr":
                    threshold_info[
                        "validation_fpr"
                    ],
                "validation_fnr":
                    threshold_info[
                        "validation_fnr"
                    ],
                "model_path":
                    str(model_path),
            }
        )

        # ----------------------------------------------------
        # Evaluate frozen threshold on every target relay
        # ----------------------------------------------------

        all_results[
            source_relay
        ] = {}

        for target_relay in RELAY_ORDER:

            print(
                f"  Evaluating "
                f"{source_relay} -> "
                f"{target_relay}"
            )

            test_dataset = RelayDataset(
                X,
                y_fault_present,
                relay_indices[
                    target_relay
                ]["test"],
            )

            logits, labels = (
                predict_dataset(
                    model,
                    test_dataset,
                )
            )

            metrics = calculate_metrics(
                labels,
                logits,
                threshold,
            )

            all_results[
                source_relay
            ][
                target_relay
            ] = metrics

        # ----------------------------------------------------
        # Free GPU memory
        # ----------------------------------------------------

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Save threshold results
    # ========================================================

    thresholds_df = pd.DataFrame(
        thresholds
    )

    thresholds_df.to_csv(
        results_dir / "thresholds.csv",
        index=False,
    )

    # ========================================================
    # Save results.json
    # ========================================================

    output = {
        "config": {
            "model":
                MODEL_NAME,
            "methodology":
                "pairwise",
            "seed":
                SEED,
            "threshold_method":
                THRESHOLD_METHOD,
            "threshold_selection":
                "source_validation_only",
            "threshold_frozen_for":
                "all_target_test_relays",
            "training_run":
                str(run_dir),
            "batch_size":
                BATCH_SIZE,
            "num_workers":
                NUM_WORKERS,
            "split":
                {
                    "test_fraction":
                        0.15,
                    "validation_fraction":
                        0.15,
                    "group":
                        "sample_id",
                },
        },
        "training": {},
        "thresholds": thresholds,
        "evaluation": all_results,
    }

    results_file = (
        results_dir
        / "results.json"
    )

    with open(
        results_file,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # Generate all analysis outputs
    # ========================================================

    generate_analysis_outputs(
        all_results,
        thresholds,
        results_dir,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results: "
        f"{results_file}"
    )

    print(
        f"Thresholds: "
        f"{results_dir / 'thresholds.csv'}"
    )

    print(
        f"Total source-target evaluations: "
        f"{len(RELAY_ORDER) ** 2}"
    )

    print()
    print(
        f"Output directory:"
    )

    print(
        f"  {results_dir}"
    )

    print()

    return results_dir