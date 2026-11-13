#!/usr/bin/env python3
"""
Analyze pairwise cross-relay evaluation results.
The canonical source of performance metrics is:
    results.json
Expected structure:
    results.json
    ├── config
    ├── training
    ├── evaluation
    │   └── source_relay
    │       └── target_relay
    │           ├── accuracy
    │           ├── f1
    │           ├── auc
    │           ├── fpr
    │           ├── fnr
    │           ├── tn
    │           ├── fp
    │           ├── fn
    │           └── tp
    └── summary
Rows:
    Training/source relay
Columns:
    Test/target relay
Diagonal:
    Within-relay / self performance
Off-diagonal:
    Cross-relay performance
The evaluation protocol uses a common
simulation-level train/validation/test split
across all relays.
"""
import argparse
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import numpy as np
import pandas as pd
from .plotting.style import (
    apply_axis_style,
    format_score,
    relay_display_name,
    save_figure,
)
# ============================================================
# Configuration
# ============================================================
EXPERIMENT_NAME = (
    "compact_mlp_pairwise_default"
)
EXPERIMENT_DIR = (
    Path("results")
    / EXPERIMENT_NAME
)
DEFAULT_RESULTS_DIR = (
    EXPERIMENT_DIR
    / "latest"
)
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
METRICS = [
    "accuracy",
    "f1",
    "auc",
    "fpr",
    "fnr",
]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1-score",
    "auc": "ROC AUC",
    "fpr": "FPR",
    "fnr": "FNR",
}
# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze canonical pairwise "
            "evaluation results."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Run directory containing results.json. "
            "Defaults to results/"
            "compact_mlp_pairwise_default/latest."
        ),
    )
    return parser.parse_args()
# ============================================================
# Load results
# ============================================================
def load_results(
    results_file: Path,
):
    if not results_file.exists():
        raise FileNotFoundError(
            f"Could not find results file:\n"
            f"{results_file}"
        )
    print("=" * 70)
    print("LOADING RESULTS")
    print("=" * 70)
    print(
        f"Results file: {results_file}"
    )
    print()
    with open(
        results_file,
        "r",
    ) as f:
        results = json.load(f)
    if not isinstance(
        results,
        dict,
    ):
        raise TypeError(
            "results.json must contain "
            "a JSON object/dictionary."
        )
    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The normal 0.5-threshold results.json contains:
    #   config
    #   training
    #   evaluation
    #   summary
    #
    # The Youden-J results.json under:
    #   analysis/threshold/results.json
    #
    # may contain only the evaluation structure.
    #
    # For heatmap generation, only "evaluation" is required.
    # --------------------------------------------------------
    required_keys = {
        "evaluation",
    }
    missing = (
        required_keys
        - set(results.keys())
    )
    if missing:
        raise KeyError(
            "results.json is missing "
            f"required top-level keys: {missing}"
        )
    print(
        "Top-level keys:",
        list(results.keys()),
    )
    print()
    return results
# ============================================================
# Validate evaluation structure
# ============================================================
def validate_evaluation_structure(
    results,
):
    evaluation = results[
        "evaluation"
    ]
    if not isinstance(
        evaluation,
        dict,
    ):
        raise TypeError(
            "results['evaluation'] "
            "must be a dictionary."
        )
    print("=" * 70)
    print("VALIDATING EVALUATION STRUCTURE")
    print("=" * 70)
    for relay in RELAY_ORDER:
        if relay not in evaluation:
            raise KeyError(
                f"Source relay '{relay}' "
                "not found in results."
            )
        for target in RELAY_ORDER:
            if (
                target
                not in evaluation[relay]
            ):
                raise KeyError(
                    f"Target relay '{target}' "
                    f"not found for source "
                    f"relay '{relay}'."
                )
            entry = evaluation[
                relay
            ][
                target
            ]
            for metric in METRICS:
                if metric not in entry:
                    raise KeyError(
                        f"Metric '{metric}' "
                        f"missing for "
                        f"{relay} -> {target}."
                    )
    print(
        "Evaluation structure: PASSED"
    )
    print(
        f"Number of relays: "
        f"{len(RELAY_ORDER)}"
    )
    print(
        f"Number of source-target pairs: "
        f"{len(RELAY_ORDER) ** 2}"
    )
    print()
# ============================================================
# Matrix extraction
# ============================================================
def extract_metric_matrix(
    results,
    metric,
    relay_order,
):
    evaluation = results[
        "evaluation"
    ]
    matrix = np.zeros(
        (
            len(relay_order),
            len(relay_order),
        ),
        dtype=float,
    )
    for i, source_relay in enumerate(
        relay_order
    ):
        for j, target_relay in enumerate(
            relay_order
        ):
            value = (
                evaluation[
                    source_relay
                ][
                    target_relay
                ][
                    metric
                ]
            )
            matrix[i, j] = float(
                value
            )
    return matrix
def matrix_to_dataframe(
    matrix,
    relay_order,
):
    return pd.DataFrame(
        matrix,
        index=relay_order,
        columns=relay_order,
    )
# ============================================================
# Self / cross helpers
# ============================================================
def get_diagonal_values(
    matrix,
):
    return np.diag(
        matrix
    )
def get_cross_values(
    matrix,
):
    mask = ~np.eye(
        matrix.shape[0],
        dtype=bool,
    )
    return matrix[
        mask
    ]
def calculate_self_cross_statistics(
    matrix,
    metric,
):
    self_values = (
        get_diagonal_values(
            matrix
        )
    )
    cross_values = (
        get_cross_values(
            matrix
        )
    )
    mean_self = np.mean(
        self_values
    )
    mean_cross = np.mean(
        cross_values
    )
    std_self = np.std(
        self_values
    )
    std_cross = np.std(
        cross_values
    )
    abs_gap = abs(
        mean_self
        - mean_cross
    )
    if metric in {
        "accuracy",
        "f1",
        "auc",
    }:
        worst_cross = np.min(
            cross_values
        )
    elif metric in {
        "fpr",
        "fnr",
    }:
        worst_cross = np.max(
            cross_values
        )
    else:
        raise ValueError(
            f"Unknown metric: {metric}"
        )
    return {
        "metric": metric,
        "mean_self": mean_self,
        "mean_cross": mean_cross,
        "abs_gap": abs_gap,
        "worst_cross": worst_cross,
        "std_self": std_self,
        "std_cross": std_cross,
    }
# ============================================================
# Save metric matrices
# ============================================================
def save_metric_matrices(
    results,
    output_dir,
):
    print("=" * 70)
    print("EXTRACTING METRIC MATRICES")
    print("=" * 70)
    matrices = {}
    for metric in METRICS:
        matrix = extract_metric_matrix(
            results,
            metric,
            RELAY_ORDER,
        )
        matrices[
            metric
        ] = matrix
        df = matrix_to_dataframe(
            matrix,
            RELAY_ORDER,
        )
        output_file = (
            output_dir
            / f"{metric}_matrix.csv"
        )
        df.to_csv(
            output_file
        )
        print(
            f"{metric.upper():<10} -> "
            f"{output_file}"
        )
    print()
    return matrices
# ============================================================
# Table I
# ============================================================
def create_table_I(
    matrices,
    output_dir,
):
    rows = []
    for metric in METRICS:
        stats = (
            calculate_self_cross_statistics(
                matrices[metric],
                metric,
            )
        )
        rows.append(
            {
                "Metric":
                    METRIC_LABELS[
                        metric
                    ],
                "Mean Self":
                    stats[
                        "mean_self"
                    ],
                "Mean Cross":
                    stats[
                        "mean_cross"
                    ],
                "Abs. Gap":
                    stats[
                        "abs_gap"
                    ],
                "Worst Cross":
                    stats[
                        "worst_cross"
                    ],
            }
        )
    table = pd.DataFrame(
        rows
    )
    csv_file = (
        output_dir
        / "table_I_self_cross.csv"
    )
    table.to_csv(
        csv_file,
        index=False,
    )
    return table
def print_table_I(
    table,
):
    print("=" * 70)
    print(
        "TABLE I — SELF VS CROSS-RELAY PERFORMANCE"
    )
    print("=" * 70)
    display_table = (
        table.copy()
    )
    numeric_columns = [
        "Mean Self",
        "Mean Cross",
        "Abs. Gap",
        "Worst Cross",
    ]
    for col in numeric_columns:
        display_table[
            col
        ] = display_table[
            col
        ].map(
            lambda x:
                f"{x:.4f}"
        )
    print(
        display_table.to_string(
            index=False
        )
    )
    print()
def save_table_I_latex(
    table,
    output_dir,
):
    latex_file = (
        output_dir
        / "table_I.tex"
    )
    lines = []
    lines.append(
        r"\begin{table}[t]"
    )
    lines.append(
        r"\centering"
    )
    lines.append(
        r"\caption{Summary of within-relay and "
        r"cross-relay performance for the "
        r"MLP baseline. The gap is defined "
        r"as the absolute difference between "
        r"mean self and mean cross.}"
    )
    lines.append(
        r"\label{tab:summary}"
    )
    lines.append(
        r"\renewcommand{\arraystretch}{1.2}"
    )
    lines.append(
        r"\begin{tabular}{lcccc}"
    )
    lines.append(
        r"\toprule"
    )
    lines.append(
        r"\textbf{Metric} & "
        r"\textbf{Mean Self} & "
        r"\textbf{Mean Cross} & "
        r"\textbf{Abs. Gap} & "
        r"\textbf{Worst Cross} \\\\\\\\"
    )
    lines.append(
        r"\midrule"
    )
    for _, row in table.iterrows():
        lines.append(
            f"{row['Metric']} & "
            f"{row['Mean Self']:.4f} & "
            f"{row['Mean Cross']:.4f} & "
            f"{row['Abs. Gap']:.4f} & "
            f"{row['Worst Cross']:.4f} "
            r"\\\\"
        )
    lines.append(
        r"\bottomrule"
    )
    lines.append(
        r"\end{tabular}"
    )
    lines.append(
        r"\end{table}"
    )
    latex_text = "\n".join(
        lines
    )
    with open(
        latex_file,
        "w",
    ) as f:
        f.write(
            latex_text
        )
    return latex_file
# ============================================================
# Detailed self / cross summary
# ============================================================
def create_self_cross_summary(
    matrices,
    output_dir,
):
    rows = []
    for metric in METRICS:
        matrix = matrices[
            metric
        ]
        self_values = (
            get_diagonal_values(
                matrix
            )
        )
        cross_values = (
            get_cross_values(
                matrix
            )
        )
        rows.append(
            {
                "metric": metric,
                "self_mean":
                    np.mean(
                        self_values
                    ),
                "self_std":
                    np.std(
                        self_values
                    ),
                "self_min":
                    np.min(
                        self_values
                    ),
                "self_max":
                    np.max(
                        self_values
                    ),
                "cross_mean":
                    np.mean(
                        cross_values
                    ),
                "cross_std":
                    np.std(
                        cross_values
                    ),
                "cross_min":
                    np.min(
                        cross_values
                    ),
                "cross_max":
                    np.max(
                        cross_values
                    ),
                "self_cross_difference":
                    np.mean(
                        self_values
                    )
                    - np.mean(
                        cross_values
                    ),
                "absolute_gap":
                    abs(
                        np.mean(
                            self_values
                        )
                        - np.mean(
                            cross_values
                        )
                    ),
            }
        )
    summary = pd.DataFrame(
        rows
    )
    output_file = (
        output_dir
        / "self_cross_summary.csv"
    )
    summary.to_csv(
        output_file,
        index=False,
    )
    return summary
# ============================================================
# Heatmaps
# ============================================================
def create_heatmap(
    matrix,
    title,
    output_file,
):
    """
    Create a publication-style relay heatmap.
    Rows    : Training/source relay
    Columns : Test/target relay
    The matrix values are displayed using the project-specific
    score formatting and axis styling. Diagonal cells are
    outlined to indicate self-relay evaluation.
    """
    data_min = matrix.min()
    data_max = matrix.max()
    data_range = data_max - data_min
    if data_range == 0:
        padding = 0.05
    else:
        padding = 0.1 * data_range
    vmin = max(
        0.0,
        data_min - padding,
    )
    vmax = min(
        1.0,
        data_max + padding,
    )
    fig, ax = plt.subplots(
        figsize=(8.2, 7.4),
    )
    image = ax.imshow(
        matrix,
        cmap="Blues",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------
    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(
        title,
        fontsize=16,
    )
    cbar.ax.tick_params(
        labelsize=14,
    )
    # --------------------------------------------------------
    # Relay labels
    # --------------------------------------------------------
    display_relays = [
        relay_display_name(relay)
        for relay in RELAY_ORDER
    ]
    ax.set_xticks(
        np.arange(len(RELAY_ORDER))
    )
    ax.set_yticks(
        np.arange(len(RELAY_ORDER))
    )
    ax.set_xticklabels(
        display_relays,
        rotation=0,
        ha="center",
    )
    ax.set_yticklabels(
        display_relays,
    )
    ax.set_xlabel(
        "Test Relay",
        fontsize=16,
    )
    ax.set_ylabel(
        "Training Relay",
        fontsize=16,
    )
    # --------------------------------------------------------
    # Cell annotations
    # --------------------------------------------------------
    norm = plt.Normalize(
        vmin,
        vmax,
    )
    cmap_object = plt.get_cmap(
        "Blues"
    )
    for i in range(
        len(RELAY_ORDER)
    ):
        for j in range(
            len(RELAY_ORDER)
        ):
            value = matrix[i, j]
            rgba = cmap_object(
                norm(value)
            )
            r, g, b = rgba[:3]
            luminance = (
                0.2126 * r
                + 0.7152 * g
                + 0.0722 * b
            )
            text_color = (
                "black"
                if luminance > 0.55
                else "white"
            )
            ax.text(
                j,
                i,
                format_score(value),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="normal",
                color=text_color,
            )
            # Highlight diagonal/self-relay cells.
            if i == j:
                ax.add_patch(
                    plt.Rectangle(
                        (
                            j - 0.5,
                            i - 0.5,
                        ),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )
    # --------------------------------------------------------
    # Project-specific axis styling
    # --------------------------------------------------------
    apply_axis_style(ax)
    ax.spines[
        "top"
    ].set_visible(True)
    ax.spines[
        "right"
    ].set_visible(True)
    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    save_figure(
        fig,
        output_file,
    )
    plt.close(fig)
def create_all_heatmaps(
    matrices,
    output_dir,
):
    print("=" * 70)
    print("CREATING HEATMAPS")
    print("=" * 70)
    # --------------------------------------------------------
    # FPR
    # --------------------------------------------------------
    fpr_file = (
        output_dir
        / "fpr_heatmap.png"
    )
    create_heatmap(
        matrices["fpr"],
        "False Positive Rate",
        fpr_file,
    )
    print(
        f"FPR heatmap -> {fpr_file}"
    )
    # --------------------------------------------------------
    # FNR
    # --------------------------------------------------------
    fnr_file = (
        output_dir
        / "fnr_heatmap.png"
    )
    create_heatmap(
        matrices["fnr"],
        "False Negative Rate",
        fnr_file,
    )
    print(
        f"FNR heatmap -> {fnr_file}"
    )
    print()
# ============================================================
# Print matrices
# ============================================================
def print_matrix_summary(
    matrices,
):
    for metric in METRICS:
        print("=" * 70)
        print(
            f"{METRIC_LABELS[metric].upper()} MATRIX"
        )
        print("=" * 70)
        df = matrix_to_dataframe(
            matrices[metric],
            RELAY_ORDER,
        )
        print(
            df.to_string(
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )
        print()
# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    results_dir = (
        args.results_dir
        .resolve()
    )
    # --------------------------------------------------------
    # Use the Youden-J results, not the default 0.5-threshold
    # results.
    # --------------------------------------------------------
    results_file = (
        results_dir
        / "analysis"
        / "threshold"
        / "results.json"
    )
    # --------------------------------------------------------
    # Save the Youden-J heatmaps alongside the Youden results.
    # --------------------------------------------------------
    output_dir = (
        results_dir
        / "analysis"
        / "threshold"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    print()
    print("=" * 70)
    print("PAIRWISE RESULTS ANALYSIS")
    print("=" * 70)
    print()
    print(
        "Results directory:"
    )
    print(
        f"  {results_dir}"
    )
    print()
    print(
        "Output directory:"
    )
    print(
        f"  {output_dir}"
    )
    print()
    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------
    results = load_results(
        results_file
    )
    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------
    validate_evaluation_structure(
        results
    )
    # --------------------------------------------------------
    # Extract matrices
    # --------------------------------------------------------
    matrices = save_metric_matrices(
        results,
        output_dir,
    )
    # --------------------------------------------------------
    # Print matrices
    # --------------------------------------------------------
    print_matrix_summary(
        matrices
    )
    # --------------------------------------------------------
    # Table I
    # --------------------------------------------------------
    table_I = create_table_I(
        matrices,
        output_dir,
    )
    print_table_I(
        table_I
    )
    latex_file = save_table_I_latex(
        table_I,
        output_dir,
    )
    print(
        f"Table I LaTeX -> {latex_file}"
    )
    print()
    # --------------------------------------------------------
    # Detailed summary
    # --------------------------------------------------------
    summary = (
        create_self_cross_summary(
            matrices,
            output_dir,
        )
    )
    summary_file = (
        output_dir
        / "self_cross_summary.csv"
    )
    print(
        f"Self/cross summary -> "
        f"{summary_file}"
    )
    print()
    # --------------------------------------------------------
    # FPR / FNR heatmaps
    # --------------------------------------------------------
    create_all_heatmaps(
        matrices,
        output_dir,
    )
    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print()
    for file in sorted(
        output_dir.iterdir()
    ):
        if file.is_file():
            print(
                f"  {file.name}"
            )
    print()
    print("=" * 70)
if __name__ == "__main__":
    main()
