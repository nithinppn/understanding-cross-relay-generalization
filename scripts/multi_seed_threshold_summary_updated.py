#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from .analyze_pairwise_results import create_heatmap
from .plotting.style import apply_axis_style, save_figure
# ============================================================
# Configuration
# ============================================================
RESULTS_ROOT = Path(
    "results/compact_mlp_pairwise_default"
)
TARGET_SEEDS = {5, 10, 25, 42, 100}
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
# ============================================================
# Find threshold-analysis runs
# ============================================================
def find_runs():
    runs = []
    seen_run_paths = set()
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        # Resolve symlinks so the same physical run is counted once.
        resolved_run_dir = run_dir.resolve()
        if resolved_run_dir in seen_run_paths:
            continue
        seen_run_paths.add(resolved_run_dir)
        run_dir = resolved_run_dir
        config_file = run_dir / "config.yaml"
        threshold_dir = (
            run_dir
            / "analysis"
            / "threshold"
        )
        threshold_results = (
            threshold_dir
            / "results.json"
        )
        if not config_file.exists():
            continue
        if not threshold_results.exists():
            continue
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        seed = config.get("seed")
        if seed in TARGET_SEEDS:
            runs.append({
                "seed": seed,
                "run_dir": run_dir,
                "threshold_dir": threshold_dir,
                "results_file": threshold_results,
            })
    runs.sort(
        key=lambda x: x["seed"]
    )
    return runs
# ============================================================
# Load threshold results
# ============================================================
def load_threshold_results(run):
    with open(
        run["results_file"],
        "r",
    ) as f:
        data = json.load(f)
    return data
# ============================================================
# Validate threshold protocol
# ============================================================
def validate_protocol(data, seed):
    config = data.get(
        "config",
        {},
    )
    print(
        f"\nSeed {seed} protocol:"
    )
    print(
        f"  Threshold method: "
        f"{config.get('threshold_method')}"
    )
    print(
        f"  Threshold selection: "
        f"{config.get('threshold_selection')}"
    )
    print(
        f"  Frozen for: "
        f"{config.get('threshold_frozen_for')}"
    )
    print(
        f"  Split group: "
        f"{config.get('split', {}).get('group')}"
    )
    if config.get(
        "threshold_method"
    ) != "youden_j":
        raise ValueError(
            f"Seed {seed}: "
            "threshold method is not Youden J."
        )
    if config.get(
        "threshold_selection"
    ) != "source_validation_only":
        raise ValueError(
            f"Seed {seed}: "
            "threshold was not selected "
            "from source validation only."
        )
    if config.get(
        "threshold_frozen_for"
    ) != "all_target_test_relays":
        raise ValueError(
            f"Seed {seed}: "
            "threshold was not frozen "
            "for all target test relays."
        )
# ============================================================
# Extract 8 x 8 evaluation matrix
# ============================================================
def get_evaluation_matrix(data):
    evaluation = data["evaluation"]
    rows = []
    for source_relay, targets in evaluation.items():
        for target_relay, metrics in targets.items():
            row = {
                "source": source_relay,
                "target": target_relay,
            }
            for metric in METRICS:
                row[metric] = metrics[
                    metric
                ]
            rows.append(row)
    return pd.DataFrame(rows)
# ============================================================
# Calculate Self / Cross / Worst Cross
# ============================================================
def get_pairwise_metric_matrix(df, metric):
    """Return an 8x8 matrix ordered by training and test relay."""
    matrix_df = df.pivot(
        index="source",
        columns="target",
        values=metric,
    ).reindex(index=RELAY_ORDER, columns=RELAY_ORDER)
    if matrix_df.shape != (8, 8) or matrix_df.isna().any().any():
        raise ValueError(
            f"Seed results do not contain a complete 8x8 matrix for {metric}."
        )
    matrix = matrix_df.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values found in {metric} matrix.")
    return matrix
def calculate_summary(df):
    self_df = df[
        df["source"] == df["target"]
    ]
    cross_df = df[
        df["source"] != df["target"]
    ]
    assert len(self_df) == 8, (
        f"Expected 8 self evaluations, "
        f"found {len(self_df)}"
    )
    assert len(cross_df) == 56, (
        f"Expected 56 cross evaluations, "
        f"found {len(cross_df)}"
    )
    summary = {}
    for metric in METRICS:
        self_values = (
            self_df[metric]
            .to_numpy(dtype=float)
        )
        cross_values = (
            cross_df[metric]
            .to_numpy(dtype=float)
        )
        self_mean = np.mean(
            self_values
        )
        cross_mean = np.mean(
            cross_values
        )
        # Same definition as your
        # existing threshold analysis.
        if metric in [
            "fpr",
            "fnr",
        ]:
            worst_cross = np.max(
                cross_values
            )
        else:
            worst_cross = np.min(
                cross_values
            )
        summary[metric] = {
            "self": self_mean,
            "cross": cross_mean,
            "worst_cross": worst_cross,
        }
    return summary
# ============================================================
# Create PPT charts
# ============================================================
def create_chart(
    final_df,
    metrics,
    labels,
    title,
    ylabel,
    filename,
    output_dir,
):
    categories = [
        (
            "self_mean",
            "self_std",
            "Self",
        ),
        (
            "cross_mean",
            "cross_std",
            "Cross",
        ),
        (
            "worst_cross_mean",
            "worst_cross_std",
            "Worst Cross",
        ),
    ]
    x = np.arange(
        len(metrics)
    )
    width = 0.24
    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )
    # Inherit typography and Type 42 PDF embedding from plotting.style.
    # ========================================================
    # Create bars
    # ========================================================
    for i, (
        mean_col,
        std_col,
        label,
    ) in enumerate(categories):
        subset = final_df[
            final_df["metric"].isin(metrics)
        ].copy()
        # Keep requested metric order
        subset["metric"] = pd.Categorical(
            subset["metric"],
            categories=metrics,
            ordered=True,
        )
        subset = subset.sort_values(
            "metric"
        )
        means = subset[
            mean_col
        ].to_numpy(dtype=float)
        stds = subset[
            std_col
        ].to_numpy(dtype=float)
        # ----------------------------------------------------
        # Bars + standard deviation error bars
        # ----------------------------------------------------
        bars = ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=5,
            label=label,
        )
        # ----------------------------------------------------
        # Add mean value above each bar
        # ----------------------------------------------------
        for bar, mean, std in zip(
            bars,
            means,
            stds,
        ):
            # Position above the error bar
            label_y = (
                bar.get_height()
                + std
                + 0.015
            )
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                label_y,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=14,
            )
    # ========================================================
    # X-axis
    # ========================================================
    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        labels
    )
    # ========================================================
    # Y-axis
    # ========================================================
    ax.set_ylabel(
        ylabel
    )
    ax.set_title(
        title
    )
    # ========================================================
    # Determine upper limit
    # ========================================================
    max_with_error = 0
    for (
        mean_col,
        std_col,
        _,
    ) in categories:
        subset = final_df[
            final_df["metric"].isin(metrics)
        ]
        values = (
            subset[mean_col]
            .to_numpy(dtype=float)
            +
            subset[std_col]
            .to_numpy(dtype=float)
        )
        if len(values) > 0:
            max_with_error = max(
                max_with_error,
                np.max(values),
            )
    # --------------------------------------------------------
    # Add space for value labels
    # --------------------------------------------------------
    if max_with_error <= 0.9:
        upper_limit = min(
            1.0,
            max_with_error + 0.12,
        )
    else:
        # Needed for values such as
        # worst-cross FPR ≈ 0.916
        upper_limit = 1.08
    ax.set_ylim(
        0,
        upper_limit,
    )
    # ========================================================
    # Grid
    # ========================================================
    ax.grid(
        axis="y",
        alpha=0.3,
    )
    # Keep grid behind bars
    ax.set_axisbelow(
        True
    )
    # ========================================================
    # Legend
    # ========================================================
    ax.legend(
        frameon=False
    )
    # ========================================================
    # Apply publication styling
    # ========================================================
    apply_axis_style(ax)
    fig.tight_layout()
    # ========================================================
    # Save
    # ========================================================
    output_file = (
        output_dir
        / filename
    )
    save_figure(fig, output_file)
    print(
        f"  Saved chart: {output_file} and {output_file.with_suffix('.pdf')}"
    )
# ============================================================
# Main
# ============================================================
def main():
    print("=" * 90)
    print(
        "MULTI-SEED YOUDEN-FROZEN "
        "PAIRWISE AGGREGATION"
    )
    print("=" * 90)
    runs = find_runs()
    print(
        "\nThreshold-analysis runs found:"
    )
    for run in runs:
        print(
            f"  Seed {run['seed']:>3}  |  "
            f"{run['run_dir']}"
        )
    found_seeds = {
        run["seed"]
        for run in runs
    }
    missing_seeds = (
        TARGET_SEEDS
        - found_seeds
    )
    duplicate_seeds = sorted(
        seed for seed in TARGET_SEEDS
        if sum(run["seed"] == seed for run in runs) > 1
    )
    if missing_seeds or duplicate_seeds:
        raise RuntimeError(
            "Cannot calculate a five-seed aggregate. "
            f"Missing seeds: {sorted(missing_seeds)}; "
            f"duplicate seeds: {duplicate_seeds}."
        )
    if not runs:
        raise RuntimeError(
            "No Youden threshold-analysis "
            "results were found."
        )
    # ========================================================
    # Per-seed summaries
    # ========================================================
    per_seed_rows = []
    # Collect each seed's complete pairwise matrices before aggregating.
    heatmap_matrices = {"fpr": [], "fnr": []}
    for run in runs:
        seed = run["seed"]
        print("\n")
        print("-" * 90)
        print(
            f"PROCESSING SEED {seed}"
        )
        print(
            f"File: "
            f"{run['results_file']}"
        )
        print("-" * 90)
        data = load_threshold_results(
            run
        )
        # ----------------------------------------------------
        # Verify this really is the
        # frozen Youden protocol.
        # ----------------------------------------------------
        validate_protocol(
            data,
            seed,
        )
        df = get_evaluation_matrix(
            data
        )
        for metric in ("fpr", "fnr"):
            heatmap_matrices[metric].append(
                get_pairwise_metric_matrix(df, metric)
            )
        summary = calculate_summary(
            df
        )
        for metric in METRICS:
            per_seed_rows.append({
                "seed": seed,
                "metric": metric,
                "self":
                    summary[metric]["self"],
                "cross":
                    summary[metric]["cross"],
                "worst_cross":
                    summary[metric][
                        "worst_cross"
                    ],
            })
    per_seed_df = pd.DataFrame(
        per_seed_rows
    )
    # ========================================================
    # Mean ± standard deviation
    # ========================================================
    final_rows = []
    for metric in METRICS:
        subset = per_seed_df[
            per_seed_df["metric"]
            == metric
        ]
        final_rows.append({
            "metric":
                metric,
            "self_mean":
                subset["self"].mean(),
            "self_std":
                subset["self"].std(
                    ddof=1
                ),
            "cross_mean":
                subset["cross"].mean(),
            "cross_std":
                subset["cross"].std(
                    ddof=1
                ),
            "worst_cross_mean":
                subset[
                    "worst_cross"
                ].mean(),
            "worst_cross_std":
                subset[
                    "worst_cross"
                ].std(
                    ddof=1
                ),
        })
    final_df = pd.DataFrame(
        final_rows
    )
    # ========================================================
    # Print per-seed values
    # ========================================================
    print("\n")
    print("=" * 100)
    print(
        "PER-SEED RESULTS"
    )
    print("=" * 100)
    for metric in METRICS:
        print(
            f"\n{metric.upper()}"
        )
        subset = per_seed_df[
            per_seed_df["metric"]
            == metric
        ]
        for _, row in subset.iterrows():
            print(
                f"  Seed {int(row['seed']):>3}: "
                f"Self={row['self']:.4f} | "
                f"Cross={row['cross']:.4f} | "
                f"Worst Cross={row['worst_cross']:.4f}"
            )
    # ========================================================
    # Print final mean ± std
    # ========================================================
    print("\n")
    print("=" * 110)
    print(
        "FINAL RESULTS — MEAN ± STD"
    )
    print("=" * 110)
    print(
        f"{'Metric':<12}"
        f"{'Self':>25}"
        f"{'Cross':>25}"
        f"{'Worst Cross':>25}"
    )
    print(
        "-" * 110
    )
    for _, row in final_df.iterrows():
        print(
            f"{row['metric'].upper():<12}"
            f"{row['self_mean']:.4f} ± "
            f"{row['self_std']:.4f}"
            f"{row['cross_mean']:>10.4f} ± "
            f"{row['cross_std']:.4f}"
            f"{row['worst_cross_mean']:>10.4f} ± "
            f"{row['worst_cross_std']:.4f}"
        )
    # ========================================================
    # Save results
    # ========================================================
    output_dir = (
        RESULTS_ROOT
        / "multi_seed_threshold_summary"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    per_seed_file = (
        output_dir
        / "per_seed_summary.csv"
    )
    final_file = (
        output_dir
        / "mean_std_summary.csv"
    )
    per_seed_df.to_csv(
        per_seed_file,
        index=False,
    )
    final_df.to_csv(
        final_file,
        index=False,
    )
    print("\n")
    print(
        "Saved per-seed results:"
    )
    print(
        f"  {per_seed_file}"
    )
    print(
        "\nSaved mean ± std results:"
    )
    print(
        f"  {final_file}"
    )
    # ========================================================
    # Create PPT charts
    # ========================================================
    print("\n")
    print("=" * 80)
    print(
        "CREATING PPT CHARTS"
    )
    print("=" * 80)
    # --------------------------------------------------------
    # Chart 1:
    # Accuracy / F1 / ROC-AUC
    # --------------------------------------------------------
    create_chart(
        final_df=final_df,
        metrics=[
            "accuracy",
            "f1",
            "auc",
        ],
        labels=[
            "Accuracy",
            "F1",
            "ROC-AUC",
        ],
        title=(
            "Pairwise Performance — "
            "Youden-Frozen Threshold"
        ),
        ylabel="Score",
        filename="performance_metrics.png",
        output_dir=output_dir,
    )
    # --------------------------------------------------------
    # Chart 2:
    # FPR / FNR
    # --------------------------------------------------------
    create_chart(
        final_df=final_df,
        metrics=[
            "fpr",
            "fnr",
        ],
        labels=[
            "FPR",
            "FNR",
        ],
        title=(
            "Pairwise Error Rates — "
            "Youden-Frozen Threshold"
        ),
        ylabel="Rate",
        filename="error_rates.png",
        output_dir=output_dir,
    )
    # ========================================================
    # Aggregate per-relay-pair FPR/FNR heatmaps across seeds
    # ========================================================
    print("\n" + "=" * 80)
    print("CREATING FIVE-SEED FPR/FNR HEATMAPS")
    print("=" * 80)
    for metric in ("fpr", "fnr"):
        stack = np.stack(heatmap_matrices[metric], axis=0)
        # Cell-wise mean across the five seeds; sample SD is also saved.
        mean_matrix = np.mean(stack, axis=0)
        std_matrix = np.std(stack, axis=0, ddof=1)
        mean_df = pd.DataFrame(
            mean_matrix,
            index=RELAY_ORDER,
            columns=RELAY_ORDER,
        )
        std_df = pd.DataFrame(
            std_matrix,
            index=RELAY_ORDER,
            columns=RELAY_ORDER,
        )
        mean_file = output_dir / f"{metric}_mean_matrix.csv"
        std_file = output_dir / f"{metric}_std_matrix.csv"
        mean_df.to_csv(mean_file, float_format="%.8f")
        std_df.to_csv(std_file, float_format="%.8f")
        title = (
            "False Positive Rate"
            if metric == "fpr"
            else "False Negative Rate"
        )
        heatmap_file = output_dir / f"{metric}_heatmap.png"
        create_heatmap(mean_matrix, title, heatmap_file)
        print(f"{metric.upper()} mean matrix -> {mean_file}")
        print(f"{metric.upper()} std matrix  -> {std_file}")
        print(f"{metric.upper()} heatmap     -> {heatmap_file}")
# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
