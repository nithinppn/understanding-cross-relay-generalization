#!/usr/bin/env python3

"""
Master analysis entry point.

Usage
-----
python -m scripts.plot_results \
    --experiment compact_mlp_pairwise_default

This command:

1. Finds the latest completed run for the experiment.
2. Validates the canonical pairwise results.json.
3. Generates canonical pairwise analysis outputs.
4. Runs source-validation threshold analysis.
5. Runs representation analysis:
   - grouped relay-ID probe
   - independent bus-ID probe
   - t-SNE
   - non-fault MMD -> FPR
   - fault MMD -> FNR
   - MMD correlations
   - representation plots

No model training is performed.

The canonical pairwise evaluation is NOT rerun.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.plotting.threshold_pairwise import (
    run_threshold_analysis,
)


# ============================================================
# Run discovery
# ============================================================

def find_latest_run(
    experiment_name: str,
) -> Path:
    """
    Find the latest completed run under:

        results/<experiment_name>/

    A run is considered completed when it contains
    results.json.

    The 'latest' symlink is ignored.
    """

    experiment_dir = (
        Path("results")
        / experiment_name
    )

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist:\n"
            f"{experiment_dir}"
        )

    runs = sorted(
        p
        for p in experiment_dir.iterdir()
        if (
            p.is_dir()
            and p.name != "latest"
            and (
                p / "results.json"
            ).exists()
        )
    )

    if not runs:
        raise FileNotFoundError(
            "No completed run containing "
            f"results.json found in:\n"
            f"{experiment_dir}"
        )

    return runs[-1]


# ============================================================
# Subprocess helper
# ============================================================

def run_command(
    command: list[str],
    description: str,
) -> None:
    """
    Run an external analysis command.

    Stops immediately if the command fails.
    """

    print()
    print("=" * 70)
    print(description)
    print("=" * 70)
    print()

    print("Command:")
    print(
        " ".join(command)
    )

    print()

    result = subprocess.run(
        command
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed "
            f"with exit code "
            f"{result.returncode}."
        )


# ============================================================
# Canonical result validation
# ============================================================

def validate_canonical_results(
    run_dir: Path,
) -> None:
    """
    Validate the canonical pairwise results.json.

    This file is the authoritative source for:

        Accuracy
        F1
        AUC
        FPR
        FNR

    The expected structure is:

        8 source relays
        8 target relays
        64 source-target pairs
    """

    results_file = (
        run_dir
        / "results.json"
    )

    if not results_file.exists():
        raise FileNotFoundError(
            "Canonical results.json not found:\n"
            f"{results_file}"
        )

    with open(
        results_file,
        "r",
    ) as f:

        results = json.load(f)

    # --------------------------------------------------------
    # Required top-level fields
    # --------------------------------------------------------

    required = {
        "config",
        "training",
        "evaluation",
        "summary",
    }

    missing = (
        required
        - set(results)
    )

    if missing:
        raise RuntimeError(
            "Canonical results.json "
            f"is missing keys: {missing}"
        )

    # --------------------------------------------------------
    # Evaluation structure
    # --------------------------------------------------------

    evaluation = (
        results["evaluation"]
    )

    if not isinstance(
        evaluation,
        dict,
    ):
        raise RuntimeError(
            "results.json['evaluation'] "
            "is not a dictionary."
        )

    source_count = len(
        evaluation
    )

    pair_count = sum(
        len(targets)
        for targets in evaluation.values()
    )

    # --------------------------------------------------------
    # Expected relay structure
    # --------------------------------------------------------

    if source_count != 8:
        raise RuntimeError(
            "Expected 8 source relays, "
            f"found {source_count}."
        )

    if pair_count != 64:
        raise RuntimeError(
            "Expected 64 source-target "
            f"pairs, found {pair_count}."
        )

    # --------------------------------------------------------
    # Print validation result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CANONICAL RESULTS VALIDATION")
    print("=" * 70)

    print()
    print(
        f"Results file : "
        f"{results_file}"
    )

    print(
        f"Source relays: "
        f"{source_count}"
    )

    print(
        f"Pairs        : "
        f"{pair_count}"
    )

    print()

    print(
        "Canonical pairwise results: PASSED"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete cross-relay "
            "analysis for an experiment."
        )
    )

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    parser.add_argument(
        "--experiment",
        required=True,
        help=(
            "Experiment name inside results/."
        ),
    )

    # --------------------------------------------------------
    # Optional explicit run
    # --------------------------------------------------------

    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional explicit run directory. "
            "If omitted, the latest completed "
            "run is automatically selected."
        ),
    )

    # --------------------------------------------------------
    # Optional skips
    # --------------------------------------------------------

    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help=(
            "Skip source-validation "
            "threshold analysis."
        ),
    )

    parser.add_argument(
        "--skip-representation",
        action="store_true",
        help=(
            "Skip t-SNE, MMD, and "
            "linear-probe analysis."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Resolve run directory
    # ========================================================

    if args.run_dir is not None:

        run_dir = (
            Path(
                args.run_dir
            )
            .resolve()
        )

    else:

        run_dir = (
            find_latest_run(
                args.experiment
            )
            .resolve()
        )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 70)
    print("COMPLETE CROSS-RELAY ANALYSIS")
    print("=" * 70)

    print()
    print(
        f"Experiment : "
        f"{args.experiment}"
    )

    print(
        f"Run        : "
        f"{run_dir}"
    )

    print()

    # ========================================================
    # Validate run directory
    # ========================================================

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory does not exist:\n"
            f"{run_dir}"
        )

    # ========================================================
    # Validate canonical results
    # ========================================================

    validate_canonical_results(
        run_dir
    )

    # ========================================================
    # STEP 1
    # Canonical pairwise analysis
    # ========================================================

    run_command(
        [
            sys.executable,
            "-m",
            "scripts.analyze_pairwise_results",
            "--results-dir",
            str(run_dir),
        ],
        (
            "STEP 1/3 — "
            "CANONICAL PAIRWISE ANALYSIS"
        ),
    )

    # ========================================================
    # STEP 2
    # Source-validation threshold analysis
    # ========================================================

    if not args.skip_threshold:

        print()
        print("=" * 70)
        print(
            "STEP 2/3 — "
            "SOURCE-VALIDATION "
            "THRESHOLD ANALYSIS"
        )
        print("=" * 70)

        print()
        print(
            "Using canonical run:"
        )

        print(
            f"  {run_dir}"
        )

        print()
        print(
            "Threshold results will be stored in:"
        )

        print(
            f"  {run_dir / 'analysis' / 'threshold'}"
        )

        # ----------------------------------------------------
        # Direct Python call.
        #
        # No subprocess.
        # No hard-coded experiment.
        # No hard-coded timestamp.
        # ----------------------------------------------------

        threshold_dir = (
            run_threshold_analysis(
                run_dir
            )
        )

        print()
        print(
            "Threshold analysis completed:"
        )

        print(
            f"  {threshold_dir}"
        )

    else:

        print()
        print("=" * 70)
        print(
            "STEP 2/3 — "
            "THRESHOLD ANALYSIS SKIPPED"
        )
        print("=" * 70)

    # ========================================================
    # STEP 3
    # Representation analysis
    # ========================================================

    if not args.skip_representation:

        run_command(
            [
                sys.executable,
                "-c",
                (
                    "from "
                    "scripts.plotting.tsne.pipeline "
                    "import run_tsne_pipeline; "
                    f"run_tsne_pipeline("
                    f"r'{run_dir}')"
                ),
            ],
            (
                "STEP 3/3 — "
                "REPRESENTATION / t-SNE / MMD "
                "ANALYSIS"
            ),
        )

    else:

        print()
        print("=" * 70)
        print(
            "STEP 3/3 — "
            "REPRESENTATION ANALYSIS SKIPPED"
        )
        print("=" * 70)

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print("COMPLETE ANALYSIS FINISHED")
    print("=" * 70)

    print()

    print(
        "Experiment run:"
    )

    print(
        f"  {run_dir}"
    )

    print()

    # --------------------------------------------------------
    # Pairwise
    # --------------------------------------------------------

    print(
        "[OK] Canonical pairwise "
        "results validated"
    )

    print(
        "[OK] Pairwise matrices"
    )

    print(
        "[OK] FPR/FNR heatmaps"
    )

    print(
        "[OK] Table I"
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if not args.skip_threshold:

        print(
            "[OK] Source-validation "
            "threshold analysis"
        )

        print(
            "[OK] Threshold matrices"
        )

        print(
            "[OK] Threshold FPR/FNR heatmaps"
        )

        print(
            "[OK] Threshold Table I"
        )

    else:

        print(
            "[SKIPPED] Source-validation "
            "threshold analysis"
        )

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    if not args.skip_representation:

        print(
            "[OK] Grouped relay-ID probe"
        )

        print(
            "[OK] Independent bus-ID probe"
        )

        print(
            "[OK] t-SNE"
        )

        print(
            "[OK] Non-fault MMD -> FPR"
        )

        print(
            "[OK] Fault MMD -> FNR"
        )

        print(
            "[OK] MMD correlations"
        )

        print(
            "[OK] Representation plots"
        )

    else:

        print(
            "[SKIPPED] Representation "
            "analysis"
        )

    # --------------------------------------------------------
    # Safety summary
    # --------------------------------------------------------

    print()

    print(
        "No model training was performed."
    )

    print(
        "The canonical pairwise test "
        "evaluation was not rerun."
    )

    print()

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()