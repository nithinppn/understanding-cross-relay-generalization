import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.utils.dataset import load_dataset
from scripts.models.registry import MODEL_REGISTRY
from scripts.methodology.registry import METHODOLOGY_REGISTRY
from scripts.evaluation import Evaluator
from scripts.results import ResultsManager


def main():

    # ============================================================
    # ARGUMENTS
    # ============================================================

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate already-trained pairwise relay models "
            "inside an existing experiment run."
        )
    )

    parser.add_argument(
        "--model",
        default="compact_mlp",
        help="Model name from MODEL_REGISTRY",
    )

    parser.add_argument(
        "--methodology",
        default="pairwise",
        help="Methodology name from METHODOLOGY_REGISTRY",
    )

    parser.add_argument(
        "--training",
        default="default",
        help="Training configuration name",
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Existing experiment run directory containing "
            "the trained models."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed used to recreate the global "
            "simulation split."
        ),
    )

    args = parser.parse_args()

    # ============================================================
    # RUN DIRECTORY
    # ============================================================

    run_dir = Path(
        args.run_dir
    ).resolve()

    if not run_dir.exists():

        raise FileNotFoundError(
            f"Run directory does not exist:\n{run_dir}"
        )

    model_dir = (
        run_dir / "models"
    )

    if not model_dir.exists():

        raise FileNotFoundError(
            f"Model directory does not exist:\n{model_dir}"
        )

    # ============================================================
    # LOAD CONFIGURATION
    # ============================================================

    cfg = OmegaConf.load(
        "configs/experiment.yaml"
    )

    cfg.model = OmegaConf.load(
        f"configs/model/{args.model}.yaml"
    )

    cfg.methodology = OmegaConf.load(
        f"configs/methodology/{args.methodology}.yaml"
    )

    cfg.training = OmegaConf.load(
        f"configs/training/{args.training}.yaml"
    )

    cfg.seed = args.seed

    # IMPORTANT:
    # Use the actual experiment name.
    # Do NOT append "_evaluation".
    cfg.experiment_name = (
        f"{cfg.model.name}_"
        f"{cfg.methodology.name}_"
        f"{cfg.training.name}"
    )

    print("=" * 80)
    print("SAVED MODEL PAIRWISE EVALUATION")
    print("=" * 80)

    print(
        f"Experiment  : {cfg.experiment_name}"
    )

    print(
        f"Model       : {cfg.model.name}"
    )

    print(
        f"Methodology : {cfg.methodology.name}"
    )

    print(
        f"Training    : {cfg.training.name}"
    )

    print(
        f"Seed        : {cfg.seed}"
    )

    print(
        f"Run dir     : {run_dir}"
    )

    print(
        f"Model dir   : {model_dir}"
    )

    print("=" * 80)

    # ============================================================
    # LOAD DATASET
    # ============================================================

    print("\nLoading local dataset...")

    X_local, y_local = load_dataset()

    print(
        f"Local X: {X_local.shape}"
    )

    print(
        f"Local y: {y_local.shape}"
    )

    # ============================================================
    # RECREATE EXACT GLOBAL SPLIT
    # ============================================================

    print(
        "\nCreating global simulation split..."
    )

    methodology = METHODOLOGY_REGISTRY[
        cfg.methodology.name
    ]()

    relay_datasets = methodology.prepare_datasets(
        y_local=y_local,
        seed=cfg.seed,
    )

    print(
        f"\nPrepared splits for "
        f"{len(relay_datasets)} relays."
    )

    # ============================================================
    # CHECK ALL MODELS
    # ============================================================

    missing_models = []

    for relay_name in relay_datasets.keys():

        model_path = (
            model_dir
            / f"{relay_name}.pt"
        )

        if not model_path.exists():

            missing_models.append(
                str(model_path)
            )

    if missing_models:

        print(
            "\nMissing model files:"
        )

        for path in missing_models:
            print(
                f"  {path}"
            )

        raise FileNotFoundError(
            "One or more relay models are missing."
        )

    print(
        "\nAll 8 trained relay models found."
    )

    # ============================================================
    # CREATE RESULTS MANAGER
    # ============================================================

    # IMPORTANT:
    # Reuse the existing run directory.
    #
    # This does NOT create a new timestamp.
    #
    results = ResultsManager(
        cfg.experiment_name,
        run_directory=run_dir,
    )

    # Do NOT overwrite the original training
    # config.yaml when evaluating an existing run.

    results.set_config(
        cfg
    )

    # ============================================================
    # EVALUATOR
    # ============================================================

    evaluator = Evaluator(
        cfg
    )

    # ============================================================
    # PAIRWISE EVALUATION
    # ============================================================

    print("\n")
    print("=" * 80)
    print("STARTING PAIRWISE EVALUATION")
    print("=" * 80)

    total_pairs = (
        len(relay_datasets)
        ** 2
    )

    completed_pairs = 0

    # ============================================================
    # SOURCE RELAYS
    # ============================================================

    for train_relay in relay_datasets.keys():

        print("\n")
        print("-" * 80)

        print(
            f"Loading model trained on "
            f"{train_relay}"
        )

        print("-" * 80)

        # --------------------------------------------------------
        # CREATE MODEL
        # --------------------------------------------------------

        model = MODEL_REGISTRY[
            cfg.model.name
        ]()

        # --------------------------------------------------------
        # LOAD SAVED WEIGHTS
        # --------------------------------------------------------

        model_path = (
            model_dir
            / f"{train_relay}.pt"
        )

        print(
            f"Loading: {model_path}"
        )

        state_dict = torch.load(
            model_path,
            map_location="cpu",
        )

        model.load_state_dict(
            state_dict
        )

        # --------------------------------------------------------
        # TARGET RELAYS
        # --------------------------------------------------------

        for test_relay in relay_datasets.keys():

            # IMPORTANT:
            #
            # Every source -> target combination
            # uses ONLY the target relay samples
            # belonging to the GLOBAL TEST simulation set.
            #
            # This includes the diagonal.
            # ----------------------------------------------------

            test_indices = relay_datasets[
                test_relay
            ][
                "test_indices"
            ]

            metrics = evaluator.evaluate(
                model=model,
                X_local=X_local,
                y_local=y_local,
                indices=test_indices,
            )

            results.add_evaluation_result(
                train_relay,
                test_relay,
                metrics,
            )

            completed_pairs += 1

            print(
                f"{train_relay:20s} -> "
                f"{test_relay:20s} | "
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"AUC={metrics['auc']:.4f} | "
                f"FPR={metrics['fpr']:.4f} | "
                f"FNR={metrics['fnr']:.4f}"
            )

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n")
    print("=" * 80)
    print("GENERATING SUMMARY")
    print("=" * 80)

    results.generate_summary()

    results.save_results()

    # ============================================================
    # COMPLETE
    # ============================================================

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Completed pairs: "
        f"{completed_pairs}/{total_pairs}"
    )

    print(
        f"Results directory: "
        f"{results.output_directory}"
    )

    print(
        f"Model directory: "
        f"{results.model_directory}"
    )


if __name__ == "__main__":
    main()