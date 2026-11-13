import argparse

from omegaconf import OmegaConf

from scripts.utils.dataset import load_dataset
from scripts.models.registry import MODEL_REGISTRY
from scripts.methodology.registry import METHODOLOGY_REGISTRY
from scripts.training import Trainer
from scripts.evaluation import Evaluator
from scripts.results import ResultsManager


def main():

    # ============================================================
    # Arguments
    # ============================================================

    parser = argparse.ArgumentParser()

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
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed",
    )

    args = parser.parse_args()

    # ============================================================
    # Load configuration
    # ============================================================

    cfg = OmegaConf.load(
        "configs/experiment.yaml"
    )

    if args.seed is not None:
        cfg.seed = args.seed

    cfg.model = OmegaConf.load(
        f"configs/model/{args.model}.yaml"
    )

    cfg.methodology = OmegaConf.load(
        f"configs/methodology/{args.methodology}.yaml"
    )

    cfg.training = OmegaConf.load(
        f"configs/training/{args.training}.yaml"
    )

    cfg.experiment_name = (
        f"{cfg.model.name}_"
        f"{cfg.methodology.name}_"
        f"{cfg.training.name}"
    )

    # ============================================================
    # Print experiment configuration
    # ============================================================

    print("=" * 80)
    print(f"Model       : {cfg.model.name}")
    print(f"Methodology : {cfg.methodology.name}")
    print(f"Training    : {cfg.training.name}")
    print(f"Seed        : {cfg.seed}")
    print("=" * 80)

    # ============================================================
    # Load local dataset
    # ============================================================

    print("\nLoading local dataset...")

    X_local, y_local = load_dataset()

    print(f"Local X: {X_local.shape}")
    print(f"Local y: {y_local.shape}")

    # ============================================================
    # Prepare pairwise datasets
    # ============================================================

    methodology = METHODOLOGY_REGISTRY[
        cfg.methodology.name
    ]()

    relay_datasets = methodology.prepare_datasets(
        y_local=y_local,
        seed=cfg.seed,
    )

    print(
        f"\nPrepared pairwise splits for "
        f"{len(relay_datasets)} relays."
    )

    # ============================================================
    # Trainer / evaluator / results
    # ============================================================

    trainer = Trainer(cfg)

    evaluator = Evaluator(cfg)

    results = ResultsManager(
        cfg.experiment_name
    )

    results.save_config(cfg)
    results.set_config(cfg)

    # ============================================================
    # TRAINING
    # ============================================================

    for relay_name, relay_split in relay_datasets.items():

        print("\n")
        print("=" * 80)
        print(f"Training {relay_name}")
        print("=" * 80)

        # --------------------------------------------------------
        # Create model
        # --------------------------------------------------------

        model = MODEL_REGISTRY[
            cfg.model.name
        ]()

        # --------------------------------------------------------
        # Train using SOURCE TRAIN and SOURCE VALIDATION
        # simulations
        # --------------------------------------------------------

        training_result = trainer.fit(
            model=model,
            relay_name=relay_name,
            X_local=X_local,
            y_local=y_local,
            train_indices=relay_split[
                "train_indices"
            ],
            validation_indices=relay_split[
                "validation_indices"
            ],
        )

        # --------------------------------------------------------
        # Save model
        # --------------------------------------------------------

        results.save_model(
            training_result["model"],
            relay_name,
        )

        # --------------------------------------------------------
        # Save training result
        # --------------------------------------------------------

        results.add_training_result(
            relay_name,
            training_result,
        )

    # ============================================================
    # EVALUATION
    # ============================================================

    print("\n")
    print("=" * 80)
    print("STARTING PAIRWISE EVALUATION")
    print("=" * 80)

    for train_relay in relay_datasets.keys():

        print("\n")
        print(
            f"Evaluating model trained on "
            f"{train_relay}"
        )

        # --------------------------------------------------------
        # Load trained model
        # --------------------------------------------------------

        model = MODEL_REGISTRY[
            cfg.model.name
        ]()

        model = results.load_model(
            model,
            train_relay,
        )

        # --------------------------------------------------------
        # Evaluate on every relay
        # --------------------------------------------------------

        for test_relay in relay_datasets.keys():

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # Every source -> target combination is evaluated
            # ONLY on the target relay's samples from the
            # GLOBAL TEST simulations.
            #
            # This applies to:
            #
            #   source == target
            #
            # and
            #
            #   source != target
            #
            # No target training or validation samples are used
            # during evaluation.
            # ----------------------------------------------------

            indices = relay_datasets[
                test_relay
            ]["test_indices"]

            metrics = evaluator.evaluate(
                model=model,
                X_local=X_local,
                y_local=y_local,
                indices=indices,
            )

            results.add_evaluation_result(
                train_relay,
                test_relay,
                metrics,
            )

            print(
                f"{train_relay:20s} -> "
                f"{test_relay:20s} | "
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"AUC={metrics['auc']:.4f}"
            )

    # ============================================================
    # Summary
    # ============================================================

    print("\n")
    print("=" * 80)
    print("GENERATING SUMMARY")
    print("=" * 80)

    results.generate_summary()

    results.save_results()

    # ============================================================
    # Finished
    # ============================================================

    print("\n")
    print("=" * 80)
    print("COMPACT MLP PAIRWISE EXPERIMENT FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()