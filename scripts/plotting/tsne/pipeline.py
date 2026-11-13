from pathlib import Path

import json
import numpy as np
import torch
import yaml

from scripts.models.registry import get_model

from .study_data import load_study_samples
from .embeddings import extract_embeddings
from .analysis import run_tsne
from .categories import prediction_categories
from .visualization import plot_tsne
from ..style import relay_display_name

from scripts.plotting.representation_analysis import (
    relay_id_probe,
    group_relay_predictions_by_bus,
    compute_mmd,
    _median_bandwidth,
    correlate_mmd_with_fpr,
    correlate_mmd_with_fnr,
    plot_mmd_vs_fpr,
    plot_mmd_vs_fnr,
    plot_relay_probe_confusion,
    plot_bus_group_probe_confusion,
    save_representation_results,
    bus_id_probe,
)


def load_model_name(experiment_directory):
    """
    Read model architecture from config.yaml.
    """

    config_path = (
        experiment_directory / "config.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found:\n"
            f"{config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        return config["model"]["name"]

    except KeyError:
        raise KeyError(
            "Could not find "
            "'model.name' in config.yaml"
        )


def load_experiment_results(
    experiment_directory,
):
    """
    Load results.json so that MMD values can
    be paired with cross-relay FPR and FNR.
    """

    results_path = (
        experiment_directory
        / "results.json"
    )

    if not results_path.exists():
        raise FileNotFoundError(
            f"results.json not found:\n"
            f"{results_path}"
        )

    with open(
        results_path,
        "r",
    ) as f:

        return json.load(f)


def run_predictions(
    model,
    X,
    device,
    batch_size,
):
    """
    Run model predictions on all samples.
    """

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
    )

    probabilities = []

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size,
        ):

            end = start + batch_size

            X_batch = (
                X_tensor[start:end]
                .to(device)
            )

            probs = model.predict_proba(
                X_batch
            )

            probabilities.append(
                probs.cpu().numpy()
            )

    return np.concatenate(
        probabilities,
        axis=0,
    )


def load_relay_model(
    model_name,
    model_path,
    device,
):
    """
    Instantiate model and load trained
    relay-specific weights.
    """

    model = get_model(
        model_name
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model


def run_tsne_pipeline(
    experiment_directory,
    device=None,
    perplexity=30,
    batch_size=1024,
):
    """
    Cross-relay t-SNE and latent representation analysis.

    For every trained relay model:

    1. Extract embeddings for the fixed 2400-sample dataset.
    2. Generate predictions.
    3. Generate t-SNE visualization.
    4. Run grouped relay-ID linear probe.
    5. Run independent bus-ID linear probe.
    6. Calculate non-fault MMD and associate it with FPR.
    7. Calculate fault MMD and associate it with FNR.

    MMD is calculated in the original latent space,
    not on t-SNE coordinates.

    A source-specific RBF bandwidth is calculated once
    for each source model and reused for all target relays
    and for both fault/non-fault MMD calculations.
    """

    experiment_directory = Path(
        experiment_directory
    )

    model_directory = (
        experiment_directory / "models"
    )

    output_directory = (
        experiment_directory / "plots"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print()
    print("=" * 70)
    print("Cross-relay t-SNE analysis")
    print("=" * 70)

    print(
        f"Experiment : "
        f"{experiment_directory}"
    )

    print(
        f"Models     : "
        f"{model_directory}"
    )

    print(
        f"Device     : {device}"
    )

    # ---------------------------------------------------------------
    # Model architecture
    # ---------------------------------------------------------------

    model_name = load_model_name(
        experiment_directory
    )

    print(
        f"Architecture : {model_name}"
    )

    # ---------------------------------------------------------------
    # Load results
    # ---------------------------------------------------------------

    experiment_results = (
        load_experiment_results(
            experiment_directory
        )
    )

    evaluation_results = (
        experiment_results.get(
            "evaluation",
            {},
        )
    )

    # ---------------------------------------------------------------
    # Load fixed study dataset
    # ---------------------------------------------------------------

    (
        X,
        y,
        relay,
        sample_indices,
        sample_id,
    ) = load_study_samples()

    relays = np.unique(
        relay
    )

    print()
    print(
        f"Study samples : {len(X)}"
    )

    print(
        f"Relays        : {len(relays)}"
    )

    if len(X) != 2400:

        raise RuntimeError(
            f"Expected 2400 study samples, "
            f"got {len(X)}"
        )

    # ---------------------------------------------------------------
    # Verify class balance
    # ---------------------------------------------------------------

    unique_classes, class_counts = np.unique(
        y,
        return_counts=True,
    )

    print(
        f"Classes       : "
        f"{dict(zip(unique_classes, class_counts))}"
    )

    # ---------------------------------------------------------------
    # Store representation analysis
    # ---------------------------------------------------------------

    representation_results = {}

    mmd_fpr_records = []
    mmd_fnr_records = []

    # ---------------------------------------------------------------
    # Process every trained relay model
    # ---------------------------------------------------------------

    for train_relay in relays:

        print()
        print("-" * 70)
        print(
            f"Model trained on: "
            f"{train_relay}"
        )
        print("-" * 70)

        # -----------------------------------------------------------
        # Locate checkpoint
        # -----------------------------------------------------------

        model_path = (
            model_directory
            / f"{train_relay}.pt"
        )

        if not model_path.exists():

            model_path = (
                model_directory
                / f"model_{train_relay}.pt"
            )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found:\n"
                f"{model_directory}"
            )

        print(
            f"Checkpoint : {model_path}"
        )

        # -----------------------------------------------------------
        # Load model
        # -----------------------------------------------------------

        model = load_relay_model(
            model_name=model_name,
            model_path=model_path,
            device=device,
        )

        # -----------------------------------------------------------
        # Extract embeddings
        # -----------------------------------------------------------

        embeddings = extract_embeddings(
            model=model,
            X=X,
            device=device,
            batch_size=batch_size,
        )

        print(
            f"Embeddings : "
            f"{embeddings.shape}"
        )

        # -----------------------------------------------------------
        # Predictions
        # -----------------------------------------------------------

        probabilities = run_predictions(
            model=model,
            X=X,
            device=device,
            batch_size=batch_size,
        )

        print(
            f"Predictions: "
            f"{probabilities.shape}"
        )

        # -----------------------------------------------------------
        # Verify
        # -----------------------------------------------------------

        if len(embeddings) != len(X):

            raise RuntimeError(
                "Embedding count does not "
                "match study dataset."
            )

        if len(probabilities) != len(X):

            raise RuntimeError(
                "Prediction count does not "
                "match study dataset."
            )

        # -----------------------------------------------------------
        # Relay-ID linear probe
        # -----------------------------------------------------------

        print(
            "Running relay-ID linear probe..."
        )

        probe_result = relay_id_probe(
            embeddings=embeddings,
            relay=relay,
            sample_id=sample_id,
        )

        print(
            f"Relay-ID accuracy : "
            f"{probe_result['accuracy']:.4f}"
        )

        print(
            f"Relay-ID chance   : "
            f"{probe_result['chance_accuracy']:.4f}"
        )

        # -----------------------------------------------------------
        # Independent bus-ID linear probe
        # -----------------------------------------------------------

        print(
            "Running independent bus-ID linear probe..."
        )

        bus_probe_result = bus_id_probe(
            embeddings=embeddings,
            relay=relay,
            sample_id=sample_id,
        )

        print(
            f"Bus-ID accuracy   : "
            f"{bus_probe_result['accuracy']:.4f}"
        )

        print(
            f"Bus-ID chance     : "
            f"{bus_probe_result['chance_accuracy']:.4f}"
        )

        # -----------------------------------------------------------
        # Save relay-ID probe figure
        # -----------------------------------------------------------

        probe_plot_path = (
            output_directory
            / (
                f"{train_relay}"
                "_relay_id_probe.png"
            )
        )

        plot_relay_probe_confusion(
            probe_result=probe_result,
            output_path=probe_plot_path,
        )

        # -----------------------------------------------------------
        # Bus-grouped relay-ID confusion matrix
        # -----------------------------------------------------------

        bus_grouped_result = (
            group_relay_predictions_by_bus(
                relay=relay,
                predictions=probe_result[
                    "predictions"
                ],
            )
        )

        bus_probe_plot_path = (
            output_directory
            / f"{train_relay}_relay_id_probe_bus_grouped.png"
        )

        plot_bus_group_probe_confusion(
            bus_result=bus_grouped_result,
            output_path=bus_probe_plot_path,
        )

        # -----------------------------------------------------------
        # Store model-specific results
        # -----------------------------------------------------------

        representation_results[
            train_relay
        ] = {
            "relay_id_accuracy": (
                probe_result[
                    "accuracy"
                ]
            ),
            "relay_id_chance": (
                probe_result[
                    "chance_accuracy"
                ]
            ),
            "bus_id_accuracy": (
                bus_probe_result[
                    "accuracy"
                ]
            ),
            "bus_id_chance": (
                bus_probe_result[
                    "chance_accuracy"
                ]
            ),
            "pairwise_mmd_nonfault": {},
            "pairwise_mmd_fault": {},
        }

        # -----------------------------------------------------------
        # Split embeddings by relay and class
        # -----------------------------------------------------------

        relay_embeddings = {}
        relay_non_fault_embeddings = {}
        relay_fault_embeddings = {}

        for relay_name in relays:

            relay_mask = (
                relay == relay_name
            )

            non_fault_mask = (
                relay_mask
                & (y == 0)
            )

            fault_mask = (
                relay_mask
                & (y == 1)
            )

            relay_embeddings[
                relay_name
            ] = embeddings[
                relay_mask
            ]

            relay_non_fault_embeddings[
                relay_name
            ] = embeddings[
                non_fault_mask
            ]

            relay_fault_embeddings[
                relay_name
            ] = embeddings[
                fault_mask
            ]

        # -----------------------------------------------------------
        # Model-specific pairwise MMD
        # -----------------------------------------------------------

        print(
            "Computing model-specific MMD..."
        )

        # -----------------------------------------------------------
        # Source-specific MMD bandwidth
        # -----------------------------------------------------------
        #
        # Determine the bandwidth once from the complete source
        # embedding distribution and reuse it for every target.
        #
        # The same source-specific bandwidth is then used for:
        #
        #   1. non-fault MMD
        #   2. fault MMD
        #
        # This keeps the kernel scale fixed within each source model.
        # -----------------------------------------------------------

        train_embeddings = (
            relay_embeddings[
                train_relay
            ]
        )

        mmd_bandwidth = _median_bandwidth(
            train_embeddings,
            train_embeddings,
        )

        print(
            f"MMD bandwidth for "
            f"{train_relay}: "
            f"{mmd_bandwidth:.6f}"
        )

        # Store bandwidth for reproducibility.
        representation_results[
            train_relay
        ]["mmd_bandwidth"] = float(
            mmd_bandwidth
        )

        # -----------------------------------------------------------
        # Compare source model against every target relay
        # -----------------------------------------------------------

        for test_relay in relays:

            if (
                test_relay
                == train_relay
            ):
                continue

            # -------------------------------------------------------
            # Non-fault embeddings
            # -------------------------------------------------------

            source_non_fault = (
                relay_non_fault_embeddings[
                    train_relay
                ]
            )

            target_non_fault = (
                relay_non_fault_embeddings[
                    test_relay
                ]
            )

            non_fault_mmd_result = compute_mmd(
                source_non_fault,
                target_non_fault,
                bandwidth=mmd_bandwidth,
            )

            non_fault_mmd = (
                non_fault_mmd_result[
                    "mmd_squared"
                ]
            )

            # -------------------------------------------------------
            # Fault embeddings
            # -------------------------------------------------------

            source_fault = (
                relay_fault_embeddings[
                    train_relay
                ]
            )

            target_fault = (
                relay_fault_embeddings[
                    test_relay
                ]
            )

            fault_mmd_result = compute_mmd(
                source_fault,
                target_fault,
                bandwidth=mmd_bandwidth,
            )

            fault_mmd = (
                fault_mmd_result[
                    "mmd_squared"
                ]
            )

            # -------------------------------------------------------
            # Store separate MMD values
            # -------------------------------------------------------

            representation_results[
                train_relay
            ]["pairwise_mmd_nonfault"][
                test_relay
            ] = float(
                non_fault_mmd
            )

            representation_results[
                train_relay
            ]["pairwise_mmd_fault"][
                test_relay
            ] = float(
                fault_mmd
            )

            # -------------------------------------------------------
            # Retrieve evaluation result
            # -------------------------------------------------------

            try:

                evaluation = (
                    evaluation_results[
                        train_relay
                    ][
                        test_relay
                    ]
                )

            except KeyError:

                print(
                    f"WARNING: No evaluation "
                    f"result for "
                    f"{train_relay} -> "
                    f"{test_relay}"
                )

                continue

            # -------------------------------------------------------
            # FPR
            # -------------------------------------------------------

            tn = evaluation.get(
                "tn"
            )

            fp = evaluation.get(
                "fp"
            )

            if (
                tn is not None
                and fp is not None
            ):

                denominator = (
                    fp + tn
                )

                if denominator > 0:

                    fpr = (
                        fp / denominator
                    )

                    mmd_fpr_records.append(
                        {
                            "train_relay": (
                                train_relay
                            ),
                            "test_relay": (
                                test_relay
                            ),
                            "mmd": float(
                                non_fault_mmd
                            ),
                            "fpr": float(
                                fpr
                            ),
                        }
                    )

            # -------------------------------------------------------
            # FNR
            # -------------------------------------------------------

            fn = evaluation.get(
                "fn"
            )

            tp = evaluation.get(
                "tp"
            )

            if (
                fn is not None
                and tp is not None
            ):

                denominator = (
                    fn + tp
                )

                if denominator > 0:

                    fnr = (
                        fn / denominator
                    )

                    mmd_fnr_records.append(
                        {
                            "train_relay": (
                                train_relay
                            ),
                            "test_relay": (
                                test_relay
                            ),
                            "mmd": float(
                                fault_mmd
                            ),
                            "fnr": float(
                                fnr
                            ),
                        }
                    )

        # -----------------------------------------------------------
        # Prediction categories
        # -----------------------------------------------------------

        categories = (
            prediction_categories(
                y_true=y,
                y_probability=probabilities,
            )
        )

        # -----------------------------------------------------------
        # Run t-SNE
        # -----------------------------------------------------------

        print(
            "Running t-SNE..."
        )

        embedding_2d = run_tsne(
            embeddings,
            perplexity=perplexity,
        )

        print(
            f"t-SNE output: "
            f"{embedding_2d.shape}"
        )

        # -----------------------------------------------------------
        # Save t-SNE figure
        # -----------------------------------------------------------

        output_path = (
            output_directory
            / f"{train_relay}_tsne.png"
        )

        plot_tsne(
            embedding_2d=embedding_2d,
            relay=relay,
            y_true=y,
            categories=categories,
            output_path=output_path,
            title=(
                f"{relay_display_name(train_relay)} model — "
                f"{model_name}"
            ),
        )

        # -----------------------------------------------------------
        # Free model memory
        # -----------------------------------------------------------

        del model

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Aggregate relay-ID probe
    # -----------------------------------------------------------------

    relay_probe_values = [
        value["relay_id_accuracy"]
        for value
        in representation_results.values()
    ]

    relay_probe_chance = (
        1.0 / len(relays)
    )

    mean_relay_probe = (
        float(
            np.mean(
                relay_probe_values
            )
        )
        if relay_probe_values
        else None
    )

    # -----------------------------------------------------------------
    # Aggregate non-fault MMD
    # -----------------------------------------------------------------

    all_non_fault_mmd = [
        value
        for model_result
        in representation_results.values()
        for value
        in model_result[
            "pairwise_mmd_nonfault"
        ].values()
    ]

    mean_non_fault_mmd = (
        float(
            np.mean(
                all_non_fault_mmd
            )
        )
        if all_non_fault_mmd
        else None
    )

    # -----------------------------------------------------------------
    # Aggregate fault MMD
    # -----------------------------------------------------------------

    all_fault_mmd = [
        value
        for model_result
        in representation_results.values()
        for value
        in model_result[
            "pairwise_mmd_fault"
        ].values()
    ]

    mean_fault_mmd = (
        float(
            np.mean(
                all_fault_mmd
            )
        )
        if all_fault_mmd
        else None
    )

    # -----------------------------------------------------------------
    # MMD ↔ FPR correlation
    # -----------------------------------------------------------------

    correlation = (
        correlate_mmd_with_fpr(
            mmd_fpr_records
        )
        if len(mmd_fpr_records) >= 3
        else {
            "n_pairs": len(
                mmd_fpr_records
            ),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    )

    # -----------------------------------------------------------------
    # MMD ↔ FNR correlation
    # -----------------------------------------------------------------

    fnr_correlation = (
        correlate_mmd_with_fnr(
            mmd_fnr_records
        )
        if len(mmd_fnr_records) >= 3
        else {
            "n_pairs": len(
                mmd_fnr_records
            ),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("LATENT REPRESENTATION ANALYSIS")
    print("=" * 70)

    print(
        f"Mean relay-ID accuracy : "
        f"{mean_relay_probe:.4f}"
    )

    print(
        f"Relay-ID chance        : "
        f"{relay_probe_chance:.4f}"
    )

    print(
        f"Mean non-fault MMD     : "
        f"{mean_non_fault_mmd:.6f}"
    )

    print(
        f"Mean fault MMD         : "
        f"{mean_fault_mmd:.6f}"
    )

    print()
    print(
        f"MMD-FPR pairs          : "
        f"{correlation['n_pairs']}"
    )

    print(
        f"FPR Pearson r          : "
        f"{correlation['pearson_r']}"
    )

    print(
        f"FPR Pearson p          : "
        f"{correlation['pearson_p']}"
    )

    print(
        f"FPR Spearman rho       : "
        f"{correlation['spearman_rho']}"
    )

    print(
        f"FPR Spearman p         : "
        f"{correlation['spearman_p']}"
    )

    print()
    print(
        f"MMD-FNR pairs          : "
        f"{fnr_correlation['n_pairs']}"
    )

    print(
        f"FNR Pearson r          : "
        f"{fnr_correlation['pearson_r']}"
    )

    print(
        f"FNR Pearson p          : "
        f"{fnr_correlation['pearson_p']}"
    )

    print(
        f"FNR Spearman rho       : "
        f"{fnr_correlation['spearman_rho']}"
    )

    print(
        f"FNR Spearman p         : "
        f"{fnr_correlation['spearman_p']}"
    )

    # -----------------------------------------------------------------
    # MMD vs FPR plot
    # -----------------------------------------------------------------

    if len(mmd_fpr_records) >= 3:

        mmd_fpr_plot = (
            output_directory
            / "mmd_vs_fpr.png"
        )

        plot_mmd_vs_fpr(
            records=mmd_fpr_records,
            correlation=correlation,
            output_path=mmd_fpr_plot,
        )

    # -----------------------------------------------------------------
    # MMD vs FNR plot
    # -----------------------------------------------------------------

    if len(mmd_fnr_records) >= 3:

        mmd_fnr_plot = (
            output_directory
            / "mmd_vs_fnr.png"
        )

        plot_mmd_vs_fnr(
            records=mmd_fnr_records,
            correlation=fnr_correlation,
            output_path=mmd_fnr_plot,
        )

    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------

    representation_results[
        "_summary"
    ] = {
        "mean_relay_id_accuracy": (
            mean_relay_probe
        ),
        "relay_id_chance": (
            relay_probe_chance
        ),
        "mean_non_fault_mmd": (
            mean_non_fault_mmd
        ),
        "mean_fault_mmd": (
            mean_fault_mmd
        ),
        "mmd_fpr_correlation": (
            correlation
        ),
        "mmd_fnr_correlation": (
            fnr_correlation
        ),
    }

    representation_results[
        "_mmd_fpr_records"
    ] = mmd_fpr_records

    representation_results[
        "_mmd_fnr_records"
    ] = mmd_fnr_records

    metrics_path = (
        output_directory
        / "representation_metrics.json"
    )

    save_representation_results(
        results=representation_results,
        output_path=metrics_path,
    )

    print()
    print("=" * 70)
    print(
        "Cross-relay t-SNE and representation "
        "analysis complete"
    )
    print("=" * 70)