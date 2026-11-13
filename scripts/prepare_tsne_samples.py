import os
from pathlib import Path

import numpy as np

from hydra import compose, initialize_config_dir

from psp_helper.api import load_local_windowed_dataset

from scripts.plotting.tsne.study_data import (
    save_study_samples,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = Path(os.environ["EVENTBENCH_DATASET_DIR"])


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42

WINDOWS_PER_CLASS = 150

SENSOR_MAP = {
    "Bus_1_Line_1-2A": "Bus1_Ln12A",
    "Bus_1_Line_1-2B": "Bus1_Ln12B",

    "Bus_2_Line_1-2A": "Bus2_Ln12A",
    "Bus_2_Line_1-2B": "Bus2_Ln12B",

    "Bus_2_Line_2-3A": "Bus2_Ln23A",
    "Bus_2_Line_2-3B": "Bus2_Ln23B",

    "Bus_3_Line_2-3A": "Bus3_Ln23A",
    "Bus_3_Line_2-3B": "Bus3_Ln23B",
}


def main():

    rng = np.random.default_rng(SEED)

    print()
    print("=" * 70)
    print("Preparing fixed t-SNE study dataset")
    print("=" * 70)
    print()

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATASET_DIR}")
    print()

    # ---------------------------------------------------------------
    # Resolve helper config
    # ---------------------------------------------------------------

    config_dir = (
        PROJECT_ROOT
        / "helper"
        / "power-grid-and-ai-helper-functions"
        / "config"
    )

    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found:\n{config_dir}"
        )

    print("Config directory:", config_dir)
    print("Config exists   :", config_dir.exists())

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    print("Loading dataset...")
    print()

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):

        cfg = compose(
            config_name="main-config.yaml",
            overrides=[
                "dataset=hv_double_line_110kv"
            ],
        )

        # IMPORTANT:
        # Prevent psp_helper from falling back to its default dataset location

        cfg.dataset.dataset_directory = str(
            DATASET_DIR
        )

        cfg.dataset.tmp_data_directory = str(
            PROJECT_ROOT
        )

        cfg.paths.benchmark_root = str(
            DATASET_DIR
        )

        cfg.paths.adapt_grid_root = str(
            DATASET_DIR
        )

        cfg.window_extraction.window_length = 0.05

        cfg.window_extraction.step_length_seconds = 0.005

        print(
            "Dataset directory:",
            cfg.dataset.dataset_directory,
        )

        print(
            "Temporary data directory:",
            cfg.dataset.tmp_data_directory,
        )

        print()

        X_raw, y_df, _ = (
            load_local_windowed_dataset(cfg)
        )

    print()
    print("Dataset loaded.")
    print(f"X shape : {X_raw.shape}")
    print(f"Y shape : {y_df.shape}")
    print()

    # ---------------------------------------------------------------
    # Verify labels
    # ---------------------------------------------------------------

    required_columns = {
        "sensor_name",
        "event_family",
        "y_fault_present",
        "sample_id",
    }

    missing = (
        required_columns
        - set(y_df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            f"{sorted(missing)}"
        )

    print()
    print("=" * 70)
    print("AVAILABLE SENSOR NAMES")
    print("=" * 70)

    print(
        y_df["sensor_name"]
        .dropna()
        .unique()
    )

    print()
    print("FAULT DISTRIBUTION:")

    print(
        y_df["y_fault_present"].value_counts()
    )

    print()
    print("EVENT FAMILY:")

    print(
        y_df["event_family"].value_counts()
    )

    print()
    print(
        "Unique simulations:",
        y_df["sample_id"].nunique(),
    )

    # ---------------------------------------------------------------
    # Sample
    # ---------------------------------------------------------------

    sampled_indices = []
    sampled_relays = []
    sampled_faults = []

    print("Sampling balanced t-SNE dataset...")
    print()

    for raw_name, friendly_name in SENSOR_MAP.items():

        mask_normal = (
            (y_df["sensor_name"] == raw_name)
            &
            (y_df["event_family"] != "switching")
            &
            (y_df["y_fault_present"] == 0)
        )

        mask_fault = (
            (y_df["sensor_name"] == raw_name)
            &
            (y_df["event_family"] != "switching")
            &
            (y_df["y_fault_present"] == 1)
        )

        idx_normal = np.where(
            mask_normal
        )[0]

        idx_fault = np.where(
            mask_fault
        )[0]

        print(
            f"{friendly_name:<15} "
            f"normal={len(idx_normal):7d} "
            f"fault={len(idx_fault):7d}"
        )

        if (
            len(idx_normal) < WINDOWS_PER_CLASS
            or
            len(idx_fault) < WINDOWS_PER_CLASS
        ):
            raise RuntimeError(
                f"Not enough samples for "
                f"{friendly_name}: "
                f"normal={len(idx_normal)}, "
                f"fault={len(idx_fault)}"
            )

        selected_normal = rng.choice(
            idx_normal,
            size=WINDOWS_PER_CLASS,
            replace=False,
        )

        selected_fault = rng.choice(
            idx_fault,
            size=WINDOWS_PER_CLASS,
            replace=False,
        )

        sampled_indices.extend(
            selected_normal
        )

        sampled_indices.extend(
            selected_fault
        )

        sampled_relays.extend(
            [friendly_name]
            * (2 * WINDOWS_PER_CLASS)
        )

        sampled_faults.extend(
            [0] * WINDOWS_PER_CLASS
            +
            [1] * WINDOWS_PER_CLASS
        )

    # ---------------------------------------------------------------
    # Convert
    # ---------------------------------------------------------------

    sampled_indices = np.asarray(
        sampled_indices,
        dtype=np.int64,
    )

    sampled_relays = np.asarray(
        sampled_relays,
        dtype=str,
    )

    sampled_faults = np.asarray(
        sampled_faults,
        dtype=np.int64,
    )

    # ---------------------------------------------------------------
    # Extract simulation IDs
    # ---------------------------------------------------------------

    sampled_sample_ids = (
        y_df.iloc[sampled_indices]["sample_id"]
        .to_numpy()
    )

    if len(sampled_sample_ids) != len(
        sampled_indices
    ):
        raise RuntimeError(
            "sample_id count does not match "
            "sampled indices."
        )

    # ---------------------------------------------------------------
    # Verify selected samples contain no switching events
    # ---------------------------------------------------------------

    selected_event_families = np.asarray(
        y_df.iloc[sampled_indices]["event_family"]
    )

    print()
    print("=" * 70)
    print("EVENT FAMILY IN SELECTED t-SNE SAMPLES")
    print("=" * 70)

    unique_events, event_counts = np.unique(
        selected_event_families,
        return_counts=True,
    )

    for event, count in zip(
        unique_events,
        event_counts,
    ):
        print(
            f"{str(event):15s}: {count:5d}"
        )

    switching_count = np.sum(
        selected_event_families == "switching"
    )

    print()
    print(
        f"Switching samples: {switching_count}"
    )

    if switching_count > 0:
        raise RuntimeError(
            f"Found {switching_count} switching samples "
            "in the t-SNE dataset!"
        )

    print("✓ No switching samples found.")

    # ---------------------------------------------------------------
    # Extract the actual 2400 windows
    # ---------------------------------------------------------------

    X_study = np.asarray(
        X_raw[sampled_indices],
        dtype=np.float32,
    )

    expected_samples = (
        len(SENSOR_MAP)
        * WINDOWS_PER_CLASS
        * 2
    )

    print()
    print("=" * 70)
    print("t-SNE study dataset")
    print("=" * 70)

    print(
        f"Expected samples : {expected_samples}"
    )

    print(
        f"Actual samples   : {len(X_study)}"
    )

    print(
        f"X shape          : {X_study.shape}"
    )

    print(
        f"y shape          : {sampled_faults.shape}"
    )

    print(
        f"Relays           : "
        f"{len(np.unique(sampled_relays))}"
    )

    print(
        f"Simulations      : "
        f"{len(np.unique(sampled_sample_ids))}"
    )

    if len(X_study) != 2400:
        raise RuntimeError(
            "Expected exactly 2400 samples."
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    save_study_samples(
        X=X_study,
        y=sampled_faults,
        relay=sampled_relays,
        sample_indices=sampled_indices,
        sample_id=sampled_sample_ids,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()