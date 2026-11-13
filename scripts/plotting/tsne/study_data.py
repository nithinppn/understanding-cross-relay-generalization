from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TSNE_STUDY_DIR = (
    PROJECT_ROOT
    / "results"
    / "tsne_study"
)

TSNE_STUDY_FILE = (
    TSNE_STUDY_DIR
    / "study_samples.npz"
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_study_samples(
    X,
    y,
    relay,
    sample_indices,
    sample_id,
):
    """
    Save the fixed t-SNE study dataset.

    X:
        Shape (N, 480, 6)

    y:
        Fault labels, shape (N,)

    relay:
        Relay name for every sample, shape (N,)

    sample_indices:
        Original indices in the source dataset.

    sample_id:
        Simulation ID for every sample, shape (N,).
        Used as the grouping variable for simulation-level
        cross-validation.
    """

    TSNE_STUDY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        TSNE_STUDY_FILE,
        X=X,
        y=y,
        relay=relay,
        sample_indices=sample_indices,
        sample_id=sample_id,
    )

    print()
    print("Saved t-SNE study dataset:")
    print(f"  Path    : {TSNE_STUDY_FILE}")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Relays  : {len(np.unique(relay))}")
    print(
        f"  Simulations: "
        f"{len(np.unique(sample_id))}"
    )


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

def load_study_samples():
    """
    Load the fixed t-SNE study dataset.
    """

    if not TSNE_STUDY_FILE.exists():
        raise FileNotFoundError(
            "\nFixed t-SNE study dataset not found:\n"
            f"{TSNE_STUDY_FILE}\n\n"
            "Run:\n"
            "    python -m scripts.prepare_tsne_samples\n"
            "first.\n"
        )

    data = np.load(
        TSNE_STUDY_FILE,
        allow_pickle=False,
    )

    X = data["X"]
    y = data["y"]
    relay = data["relay"]
    sample_indices = data["sample_indices"]
    sample_id = data["sample_id"]

    print("Loaded fixed t-SNE study dataset:")
    print(f"  Path    : {TSNE_STUDY_FILE}")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Relays  : {len(np.unique(relay))}")
    print(
        f"  Simulations: "
        f"{len(np.unique(sample_id))}"
    )

    return (
        X,
        y,
        relay,
        sample_indices,
        sample_id,
    )