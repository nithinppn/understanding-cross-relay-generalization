import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

from scripts.models.compact_mlp import CompactMLP_FD

PROJECT_ROOT = Path.cwd()

for _ in range(5):

    if (
        (PROJECT_ROOT / "helper").exists()
        or (PROJECT_ROOT / "hv_double_line_110kv").exists()
    ):
        break

    PROJECT_ROOT = PROJECT_ROOT.parent


DATASET_DIR = Path(
    os.environ.get(
        "EVENTBENCH_DATASET_DIR",
        PROJECT_ROOT / "hv_double_line_110kv",
    )
)


if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset not found at {DATASET_DIR}. "
        "Set EVENTBENCH_DATASET_DIR to the dataset location (see README)."
    )


# ============================================================
# Resolve helper configuration path
# ============================================================

resolved_config_path = (
    "../helper/"
    "power-grid-and-ai-helper-functions/"
    "config"
)


print(
    "Detected Project Root:",
    PROJECT_ROOT,
)

print(
    "Detected Dataset Dir :",
    DATASET_DIR,
)

print(
    "Resolved Config Path :",
    resolved_config_path,
)

# ============================================================
# Configuration
# ============================================================


MODEL_RELAY = "Bus1_Ln12A"

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "compact_mlp_pairwise"
    / "models"
    / f"model_{MODEL_RELAY}.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "compact_mlp_pairwise"
    / "plots"
    / "compact_mlp_pairwise_tsne.pdf"
)

SEED = 42

WINDOWS_PER_CLASS = 150

BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Dataset / relay mapping
# ============================================================

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


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Locate project / dataset
# ============================================================



print("=" * 70)
print("Compact MLP t-SNE Analysis")
print("=" * 70)

print("Project root :", PROJECT_ROOT)
print("Dataset      :", DATASET_DIR)
print("Model        :", MODEL_PATH)
print("Device       :", DEVICE)


# ============================================================
# Load dataset
# ============================================================

from hydra import initialize, compose
from psp_helper.api import load_local_windowed_dataset


print("\nLoading dataset...")

with initialize(
    version_base=None,
    config_path=resolved_config_path,
):

    cfg = compose(
        config_name="main-config.yaml",
        overrides=[
            "dataset=hv_double_line_110kv"
        ],
    )

    cfg.dataset.dataset_directory = str(
        DATASET_DIR
    )

    cfg.dataset.tmp_data_directory = str(
        PROJECT_ROOT
    )
    cfg.paths.benchmark_root = str(
    DATASET_DIR)

    print("\nDataset directory:", cfg.dataset.dataset_directory)
    print("Temporary data directory:", cfg.dataset.tmp_data_directory)

    cfg.window_extraction.window_length = 0.05

    cfg.window_extraction.step_length_seconds = 0.005

    X_raw, y_df, _ = load_local_windowed_dataset(
        cfg
    )
    print("\nY columns:")
    print(y_df.columns.tolist())

    print("\nFirst rows:")
    print(y_df.head())

    if "sensor_name" in y_df.columns:
        print("\nSensor names:")
        print(y_df["sensor_name"].unique())
# ============================================================
# Sample balanced data
# ============================================================

print("\nSampling balanced windows...")

sampled_indices = []

sampled_relays = []

sampled_faults = []


for raw_name, friendly_name in SENSOR_MAP.items():

    mask_normal = (
        (y_df["sensor_name"] == raw_name)
        & (y_df["event_family"] != "switching")
        & (y_df["y_fault_present"] == 0)
    )

    mask_fault = (
        (y_df["sensor_name"] == raw_name)
        & (y_df["event_family"] != "switching")
        & (y_df["y_fault_present"] == 1)
    )

    idx_normal = np.where(
        mask_normal
    )[0]

    idx_fault = np.where(
        mask_fault
    )[0]

    if (
        len(idx_normal) < WINDOWS_PER_CLASS
        or len(idx_fault) < WINDOWS_PER_CLASS
    ):

        raise RuntimeError(
            f"Not enough samples for {friendly_name}: "
            f"normal={len(idx_normal)}, "
            f"fault={len(idx_fault)}"
        )

    sampled_normal = np.random.choice(
        idx_normal,
        WINDOWS_PER_CLASS,
        replace=False,
    )

    sampled_fault = np.random.choice(
        idx_fault,
        WINDOWS_PER_CLASS,
        replace=False,
    )

    sampled_indices.extend(
        sampled_normal
    )

    sampled_indices.extend(
        sampled_fault
    )

    sampled_relays.extend(
        [friendly_name]
        * (2 * WINDOWS_PER_CLASS)
    )

    sampled_faults.extend(
        [0] * WINDOWS_PER_CLASS
        + [1] * WINDOWS_PER_CLASS
    )


sampled_indices = np.asarray(
    sampled_indices
)

sampled_relays = np.asarray(
    sampled_relays
)

sampled_faults = np.asarray(
    sampled_faults
)


X_subset = X_raw[
    sampled_indices
]


print(
    f"Total samples: {len(X_subset)}"
)

print(
    f"Expected samples: "
    f"{8 * WINDOWS_PER_CLASS * 2}"
)


# ============================================================
# Flatten input
# ============================================================

X_flat = X_subset.reshape(
    X_subset.shape[0],
    -1,
)


# ============================================================
# Load trained Compact MLP
# ============================================================

print("\nLoading trained model...")

model = CompactMLP_FD(
    input_dim=X_flat.shape[1]
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    state_dict
)

model.to(DEVICE)
model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# Extract latent embeddings
# ============================================================

def extract_embeddings(
    model,
    X_flat,
):

    X_tensor = torch.tensor(
        X_flat,
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    embeddings = []

    probabilities = []

    with torch.no_grad():

        for (Xb,) in loader:

            Xb = Xb.to(DEVICE)

            # Compact MLP structure:
            # Linear -> ReLU
            # Linear -> ReLU
            # Linear -> ReLU
            #
            # net[:4] gives the output after
            # the second ReLU layer.

            features = model.net[:4](
                Xb
            )

            logits = model(
                Xb
            )

            probs = torch.sigmoid(
                logits
            )

            embeddings.append(
                features.cpu().numpy()
            )

            probabilities.append(
                probs.cpu().numpy()
            )

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    return embeddings, probabilities


print(
    "\nExtracting latent features..."
)

embeddings, probabilities = (
    extract_embeddings(
        model,
        X_flat,
    )
)


print(
    "Latent representation shape:",
    embeddings.shape,
)


# ============================================================
# Predictions
# ============================================================

predictions = (
    probabilities >= 0.5
).astype(int)


def get_categories(
    y_true,
    y_pred,
):

    categories = []

    for yt, yp in zip(
        y_true,
        y_pred,
    ):

        if yt == 1 and yp == 1:

            categories.append(
                "True Positive (TP)"
            )

        elif yt == 0 and yp == 0:

            categories.append(
                "True Negative (TN)"
            )

        elif yt == 0 and yp == 1:

            categories.append(
                "False Positive (FP)"
            )

        else:

            categories.append(
                "False Negative (FN)"
            )

    return np.asarray(
        categories
    )


categories = get_categories(
    sampled_faults,
    predictions,
)


print(
    "\nPrediction category breakdown:"
)

for category, count in sorted(
    Counter(categories).items()
):

    print(
        f"  {category:<22}: {count}"
    )


# ============================================================
# t-SNE
# ============================================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=SEED,
    init="pca",
    learning_rate="auto",
)

embedding_2d = tsne.fit_transform(
    embeddings
)


print(
    "t-SNE completed."
)


# ============================================================
# Plot
# ============================================================

sns.set_theme(
    style="whitegrid"
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6),
)


# ------------------------------------------------------------
# Relay colours
# ------------------------------------------------------------

relay_palette = sns.color_palette(
    "colorblind",
    len(SENSOR_MAP),
)


# ------------------------------------------------------------
# Panel 1 — Relay
# ------------------------------------------------------------

sns.scatterplot(
    x=embedding_2d[:, 0],
    y=embedding_2d[:, 1],
    hue=sampled_relays,
    palette=relay_palette,
    alpha=0.70,
    s=30,
    linewidth=0,
    ax=axes[0],
)


axes[0].set_title(
    "Latent Space Colored by Relay",
    fontsize=13,
    fontweight="bold",
)

axes[0].set_xlabel(
    "t-SNE Component 1"
)

axes[0].set_ylabel(
    "t-SNE Component 2"
)

axes[0].legend(
    title="Relay",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8,
)


# ------------------------------------------------------------
# Panel 2 — Prediction category
# ------------------------------------------------------------

category_colors = {

    "True Negative (TN)": "#7FBF7B",

    "True Positive (TP)": "#377EB8",

    "False Positive (FP)": "#E6AB02",

    "False Negative (FN)": "#D73027",
}


for category, color in category_colors.items():

    mask = (
        categories == category
    )

    if mask.sum() == 0:
        continue

    axes[1].scatter(
        embedding_2d[mask, 0],
        embedding_2d[mask, 1],
        color=color,
        alpha=0.70,
        s=30,
        linewidth=0,
        label=category,
    )


axes[1].set_title(
    "Latent Space Colored by Prediction Category",
    fontsize=13,
    fontweight="bold",
)

axes[1].set_xlabel(
    "t-SNE Component 1"
)

axes[1].set_ylabel(
    "t-SNE Component 2"
)

axes[1].legend(
    title="Prediction",
    fontsize=8,
)


# ============================================================
# Figure title
# ============================================================

fig.suptitle(
    "Compact MLP Pairwise Latent Space Analysis",
    fontsize=16,
    fontweight="bold",
)


plt.tight_layout()


# ============================================================
# Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT_PATH,
    format="pdf",
    bbox_inches="tight",
)

plt.close(fig)


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print("=" * 70)