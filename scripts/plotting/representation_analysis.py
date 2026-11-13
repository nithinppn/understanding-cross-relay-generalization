import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------
# Linear relay-ID probe
# ---------------------------------------------------------------------

def relay_id_probe(
    embeddings,
    relay,
    sample_id,
    random_state=42,
):
    """
    Measure how easily relay identity can be recovered
    from the latent representation.

    Uses standardized embeddings and stratified 5-fold
    group-aware CV, where all windows from the same
    simulation remain in the same fold.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    relay = np.asarray(relay)
    sample_id = np.asarray(sample_id)

    if not (
        len(embeddings)
        == len(relay)
        == len(sample_id)
    ):
        raise ValueError(
            "embeddings, relay, and sample_id "
            "must have the same length."
        )

    unique_relays = np.unique(relay)

    predictions = np.empty(
        len(relay),
        dtype=relay.dtype,
    )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    for train_idx, test_idx in cv.split(
        embeddings,
        relay,
        groups=sample_id,
    ):

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=random_state,
            ),
        )

        classifier.fit(
            embeddings[train_idx],
            relay[train_idx],
        )

        predictions[test_idx] = (
            classifier.predict(
                embeddings[test_idx]
            )
        )

    accuracy = accuracy_score(
        relay,
        predictions,
    )

    cm = confusion_matrix(
        relay,
        predictions,
        labels=unique_relays,
    )

    chance_accuracy = (
        1.0 / len(unique_relays)
    )

    return {
        "accuracy": float(accuracy),
        "chance_accuracy": float(
            chance_accuracy
        ),
        "confusion_matrix": cm,
        "labels": unique_relays,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------
# Independent bus-ID linear probe
# ---------------------------------------------------------------------

def bus_id_probe(
    embeddings,
    relay,
    sample_id,
    random_state=42,
):
    """
    Measure how easily physical bus identity can be recovered
    directly from the latent representation.

    Uses standardized embeddings and stratified 5-fold
    group-aware CV, where all windows from the same
    simulation remain in the same fold.

    Bus mapping:
        Bus1_Ln12A, Bus1_Ln12B             -> Bus 1
        Bus2_Ln12A, Bus2_Ln12B,
        Bus2_Ln23A, Bus2_Ln23B             -> Bus 2
        Bus3_Ln23A, Bus3_Ln23B             -> Bus 3
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    relay = np.asarray(relay)
    sample_id = np.asarray(sample_id)

    if not (
        len(embeddings)
        == len(relay)
        == len(sample_id)
    ):
        raise ValueError(
            "embeddings, relay, and sample_id "
            "must have the same length."
        )

    relay_to_bus = {
        "Bus1_Ln12A": "Bus 1",
        "Bus1_Ln12B": "Bus 1",

        "Bus2_Ln12A": "Bus 2",
        "Bus2_Ln12B": "Bus 2",
        "Bus2_Ln23A": "Bus 2",
        "Bus2_Ln23B": "Bus 2",

        "Bus3_Ln23A": "Bus 3",
        "Bus3_Ln23B": "Bus 3",
    }

    bus = np.asarray([
        relay_to_bus[r]
        for r in relay
    ])

    labels = np.array([
        "Bus 1",
        "Bus 2",
        "Bus 3",
    ])

    predictions = np.empty(
        len(bus),
        dtype=bus.dtype,
    )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    for train_idx, test_idx in cv.split(
        embeddings,
        bus,
        groups=sample_id,
    ):

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=random_state,
            ),
        )

        classifier.fit(
            embeddings[train_idx],
            bus[train_idx],
        )

        predictions[test_idx] = (
            classifier.predict(
                embeddings[test_idx]
            )
        )

    accuracy = accuracy_score(
        bus,
        predictions,
    )

    cm = confusion_matrix(
        bus,
        predictions,
        labels=labels,
    )

    return {
        "accuracy": float(accuracy),
        "chance_accuracy": 1.0 / len(labels),
        "confusion_matrix": cm,
        "labels": labels,
        "predictions": predictions,
    }

# ---------------------------------------------------------------------
# Bus-group relay probe
# ---------------------------------------------------------------------

def group_relay_predictions_by_bus(
    relay,
    predictions,
):
    """
    Aggregate relay-ID predictions into physical bus groups.

    R1, R2             -> Bus 1
    R3, R4, R5, R6     -> Bus 2
    R7, R8             -> Bus 3
    """

    relay_to_bus = {
        "Bus1_Ln12A": "Bus 1",
        "Bus1_Ln12B": "Bus 1",

        "Bus2_Ln12A": "Bus 2",
        "Bus2_Ln12B": "Bus 2",
        "Bus2_Ln23A": "Bus 2",
        "Bus2_Ln23B": "Bus 2",

        "Bus3_Ln23A": "Bus 3",
        "Bus3_Ln23B": "Bus 3",
    }

    true_bus = np.array([
        relay_to_bus[r]
        for r in relay
    ])

    predicted_bus = np.array([
        relay_to_bus[r]
        for r in predictions
    ])

    labels = np.array([
        "Bus 1",
        "Bus 2",
        "Bus 3",
    ])

    cm = confusion_matrix(
        true_bus,
        predicted_bus,
        labels=labels,
    )

    accuracy = accuracy_score(
        true_bus,
        predicted_bus,
    )

    return {
        "confusion_matrix": cm,
        "labels": labels,
        "accuracy": float(accuracy),
        "chance_accuracy": 1.0 / 3.0,
    }

# ---------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------

def _median_bandwidth(
    X,
    Y,
):
    """
    Median heuristic for the RBF kernel bandwidth.
    """

    combined = np.concatenate(
        [X, Y],
        axis=0,
    )

    # Pairwise squared distances.
    diff = (
        combined[:, None, :]
        - combined[None, :, :]
    )

    distances = np.sum(
        diff * diff,
        axis=-1,
    )

    distances = distances[
        np.triu_indices(
            len(combined),
            k=1,
        )
    ]

    distances = distances[
        distances > 0
    ]

    if len(distances) == 0:
        return 1.0

    return float(
        np.sqrt(
            0.5
            * np.median(distances)
        )
    )


def _rbf_kernel(
    X,
    Y,
    bandwidth,
):
    """
    RBF kernel matrix.
    """

    diff = (
        X[:, None, :]
        - Y[None, :, :]
    )

    squared_distance = np.sum(
        diff * diff,
        axis=-1,
    )

    return np.exp(
        -squared_distance
        / (
            2.0
            * bandwidth
            * bandwidth
        )
    )


def compute_mmd(
    X,
    Y,
    bandwidth=None,
):
    """
    Compute biased MMD^2 using an RBF kernel.

    MMD is calculated in the original latent space,
    never on t-SNE coordinates.

    Parameters
    ----------
    X : array-like
        Source embeddings.

    Y : array-like
        Target embeddings.

    bandwidth : float or None
        RBF kernel bandwidth. If None, the median heuristic
        is computed from X and Y. For pairwise analysis,
        a source-specific bandwidth should be supplied so
        that the same bandwidth is reused across all targets
        of a given source model.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    Y = np.asarray(
        Y,
        dtype=np.float32,
    )

    if bandwidth is None:

        bandwidth = _median_bandwidth(
            X,
            Y,
        )

    bandwidth = float(bandwidth)

    if bandwidth <= 0.0:
        raise ValueError(
            "MMD bandwidth must be positive."
        )

    K_xx = _rbf_kernel(
        X,
        X,
        bandwidth,
    )

    K_yy = _rbf_kernel(
        Y,
        Y,
        bandwidth,
    )

    K_xy = _rbf_kernel(
        X,
        Y,
        bandwidth,
    )

    mmd_squared = (
        K_xx.mean()
        + K_yy.mean()
        - 2.0 * K_xy.mean()
    )

    return {
        "mmd_squared": float(
            max(
                mmd_squared,
                0.0,
            )
        ),
        "bandwidth": float(
            bandwidth
        ),
    }

# ---------------------------------------------------------------------
# Pairwise MMD
# ---------------------------------------------------------------------

def compute_pairwise_mmd(
    embeddings,
    relay,
):
    """
    Compute MMD between every pair of relays.

    Returns one entry for every unordered relay pair.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    relay = np.asarray(relay)

    relays = np.unique(relay)

    results = {}

    for i, relay_a in enumerate(relays):

        X_a = embeddings[
            relay == relay_a
        ]

        for relay_b in relays[i + 1:]:

            X_b = embeddings[
                relay == relay_b
            ]

            result = compute_mmd(
                X_a,
                X_b,
            )

            key = (
                f"{relay_a}__{relay_b}"
            )

            results[key] = {
                "relay_a": str(relay_a),
                "relay_b": str(relay_b),
                **result,
            }

    return results


# ---------------------------------------------------------------------
# MMD ↔ FPR correlation
# ---------------------------------------------------------------------

def correlate_mmd_with_fpr(
    mmd_records,
):
    """
    Correlate latent MMD with cross-relay FPR.

    Each record must contain:
        mmd
        fpr
    """

    mmd_values = np.asarray(
        [
            record["mmd"]
            for record in mmd_records
        ],
        dtype=float,
    )

    fpr_values = np.asarray(
        [
            record["fpr"]
            for record in mmd_records
        ],
        dtype=float,
    )

    pearson_r, pearson_p = pearsonr(
        mmd_values,
        fpr_values,
    )

    spearman_rho, spearman_p = (
        spearmanr(
            mmd_values,
            fpr_values,
        )
    )

    return {
        "n_pairs": len(mmd_values),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(
            spearman_rho
        ),
        "spearman_p": float(
            spearman_p
        ),
    }

# ---------------------------------------------------------------------
# MMD ↔ FNR correlation
# ---------------------------------------------------------------------

def correlate_mmd_with_fnr(
    mmd_records,
):
    """
    Correlate latent MMD with cross-relay FNR.

    Each record must contain:
        mmd
        fnr
    """

    mmd_values = np.asarray(
        [
            record["mmd"]
            for record in mmd_records
        ],
        dtype=float,
    )

    fnr_values = np.asarray(
        [
            record["fnr"]
            for record in mmd_records
        ],
        dtype=float,
    )

    pearson_r, pearson_p = pearsonr(
        mmd_values,
        fnr_values,
    )

    spearman_rho, spearman_p = spearmanr(
        mmd_values,
        fnr_values,
    )

    return {
        "n_pairs": len(mmd_values),
        "pearson_r": float(
            pearson_r
        ),
        "pearson_p": float(
            pearson_p
        ),
        "spearman_rho": float(
            spearman_rho
        ),
        "spearman_p": float(
            spearman_p
        ),
    }

# ---------------------------------------------------------------------
# MMD vs FPR plot
# ---------------------------------------------------------------------

from pathlib import Path

path = Path("scripts/plotting/representation_analysis.py")

text = path.read_text()

start = text.index("def plot_mmd_vs_fpr(")
end = text.index(
    "# ---------------------------------------------------------------------\n# Relay-ID confusion matrix",
    start,
)

def plot_mmd_vs_fpr(
    records,
    correlation,
    output_path,
):
    """
    Plot latent MMD against cross-relay FPR.

    MMD is computed in the original latent space,
    not on t-SNE coordinates.
    """

    from .style import (
        apply_axis_style,
        format_score,
        save_figure,
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------
    # Data
    # -----------------------------------------------------------

    x = np.asarray(
        [
            record["mmd"]
            for record in records
        ],
        dtype=float,
    )

    y = np.asarray(
        [
            record["fpr"]
            for record in records
        ],
        dtype=float,
    )

    # -----------------------------------------------------------
    # Figure
    # -----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    ax.scatter(
        x,
        y,
        color="0.25",
        edgecolors="black",
        linewidths=0.6,
        s=55,
        alpha=0.80,
    )

    # -----------------------------------------------------------
    # Linear trend
    # -----------------------------------------------------------

    if len(x) >= 2:

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            200,
        )

        y_line = (
            slope * x_line
            + intercept
        )

        ax.plot(
            x_line,
            y_line,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Linear trend",
        )

    # -----------------------------------------------------------
    # Labels
    # -----------------------------------------------------------

    ax.set_xlabel(
        r"Latent-space MMD$^2$",
        fontsize=14,
    )

    ax.set_ylabel(
        "Cross-relay FPR",
        fontsize=14,
    )

    ax.set_title(
        "Latent Distribution Shift vs Cross-Relay FPR",
        fontsize=16,
        fontweight="normal",
        pad=12,
    )

    # -----------------------------------------------------------
    # Correlation statistics
    # -----------------------------------------------------------

    pearson_r = correlation[
        "pearson_r"
    ]

    spearman_rho = correlation[
        "spearman_rho"
    ]

    annotation = (
        f"Pearson r = "
        f"{format_score(pearson_r)}\\n"
        f"Spearman ρ = "
        f"{format_score(spearman_rho)}"
    )

    ax.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        bbox=dict(
            facecolor="white",
            edgecolor="0.5",
            linewidth=0.8,
            boxstyle="round,pad=0.35",
        ),
    )

    # -----------------------------------------------------------
    # Grid
    # -----------------------------------------------------------

    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.8,
        alpha=0.45,
    )

    # -----------------------------------------------------------
    # Styling
    # -----------------------------------------------------------

    apply_axis_style(
        ax
    )

    if len(x) >= 2:
        ax.legend(
            fontsize=11,
            frameon=False,
        )

    fig.tight_layout()

    # -----------------------------------------------------------
    # Save PNG + PDF
    # -----------------------------------------------------------

    save_figure(
        fig,
        output_path,
    )

# ---------------------------------------------------------------------
# MMD vs FNR plot
# ---------------------------------------------------------------------

def plot_mmd_vs_fnr(
    records,
    correlation,
    output_path,
):
    """
    Plot latent MMD against cross-relay FNR.

    MMD is computed in the original latent space,
    not on t-SNE coordinates.
    """

    from .style import (
        apply_axis_style,
        format_score,
        save_figure,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------
    # Data
    # -----------------------------------------------------------

    x = np.asarray(
        [
            record["mmd"]
            for record in records
        ],
        dtype=float,
    )

    y = np.asarray(
        [
            record["fnr"]
            for record in records
        ],
        dtype=float,
    )

    # -----------------------------------------------------------
    # Figure
    # -----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    ax.scatter(
        x,
        y,
        color="0.25",
        edgecolors="black",
        linewidths=0.6,
        s=55,
        alpha=0.80,
    )

    # -----------------------------------------------------------
    # Linear trend
    # -----------------------------------------------------------

    if len(x) >= 2:

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            200,
        )

        y_line = (
            slope * x_line
            + intercept
        )

        ax.plot(
            x_line,
            y_line,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Linear trend",
        )

    # -----------------------------------------------------------
    # Labels
    # -----------------------------------------------------------

    ax.set_xlabel(
        r"Latent-space MMD$^2$",
        fontsize=14,
    )

    ax.set_ylabel(
        "Cross-relay FNR",
        fontsize=14,
    )

    ax.set_title(
        "Latent Distribution Shift vs Cross-Relay FNR",
        fontsize=16,
        fontweight="normal",
        pad=12,
    )

    # -----------------------------------------------------------
    # Correlation statistics
    # -----------------------------------------------------------

    pearson_r = correlation[
        "pearson_r"
    ]

    spearman_rho = correlation[
        "spearman_rho"
    ]

    annotation = (
        f"Pearson r = "
        f"{format_score(pearson_r)}\n"
        f"Spearman ρ = "
        f"{format_score(spearman_rho)}"
    )

    ax.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        bbox=dict(
            facecolor="white",
            edgecolor="0.5",
            linewidth=0.8,
            boxstyle="round,pad=0.35",
        ),
    )

    # -----------------------------------------------------------
    # Grid
    # -----------------------------------------------------------

    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.8,
        alpha=0.45,
    )

    # -----------------------------------------------------------
    # Styling
    # -----------------------------------------------------------

    apply_axis_style(ax)

    if len(x) >= 2:
        ax.legend(
            fontsize=11,
            frameon=False,
        )

    fig.tight_layout()

    # -----------------------------------------------------------
    # Save PNG + PDF
    # -----------------------------------------------------------

    save_figure(
        fig,
        output_path,
    )

    
# ---------------------------------------------------------------------
# Relay-ID confusion matrix
# ---------------------------------------------------------------------

def plot_relay_probe_confusion(
    probe_result,
    output_path,
):
    """
    Plot relay-ID probe confusion matrix.

    Relay identifiers are converted to R1-R8
    for display only.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    from .style import (
        apply_axis_style,
        relay_display_name,
        save_figure,
    )

    # -----------------------------------------------------------
    # Data
    # -----------------------------------------------------------

    matrix = np.asarray(
        probe_result[
            "confusion_matrix"
        ]
    )

    labels = [
        relay_display_name(
            relay
        )
        for relay in probe_result[
            "labels"
        ]
    ]

    # -----------------------------------------------------------
    # Figure
    # -----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.5, 7.5)
    )

    image = ax.imshow(
        matrix,
        cmap="Blues",
        aspect="equal",
    )

    # -----------------------------------------------------------
    # Colorbar
    # -----------------------------------------------------------

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Number of Samples",
        fontsize=14,
    )

    cbar.ax.tick_params(
        labelsize=12,
    )

    # -----------------------------------------------------------
    # Axis labels
    # -----------------------------------------------------------

    positions = np.arange(
        len(labels)
    )

    ax.set_xticks(
        positions
    )

    ax.set_yticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        fontsize=12,
    )

    ax.set_yticklabels(
        labels,
        fontsize=12,
    )

    ax.set_xlabel(
        "Predicted Relay",
        fontsize=14,
    )

    ax.set_ylabel(
        "True Relay",
        fontsize=14,
    )

    ax.set_title(
        "Linear Probe: Relay Identification",
        fontsize=16,
        fontweight="normal",
        pad=12,
    )

    # -----------------------------------------------------------
    # Cell annotations
    # -----------------------------------------------------------

    threshold = (
        matrix.max() / 2.0
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            value = int(
                matrix[i, j]
            )

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                color=text_color,
            )

    # -----------------------------------------------------------
    # Grid between cells
    # -----------------------------------------------------------

    ax.set_xticks(
        np.arange(
            -0.5,
            len(labels),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(labels),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.8,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # -----------------------------------------------------------
    # Styling
    # -----------------------------------------------------------

    apply_axis_style(
        ax
    )

    # Keep matrix borders visible.
    ax.spines["top"].set_visible(
        True
    )

    ax.spines["right"].set_visible(
        True
    )

    # -----------------------------------------------------------
    # Save PNG + PDF
    # -----------------------------------------------------------

    save_figure(
        fig,
        output_path,
    )

# ---------------------------------------------------------------------
# Bus-group relay probe confusion matrix
# ---------------------------------------------------------------------

def plot_bus_group_probe_confusion(
    bus_result,
    output_path,
):
    """
    Plot relay-ID probe confusion matrix aggregated
    by physical bus location.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=bus_result[
            "confusion_matrix"
        ],
        display_labels=bus_result[
            "labels"
        ],
    )

    display.plot(
        ax=ax,
        xticks_rotation=0,
        colorbar=True,
        cmap="Blues",
    )


    ax.set_xlabel(
        "Predicted Bus",
        fontsize=13,
    )

    ax.set_ylabel(
        "True Bus",
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    pdf_path = output_path.with_suffix(
        ".pdf"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved bus-group probe matrix: "
        f"{output_path}"
    )

    print(
        f"Saved bus-group probe matrix: "
        f"{pdf_path}"
    )

# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_representation_results(
    results,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"Saved representation metrics: "
        f"{output_path}"
    )
