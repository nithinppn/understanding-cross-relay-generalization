from pathlib import Path

import matplotlib.pyplot as plt

from ..style import (
    RELAY_MARKERS,
    apply_axis_style,
    relay_display_name,
    save_figure,
)


# ===============================================================
# Prediction categories
# ===============================================================

CATEGORY_STYLES = {
    "True Negative (TN)": {
        "color": "0.80",
        "marker": "o",
    },

    "True Positive (TP)": {
        "color": "0.20",
        "marker": "s",
    },

    "False Positive (FP)": {
        "color": "0.55",
        "marker": "^",
    },

    "False Negative (FN)": {
        "color": "0.35",
        "marker": "X",
    },
}


def plot_tsne(
    embedding_2d,
    relay,
    y_true,
    categories,
    output_path,
    title,
):
    """
    Create a two-panel t-SNE visualization.

    Left:
        Points are differentiated by their source relay.

    Right:
        Points are differentiated by prediction category.

    Relay identity is encoded using both marker shape
    and grayscale-safe color.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------
    # Figure
    # -----------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    # ===========================================================
    # Panel 1 — Source relay
    # ===========================================================

    relays = list(
        dict.fromkeys(relay)
    )

    cmap = plt.get_cmap(
        "cividis"
    )

    n_relays = len(relays)

    relay_colors = [
        cmap(
            0.10
            + 0.80 * i / max(
                n_relays - 1,
                1,
            )
        )
        for i in range(n_relays)
    ]

    for i, relay_name in enumerate(
        relays
    ):

        mask = relay == relay_name

        if mask.sum() == 0:
            continue

        axes[0].scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],

            color=relay_colors[i],

            marker=RELAY_MARKERS[
                i % len(RELAY_MARKERS)
            ],

            alpha=0.75,

            s=38,

            edgecolors="black",

            linewidths=0.35,

            label=relay_display_name(
                relay_name
            ),
        )

    axes[0].set_title(
        "Latent Space Colored by Relay",
        fontsize=16,
        fontweight="normal",
    )

    axes[0].set_xlabel(
        "t-SNE Component 1",
        fontsize=14,
    )

    axes[0].set_ylabel(
        "t-SNE Component 2",
        fontsize=14,
    )

    axes[0].legend(
        title="Relay",
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
        fontsize=11,
        title_fontsize=12,
        frameon=False,
    )

    apply_axis_style(
        axes[0]
    )

    # ===========================================================
    # Panel 2 — Prediction category
    # ===========================================================

    for category, style in (
        CATEGORY_STYLES.items()
    ):

        mask = (
            categories == category
        )

        if mask.sum() == 0:
            continue

        axes[1].scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],

            color=style["color"],

            marker=style["marker"],

            alpha=0.75,

            s=38,

            edgecolors="black",

            linewidths=0.35,

            label=category,
        )

    axes[1].set_title(
        "Latent Space Colored by Prediction Category",
        fontsize=16,
        fontweight="normal",
    )

    axes[1].set_xlabel(
        "t-SNE Component 1",
        fontsize=14,
    )

    axes[1].set_ylabel(
        "t-SNE Component 2",
        fontsize=14,
    )

    axes[1].legend(
        title="Prediction",
        fontsize=11,
        title_fontsize=12,
        frameon=False,
    )

    apply_axis_style(
        axes[1]
    )

    # ===========================================================
    # Figure title
    # ===========================================================

    fig.suptitle(
        title,
        fontsize=17,
        fontweight="normal",
        y=1.01,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.98,
        ]
    )

    # -----------------------------------------------------------
    # Save PNG + PDF
    # -----------------------------------------------------------

    save_figure(
        fig,
        output_path,
    )
