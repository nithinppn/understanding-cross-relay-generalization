import math

import matplotlib.pyplot as plt

from .style import (
    apply_axis_style,
    relay_display_name,
    save_figure,
)


def plot_loss_curves(
    results,
    output_directory,
):

    training_results = results["training"]

    n_relays = len(
        training_results
    )

    cols = 2
    rows = math.ceil(
        n_relays / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 4 * rows),
        sharex=True,
    )

    axes = axes.flatten()

    for ax, (
        relay_name,
        relay_result,
    ) in zip(
        axes,
        training_results.items(),
    ):

        history = relay_result[
            "history"
        ]

        # -------------------------------------------------------
        # Training loss
        # -------------------------------------------------------

        ax.plot(
            history["train_loss"],
            color="black",
            linestyle="-",
            linewidth=1.8,
            label="Train",
        )

        # -------------------------------------------------------
        # Validation loss
        # -------------------------------------------------------

        ax.plot(
            history["validation_loss"],
            color="0.45",
            linestyle="--",
            linewidth=1.8,
            label="Validation",
        )

        # -------------------------------------------------------
        # Relay label
        # -------------------------------------------------------

        ax.set_title(
            relay_display_name(
                relay_name
            ),
            fontsize=15,
            fontweight="normal",
        )

        # -------------------------------------------------------
        # Axis labels
        # -------------------------------------------------------

        ax.set_xlabel(
            "Epoch",
            fontsize=14,
        )

        ax.set_ylabel(
            "Binary Cross-Entropy Loss",
            fontsize=14,
        )

        # -------------------------------------------------------
        # Log scale
        # -------------------------------------------------------

        ax.set_yscale("log")

        # -------------------------------------------------------
        # Grid
        # -------------------------------------------------------

        ax.grid(
            True,
            which="major",
            linestyle=":",
            linewidth=0.8,
            alpha=0.5,
        )

        # Avoid excessive minor-grid clutter.
        ax.grid(
            False,
            which="minor",
        )

        # -------------------------------------------------------
        # Legend
        # -------------------------------------------------------

        ax.legend(
            fontsize=11,
            loc="best",
        )

        # -------------------------------------------------------
        # Common styling
        # -------------------------------------------------------

        apply_axis_style(ax)

    # -----------------------------------------------------------
    # Remove unused subplots
    # -----------------------------------------------------------

    for ax in axes[n_relays:]:

        fig.delaxes(ax)

    # -----------------------------------------------------------
    # Figure title
    # -----------------------------------------------------------

    fig.suptitle(
        "Training and Validation Loss Curves",
        fontsize=17,
        fontweight="normal",
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.985,
        ]
    )

    # -----------------------------------------------------------
    # Save
    # -----------------------------------------------------------

    save_figure(
        fig,
        output_directory
        / "loss_curves.png",
    )
