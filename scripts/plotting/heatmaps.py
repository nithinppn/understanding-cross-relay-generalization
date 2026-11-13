import matplotlib.pyplot as plt
import numpy as np

from .style import (
    apply_axis_style,
    format_score,
    relay_display_name,
    save_figure,
)


def plot_heatmap(
    results,
    metric,
    title,
    filename,
    output_directory,
    cmap="Blues",
):
    """
    Plot a train-relay × test-relay heatmap.

    Internal relay identifiers are converted to
    R1-R8 only for display.
    """

    evaluation = results["evaluation"]

    train_relays = list(
        evaluation.keys()
    )

    test_relays = list(
        evaluation[train_relays[0]].keys()
    )

    matrix = np.zeros(
        (
            len(train_relays),
            len(test_relays),
        )
    )

    # ---------------------------------------------------------------
    # Build metric matrix
    # ---------------------------------------------------------------

    for i, train_relay in enumerate(
        train_relays
    ):

        for j, test_relay in enumerate(
            test_relays
        ):

            metrics = evaluation[
                train_relay
            ][test_relay]

            if metric == "fpr":

                fp = metrics["fp"]
                tn = metrics["tn"]

                value = (
                    fp / (fp + tn)
                    if (fp + tn) > 0
                    else 0.0
                )

            elif metric == "fnr":

                fn = metrics["fn"]
                tp = metrics["tp"]

                value = (
                    fn / (fn + tp)
                    if (fn + tp) > 0
                    else 0.0
                )

            else:

                value = metrics[metric]

            matrix[i, j] = value

    # ---------------------------------------------------------------
    # Colour scale
    # ---------------------------------------------------------------

    data_min = matrix.min()
    data_max = matrix.max()

    data_range = data_max - data_min

    if data_range == 0:
        padding = 0.05
    else:
        padding = 0.1 * data_range

    vmin = data_min - padding
    vmax = data_max + padding

    # Keep the scale within the valid metric range.
    vmin = max(0.0, vmin)
    vmax = min(1.0, vmax)

    # ---------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.2, 7.4),
    )

    image = ax.imshow(
        matrix,
        cmap=cmap,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    # ---------------------------------------------------------------
    # Colorbar
    # ---------------------------------------------------------------

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        title,
        fontsize=14,
    )

    cbar.ax.tick_params(
        labelsize=12,
    )

    # ---------------------------------------------------------------
    # Relay labels
    # ---------------------------------------------------------------

    display_test_relays = [
        relay_display_name(
            relay
        )
        for relay in test_relays
    ]

    display_train_relays = [
        relay_display_name(
            relay
        )
        for relay in train_relays
    ]

    ax.set_xticks(
        np.arange(
            len(test_relays)
        )
    )

    ax.set_yticks(
        np.arange(
            len(train_relays)
        )
    )

    ax.set_xticklabels(
        display_test_relays,
        rotation=0,
        ha="center",
    )

    ax.set_yticklabels(
        display_train_relays,
    )

    # ---------------------------------------------------------------
    # Axis labels
    # ---------------------------------------------------------------

    ax.set_xlabel(
        "Test Relay",
        fontsize=14,
    )

    ax.set_ylabel(
        "Training Relay",
        fontsize=14,
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="normal",
        pad=12,
    )

    # ---------------------------------------------------------------
    # Cell annotations
    # ---------------------------------------------------------------

    norm = plt.Normalize(
        vmin,
        vmax,
    )

    cmap_object = plt.get_cmap(
        cmap
    )

    for i in range(
        len(train_relays)
    ):

        for j in range(
            len(test_relays)
        ):

            value = matrix[i, j]

            # Determine luminance of the cell
            # so the annotation remains readable.
            rgba = cmap_object(
                norm(value)
            )

            r, g, b = rgba[:3]

            luminance = (
                0.2126 * r
                + 0.7152 * g
                + 0.0722 * b
            )

            text_color = (
                "black"
                if luminance > 0.55
                else "white"
            )

            ax.text(
                j,
                i,
                format_score(value),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="normal",
                color=text_color,
            )

            # Highlight self-relay performance.
            if i == j:

                ax.add_patch(
                    plt.Rectangle(
                        (
                            j - 0.5,
                            i - 0.5,
                        ),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )

    # ---------------------------------------------------------------
    # General axis styling
    # ---------------------------------------------------------------

    apply_axis_style(ax)

    # Keep all four sides visible for the matrix.
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    save_figure(
        fig,
        output_directory / filename,
    )


def plot_all_heatmaps(
    results,
    output_directory,
):

    plot_heatmap(
        results,
        metric="accuracy",
        title="Accuracy",
        filename="accuracy_heatmap.png",
        output_directory=output_directory,
    )

    plot_heatmap(
        results,
        metric="f1",
        title="F1 Score",
        filename="f1_heatmap.png",
        output_directory=output_directory,
    )

    plot_heatmap(
        results,
        metric="auc",
        title="ROC-AUC",
        filename="auc_heatmap.png",
        output_directory=output_directory,
    )

    plot_heatmap(
        results,
        metric="fpr",
        title="False Positive Rate",
        filename="fpr_heatmap.png",
        output_directory=output_directory,
    )

    plot_heatmap(
        results,
        metric="fnr",
        title="False Negative Rate",
        filename="fnr_heatmap.png",
        output_directory=output_directory,
    )
