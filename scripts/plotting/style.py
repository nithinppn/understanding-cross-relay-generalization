from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# ===============================================================
# Global thesis plotting style
# ===============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],

    "mathtext.fontset": "stix",

    # Embed TrueType fonts in PDF/PS instead of Type 3 fonts.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.size": 14,

    "axes.titlesize": 17,
    "axes.labelsize": 16,

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "legend.fontsize": 13,

    "figure.titlesize": 17,

    "axes.linewidth": 1.0,

    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,

    "xtick.major.size": 5,
    "ytick.major.size": 5,

    "lines.linewidth": 1.8,

    "figure.dpi": 150,

    "savefig.dpi": 300,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    "legend.frameon": False,
})


# ===============================================================
# Colormaps
# ===============================================================

SEQUENTIAL_CMAP = "viridis"

GRAYSCALE_CMAP = "Greys"

DIVERGING_CMAP = "cividis"


# ===============================================================
# Relay display names
# ===============================================================

RELAY_DISPLAY_NAMES = {
    "Bus1_Ln12A": "R1",
    "Bus1_Ln12B": "R2",
    "Bus2_Ln12A": "R3",
    "Bus2_Ln12B": "R4",
    "Bus2_Ln23A": "R5",
    "Bus2_Ln23B": "R6",
    "Bus3_Ln23A": "R7",
    "Bus3_Ln23B": "R8",
}


def relay_display_name(relay_name):
    """
    Convert internal relay identifier to
    thesis figure label.
    """

    return RELAY_DISPLAY_NAMES.get(
        relay_name,
        relay_name,
    )


# ===============================================================
# Relay markers
# ===============================================================

RELAY_MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "*",
]


# ===============================================================
# Score formatting
# ===============================================================

def format_score(
    value,
    decimals=3,
):
    """
    Format numerical scores for thesis figures.

    Examples:
        0.85623 -> .856
        0.125   -> .125
        1.000   -> 1.000
        0.050   -> .050
    """

    if value is None:
        return "N/A"

    text = f"{float(value):.{decimals}f}"

    if text.startswith("0."):
        return text[1:]

    if text.startswith("-0."):
        return "-" + text[2:]

    return text


# ===============================================================
# Axis styling
# ===============================================================

def apply_axis_style(ax):
    """
    Apply consistent thesis styling to an axis.
    """

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        width=1.0,
        length=5,
    )

    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    title = ax.get_title()

    if title:
        ax.set_title(
            title,
            fontsize=17,
            fontweight="normal",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ===============================================================
# Figure saving
# ===============================================================

def save_figure(
    fig,
    output_path,
):
    """
    Save both high-resolution PNG and vector PDF.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # PNG
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    # Vector PDF
    pdf_path = (
        output_path.with_suffix(".pdf")
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )
