import numpy as np


def prediction_categories(
    y_true,
    y_probability,
    threshold=0.5,
):
    """
    Convert predictions into TN/TP/FP/FN categories.
    """

    y_true = np.asarray(y_true)

    y_pred = (
        np.asarray(y_probability) >= threshold
    ).astype(int)

    categories = np.empty(
        len(y_true),
        dtype=object,
    )

    categories[
        (y_true == 1) & (y_pred == 1)
    ] = "True Positive (TP)"

    categories[
        (y_true == 0) & (y_pred == 0)
    ] = "True Negative (TN)"

    categories[
        (y_true == 0) & (y_pred == 1)
    ] = "False Positive (FP)"

    categories[
        (y_true == 1) & (y_pred == 0)
    ] = "False Negative (FN)"

    return categories