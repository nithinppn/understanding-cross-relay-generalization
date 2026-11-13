import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(
    y_true,
    y_probability,
    threshold=0.5,
):
    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )


def _linear_probe(
    embeddings,
    labels,
    random_state=42,
):
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    labels = np.asarray(labels)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    predictions = np.empty(
        len(labels),
        dtype=labels.dtype,
    )

    for train_idx, test_idx in cv.split(
        embeddings,
        labels,
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
            labels[train_idx],
        )

        predictions[test_idx] = (
            classifier.predict(
                embeddings[test_idx]
            )
        )

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }


def fault_separability(
    embeddings,
    y_true,
    random_state=42,
):
    return _linear_probe(
        embeddings,
        y_true,
        random_state,
    )


def domain_confusion(
    embeddings,
    relay,
    random_state=42,
):
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    relay = np.asarray(relay)

    unique_relays = np.unique(relay)
    n_domains = len(unique_relays)

    if n_domains < 2:
        raise ValueError(
            "Domain confusion requires at least "
            "two relay domains."
        )

    probe = _linear_probe(
        embeddings,
        relay,
        random_state,
    )

    accuracy = probe["accuracy"]

    chance_accuracy = (
        1.0 / n_domains
    )

    confusion = 1.0 - (
        (accuracy - chance_accuracy)
        / (1.0 - chance_accuracy)
    )

    return {
        "domain_accuracy": float(accuracy),
        "chance_accuracy": float(
            chance_accuracy
        ),
        "domain_confusion": float(
            np.clip(
                confusion,
                0.0,
                1.0,
            )
        ),
    }
