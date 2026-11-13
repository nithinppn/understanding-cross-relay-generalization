from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


def create_simulation_split(
    sample_ids,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list]:
    """
    Create ONE global train/validation/test split based on simulation IDs.

    The returned split is independent of relay. Every relay must reuse
    exactly this mapping.
    """

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    sample_ids = pd.Series(sample_ids).drop_duplicates().tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(sample_ids)

    n = len(sample_ids)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train:n_train + n_val]
    test_ids = sample_ids[n_train + n_val:]

    split = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    _validate_split(split)

    return split


def _validate_split(split: dict[str, list]) -> None:
    """Verify that the three sets are disjoint and exhaustive."""

    train = set(split["train"])
    val = set(split["val"])
    test = set(split["test"])

    if train & val:
        raise RuntimeError("Train/validation simulation overlap detected.")

    if train & test:
        raise RuntimeError("Train/test simulation overlap detected.")

    if val & test:
        raise RuntimeError("Validation/test simulation overlap detected.")


def save_simulation_split(
    split: dict[str, list],
    path: str | Path,
) -> None:
    """Save the global simulation split so every experiment can reuse it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(split, f, indent=2)


def load_simulation_split(
    path: str | Path,
) -> dict[str, set]:
    """Load a previously generated global simulation split."""

    path = Path(path)

    with path.open("r") as f:
        split = json.load(f)

    _validate_split(split)

    return {
        "train": set(split["train"]),
        "val": set(split["val"]),
        "test": set(split["test"]),
    }


def get_split_mask(
    sample_ids,
    split_ids: set,
) -> np.ndarray:
    """Return a boolean mask selecting simulations belonging to a split."""

    sample_ids = np.asarray(sample_ids)

    return np.isin(sample_ids, list(split_ids))