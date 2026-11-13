from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from psp_helper.api import load_local_windowed_dataset

from .config import PROJECT_ROOT, DATASET_DIR


def _build_config():

    absolute_config_path = (
        PROJECT_ROOT
        / "helper"
        / "power-grid-and-ai-helper-functions"
        / "config"
    )

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(
        version_base=None,
        config_dir=str(absolute_config_path),
    ):

        cfg = compose(
            config_name="main-config.yaml",
            overrides=[
                "dataset=hv_double_line_110kv",
            ],
        )

        cfg.dataset.dataset_directory = str(DATASET_DIR)

        cfg.dataset.tmp_data_directory = str(PROJECT_ROOT)

        cfg.window_extraction.window_length = 0.05

        cfg.window_extraction.step_length_seconds = 0.005

        cfg.paths.benchmark_root = str(DATASET_DIR)

        return cfg


def load_dataset():

    cfg = _build_config()

    X_local, y_local, _ = load_local_windowed_dataset(cfg)

    return X_local, y_local