import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATASET_DIR = Path(
    os.environ.get(
        "EVENTBENCH_DATASET_DIR",
        PROJECT_ROOT / "hv_double_line_110kv",
    )
)


if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset directory not found:\n{DATASET_DIR}\n\n"
        "Set EVENTBENCH_DATASET_DIR correctly."
    )


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


ALL_RELAYS = list(SENSOR_MAP.values())