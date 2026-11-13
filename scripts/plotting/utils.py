import json

from pathlib import Path


def load_results(
    experiment_dir: Path,
):

    results_file = experiment_dir / "results.json"

    with open(
        results_file,
        "r",
    ) as f:

        return json.load(f)


def save_figure(
    figure,
    output_path: Path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved {output_path}"
    )