from .utils import load_results
from .loss_curves import plot_loss_curves
from .heatmaps import plot_all_heatmaps
from .tsne.pipeline import run_tsne_pipeline
from .summary import create_summary

class Plotter:

    def __init__(
        self,
        experiment_directory,
    ):

        self.experiment_directory = experiment_directory

        self.output_directory = (
            experiment_directory / "plots"
        )

        self.results = load_results(
            experiment_directory
        )

    def plot_all(self):

        print(
            "Generating loss curves..."
        )

        plot_loss_curves(
            results=self.results,
            output_directory=self.output_directory,
        )

        print(
            "Generating heatmaps..."
        )

        plot_all_heatmaps(
            results=self.results,
            output_directory=self.output_directory,
        )

        print(
            "Generating t-SNE..."
        )

        run_tsne_pipeline(
            experiment_directory=self.experiment_directory,
        )

        print(
            "Finished plotting."
        )

        print(
            "Generating summary..."
        )

        create_summary(
            experiment_directory=self.experiment_directory,
        )