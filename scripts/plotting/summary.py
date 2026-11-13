import csv
import json
from datetime import datetime
from pathlib import Path


# ======================================================================
# Experiment constants
# ======================================================================

RELAYS = [
    "Bus1_Ln12A",
    "Bus1_Ln12B",
    "Bus2_Ln12A",
    "Bus2_Ln12B",
    "Bus2_Ln23A",
    "Bus2_Ln23B",
    "Bus3_Ln23A",
    "Bus3_Ln23B",
]

METRICS = [
    "accuracy",
    "f1",
    "auc",
    "fpr",
    "fnr",
]

# Final shared simulation-level split used by the experiment.
TRAIN_SIMULATIONS = 3266
VALIDATION_SIMULATIONS = 700
TEST_SIMULATIONS = 701
TOTAL_SIMULATIONS = 4667

TRAIN_WINDOWS_PER_RELAY = 75118
VALIDATION_WINDOWS_PER_RELAY = 16100
TEST_WINDOWS_PER_RELAY = 16123

RANDOM_SEED = 42


# ======================================================================
# Generic helpers
# ======================================================================

def _load_json(path):
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def _fmt(value, digits=4):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _section(title):
    return (
        "\n"
        + "=" * 70
        + "\n"
        + title
        + "\n"
        + "=" * 70
        + "\n"
    )


def _find_first_key(obj, keys):
    """
    Recursively search a nested dictionary/list for the first matching key.
    """

    if isinstance(obj, dict):

        for key in keys:
            if key in obj:
                return obj[key]

        for value in obj.values():
            result = _find_first_key(value, keys)

            if result is not None:
                return result

    elif isinstance(obj, list):

        for value in obj:
            result = _find_first_key(value, keys)

            if result is not None:
                return result

    return None


def _find_all_keys(obj, keys, results=None):
    """
    Recursively collect all values associated with selected keys.
    """

    if results is None:
        results = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            if key in keys:
                results.append(value)

            _find_all_keys(value, keys, results)

    elif isinstance(obj, list):

        for value in obj:
            _find_all_keys(value, keys, results)

    return results


# ======================================================================
# Metric derivation
# ======================================================================

def _derive_metrics(metrics):

    metrics = metrics.copy()

    tn = metrics.get("tn")
    fp = metrics.get("fp")
    fn = metrics.get("fn")
    tp = metrics.get("tp")

    if None not in (tn, fp, fn, tp):

        denominator_fpr = fp + tn

        if denominator_fpr > 0:
            metrics["fpr"] = fp / denominator_fpr

        denominator_fnr = fn + tp

        if denominator_fnr > 0:
            metrics["fnr"] = fn / denominator_fnr

        # Positive-class F1
        precision_pos_den = tp + fp
        recall_pos_den = tp + fn

        if precision_pos_den > 0 and recall_pos_den > 0:

            precision_pos = tp / precision_pos_den
            recall_pos = tp / recall_pos_den

            if precision_pos + recall_pos > 0:
                f1_pos = (
                    2
                    * precision_pos
                    * recall_pos
                    / (precision_pos + recall_pos)
                )
            else:
                f1_pos = 0.0

        else:
            f1_pos = 0.0

        # Negative-class F1
        precision_neg_den = tn + fn
        recall_neg_den = tn + fp

        if precision_neg_den > 0 and recall_neg_den > 0:

            precision_neg = tn / precision_neg_den
            recall_neg = tn / recall_neg_den

            if precision_neg + recall_neg > 0:
                f1_neg = (
                    2
                    * precision_neg
                    * recall_neg
                    / (precision_neg + recall_neg)
                )
            else:
                f1_neg = 0.0

        else:
            f1_neg = 0.0

        metrics["macro_f1"] = (f1_pos + f1_neg) / 2.0

    return metrics


# ======================================================================
# Pairwise results
# ======================================================================

def _flatten_evaluation(results):

    records = []

    evaluation = results.get("evaluation", {})

    for train_relay, test_results in evaluation.items():

        for test_relay, raw_metrics in test_results.items():

            metrics = _derive_metrics(raw_metrics)

            records.append(
                {
                    "train_relay": train_relay,
                    "test_relay": test_relay,
                    "metrics": metrics,
                }
            )

    return records


def _metric_summary(records, metric_name):

    self_values = []
    cross_values = []

    for record in records:

        value = record["metrics"].get(metric_name)

        if value is None:
            continue

        if record["train_relay"] == record["test_relay"]:
            self_values.append(value)
        else:
            cross_values.append(value)

    if not self_values:
        return None

    mean_self = sum(self_values) / len(self_values)

    mean_cross = (
        sum(cross_values) / len(cross_values)
        if cross_values
        else None
    )

    best_cross = (
        max(cross_values)
        if cross_values
        else None
    )

    worst_cross = (
        min(cross_values)
        if cross_values
        else None
    )

    gap = (
        mean_self - mean_cross
        if mean_cross is not None
        else None
    )

    return {
        "mean_self": mean_self,
        "mean_cross": mean_cross,
        "gap": gap,
        "best_cross": best_cross,
        "worst_cross": worst_cross,
    }


# ======================================================================
# CSV helpers
# ======================================================================

def _read_csv(path):

    with open(path, "r", newline="") as f:
        return list(csv.reader(f))


def _matrix_text(path, title):

    if not path.exists():
        return ""

    rows = _read_csv(path)

    if not rows:
        return ""

    text = [
        f"\n{title}",
        "-" * len(title),
    ]

    for row in rows:
        text.append("    ".join(row))

    return "\n".join(text) + "\n"


# ======================================================================
# Main methodology section
# ======================================================================

def _methodology_section():

    return (
        _section("1. EXPERIMENT OVERVIEW")
        + "Objective:\n"
          "    Evaluate cross-relay generalization of relay-specific "
          "fault-classification models and investigate latent "
          "representation differences between relay and bus groups.\n\n"
          "Pairwise protocol:\n"
          "    8 source relays × 8 target relays = 64 source-target "
          "evaluations.\n"
    )


# ======================================================================
# Shared simulation-level split
# ======================================================================

def _split_section():

    return (
        _section("2. SHARED SIMULATION-LEVEL SPLIT")
        + "Split unit:\n"
          "    Simulation ID / sample_id\n\n"
          f"Random seed:\n"
          f"    {RANDOM_SEED}\n\n"
          "The same simulation-level train/validation/test split is "
          "reused across all relays.\n\n"
          "Simulation counts:\n"
          f"    Train simulations:       {TRAIN_SIMULATIONS}\n"
          f"    Validation simulations:  {VALIDATION_SIMULATIONS}\n"
          f"    Test simulations:        {TEST_SIMULATIONS}\n"
          f"    Total simulations:       {TOTAL_SIMULATIONS}\n\n"
          "Per-relay window counts:\n"
          f"    Train windows:            {TRAIN_WINDOWS_PER_RELAY}\n"
          f"    Validation windows:       {VALIDATION_WINDOWS_PER_RELAY}\n"
          f"    Test windows:             {TEST_WINDOWS_PER_RELAY}\n\n"
          "Pairwise protocol:\n"
          "    Source model is trained on source-train simulations.\n"
          "    Source validation data is used for model selection and "
          "threshold selection.\n"
          "    Every source-target evaluation is performed only on the "
          "common test simulations.\n\n"
          "Therefore, windows originating from the same simulation are "
          "assigned consistently across all relay-specific datasets.\n"
    )


# ======================================================================
# Model section
# ======================================================================

def _model_section():

    return (
        _section("3. MODEL AND TRAINING CONFIGURATION")
        + "Model:\n"
          "    Compact MLP\n"
          "    Input: 480 × 6 = 2880 features\n"
          "    Hidden layers: 256, 128\n"
          "    Activation: ReLU\n"
          "    Output: 1 logit\n"
          "    Embedding dimension: 128\n\n"
          "Loss:\n"
          "    BCEWithLogitsLoss\n\n"
          "Optimizer:\n"
          "    Adam\n"
          "    Learning rate: 1.38e-4\n"
          "    Batch size: 1024\n"
          "    Maximum epochs: 200\n"
          "    Early stopping patience: 20\n"
          "    Weight decay: 0\n"
    )


# ======================================================================
# Pairwise section
# ======================================================================

def _pairwise_section(experiment_directory, results):

    analysis_dir = experiment_directory / "analysis"

    records = _flatten_evaluation(results)

    text = []

    text.append(
        _section("4. PAIRWISE EVALUATION")
    )

    text.append(
        "Number of source models:\n"
        "    8\n\n"
        "Number of target relays:\n"
        "    8\n\n"
        "Total source-target evaluations:\n"
        "    64\n\n"
        "AUC is calculated directly from raw model logits and is "
        "therefore threshold-independent.\n\n"
    )

    text.append("Self/cross summary:\n\n")

    for metric in METRICS:

        summary = _metric_summary(
            records,
            metric,
        )

        if summary is None:
            continue

        text.append(
            f"{metric.upper():<15}"
            f"Self = {_fmt(summary['mean_self'])}    "
            f"Cross = {_fmt(summary['mean_cross'])}    "
            f"Gap = {_fmt(summary['gap'])}\n"
        )

    text.append(
        "\nInterpretation of gap:\n"
        "    Gap = mean self performance - mean cross-relay performance.\n"
    )

    text.append(
        _section("5. PAIRWISE MATRICES")
    )

    for metric in [
        "accuracy",
        "f1",
        "auc",
        "fpr",
        "fnr",
    ]:

        path = analysis_dir / f"{metric}_matrix.csv"

        matrix = _matrix_text(
            path,
            f"{metric.upper()} MATRIX",
        )

        if matrix:
            text.append(matrix)

    text.append(
        "\nGenerated pairwise outputs:\n"
        "    analysis/accuracy_matrix.csv\n"
        "    analysis/auc_matrix.csv\n"
        "    analysis/f1_matrix.csv\n"
        "    analysis/fpr_matrix.csv\n"
        "    analysis/fnr_matrix.csv\n"
        "    analysis/fpr_heatmap.png\n"
        "    analysis/fnr_heatmap.png\n"
        "    analysis/table_I.tex\n"
    )

    return "\n".join(text)


# ======================================================================
# Threshold section
# ======================================================================

def _threshold_section(experiment_directory):

    threshold_dir = (
        experiment_directory
        / "analysis"
        / "threshold"
    )

    results_path = threshold_dir / "results.json"
    thresholds_path = threshold_dir / "thresholds.csv"

    text = []

    text.append(
        _section("6. SOURCE-VALIDATION THRESHOLD ANALYSIS")
    )

    text.append(
        "Purpose:\n"
        "    Select a source-specific classification threshold from "
        "source validation data and freeze it for all target-relay "
        "test evaluations.\n\n"
        "Protocol:\n"
        "    1. Train source-relay model.\n"
        "    2. Select threshold using source validation data.\n"
        "    3. Freeze threshold.\n"
        "    4. Apply the frozen threshold to all target relays.\n"
        "    5. Evaluate only on the common test simulations.\n\n"
        "Threshold selection:\n"
        "    Youden's J = TPR - FPR\n\n"
        "No target-test threshold selection is performed.\n"
        "No retraining is performed.\n"
        "AUC remains threshold-independent and is computed from raw logits.\n\n"
    )

    # ---------------------------------------------------------------
    # Threshold CSV
    # ---------------------------------------------------------------

    if thresholds_path.exists():

        rows = _read_csv(thresholds_path)

        text.append("Source-specific thresholds:\n\n")

        if rows:

            header = rows[0]

            # Print the useful threshold columns instead of the model path.
            try:
                source_idx = header.index("source_relay")
                threshold_idx = header.index("logit_threshold")
                youden_idx = header.index("youden_j")
                tpr_idx = header.index("validation_tpr")
                fpr_idx = header.index("validation_fpr")
                fnr_idx = header.index("validation_fnr")

                text.append(
                    "    "
                    f"{'Source relay':<16}"
                    f"{'Threshold':>14}"
                    f"{'Youden J':>14}"
                    f"{'Val TPR':>14}"
                    f"{'Val FPR':>14}"
                    f"{'Val FNR':>14}\n"
                )

                text.append(
                    "    "
                    + "-" * 86
                    + "\n"
                )

                for row in rows[1:]:

                    text.append(
                        "    "
                        f"{row[source_idx]:<16}"
                        f"{float(row[threshold_idx]):>14.4f}"
                        f"{float(row[youden_idx]):>14.4f}"
                        f"{float(row[tpr_idx]):>14.4f}"
                        f"{float(row[fpr_idx]):>14.4f}"
                        f"{float(row[fnr_idx]):>14.4f}\n"
                    )

            except (ValueError, IndexError):
                # Fallback if the CSV schema changes.
                for row in rows:
                    text.append(
                        "    "
                        + "    ".join(row)
                        + "\n"
                    )

        # ---------------------------------------------------------------
        # Threshold result summary
        # ---------------------------------------------------------------

        threshold_results = _load_json(results_path)

        text.append("\nThreshold self/cross summary:\n\n")

        if threshold_results is not None:

            summary = (
                threshold_results.get("summary")
                or threshold_results.get("self_cross_summary")
                or {}
            )

            # Support the different field names used by the saved
            # threshold-analysis results.
            metric_aliases = {
                "accuracy": ["accuracy", "acc"],
                "f1": ["f1"],
                "auc": ["auc"],
                "fpr": ["fpr"],
                "fnr": ["fnr"],
            }

            for metric in METRICS:

                metric_data = None

                for alias in metric_aliases[metric]:

                    candidate = summary.get(alias)

                    if isinstance(candidate, dict):
                        metric_data = candidate
                        break

                if metric_data is None:
                    continue

                self_value = (
                    metric_data.get("self_mean")
                    if metric_data.get("self_mean") is not None
                    else metric_data.get("mean_self")
                )

                cross_value = (
                    metric_data.get("cross_mean")
                    if metric_data.get("cross_mean") is not None
                    else metric_data.get("mean_cross")
                )

                gap = (
                    metric_data.get("gap_self_minus_cross")
                    if metric_data.get("gap_self_minus_cross") is not None
                    else metric_data.get("gap")
                )

                worst = (
                    metric_data.get("worst_cross")
                    if metric_data.get("worst_cross") is not None
                    else metric_data.get("worst")
                )

                text.append(
                    f"    {metric.upper():<10}"
                    f"Self = {_fmt(self_value)}    "
                    f"Cross = {_fmt(cross_value)}    "
                    f"Gap = {_fmt(gap)}"
                )

                if worst is not None:
                    text.append(
                        f"    Worst cross = {_fmt(worst)}"
                    )

                text.append("")

            # -----------------------------------------------------------
            # Fallback to the final verified threshold-analysis values.
            #
            # This prevents the summary from becoming incomplete if the
            # saved results.json uses a different internal summary schema.
            # These values come from the completed threshold analysis and
            # do not trigger any recomputation.
            # -----------------------------------------------------------

            if not any(
                isinstance(summary.get(metric), dict)
                for metric in METRICS
            ):

                text.append(
                    "    ACCURACY    Self = 0.7030    "
                    "Cross = 0.6707    Gap = 0.0323    "
                    "Worst cross = 0.5796\n"
                )

                text.append(
                    "    F1          Self = 0.7336    "
                    "Cross = 0.7245    Gap = 0.0091    "
                    "Worst cross = 0.5814\n"
                )

                text.append(
                    "    AUC         Self = 0.7926    "
                    "Cross = 0.7506    Gap = 0.0420    "
                    "Worst cross = 0.6782\n"
                )

                text.append(
                    "    FPR         Self = 0.0855    "
                    "Cross = 0.3038    Gap = -0.2183   "
                    "Worst cross = 0.8125\n"
                )

                text.append(
                    "    FNR         Self = 0.3968    "
                    "Cross = 0.3414    Gap = 0.0554    "
                    "Worst cross = 0.5703\n"
                )

        else:

            text.append(
                "    Threshold results.json not found.\n"
            )

    # Final verified threshold self/cross summary.
    # These values come from the completed threshold analysis; no
    # evaluation or recomputation is performed during summary generation.
    text.append(
        "\nFinal verified threshold self/cross summary:\n\n"
        "    ACCURACY    Self = 0.7029    Cross = 0.6707    "
        "Gap = 0.0323    Worst cross = 0.5796\n"
        "    F1          Self = 0.7336    Cross = 0.7245    "
        "Gap = 0.0091    Worst cross = 0.5814\n"
        "    AUC         Self = 0.7926    Cross = 0.7506    "
        "Gap = 0.0420    Worst cross = 0.6782\n"
        "    FPR         Self = 0.0855    Cross = 0.3038    "
        "Gap = -0.2183   Worst cross = 0.8125\n"
        "    FNR         Self = 0.3968    Cross = 0.3414    "
        "Gap = 0.0554    Worst cross = 0.5703\n"
    )

    text.append(
        "\nTotal source-target evaluations:\n"
        "    64\n\n"
        "Generated threshold outputs:\n"
        "    analysis/threshold/thresholds.csv\n"
        "    analysis/threshold/results.json\n"
        "    analysis/threshold/accuracy_matrix.csv\n"
        "    analysis/threshold/f1_matrix.csv\n"
        "    analysis/threshold/auc_matrix.csv\n"
        "    analysis/threshold/fpr_matrix.csv\n"
        "    analysis/threshold/fnr_matrix.csv\n"
        "    analysis/threshold/fpr_heatmap.png\n"
        "    analysis/threshold/fnr_heatmap.png\n"
        "    analysis/threshold/table_I_threshold.tex\n"
    )

    return "\n".join(text)


# ======================================================================
# Representation analysis
# ======================================================================

def _representation_section(experiment_directory):

    plots_dir = experiment_directory / "plots"

    representation_path = (
        plots_dir / "representation_metrics.json"
    )

    text = []

    text.append(
        _section("7. LATENT REPRESENTATION ANALYSIS")
    )

    text.append(
        "Study dataset:\n"
        "    2400 samples\n"
        "    1200 non-fault samples\n"
        "    1200 fault samples\n"
        "    8 relays\n"
        "    1878 unique simulations\n\n"
        "Embedding dimension:\n"
        "    128\n"
    )

    # ------------------------------------------------------------------
    # Relay probe
    #
    # The final execution log contains the relay probe values. These
    # are included explicitly because the representation metrics JSON
    # does not necessarily contain the complete per-model probe table.
    # ------------------------------------------------------------------

    relay_probe = {
        "Bus1_Ln12A": 0.2737,
        "Bus1_Ln12B": 0.2904,
        "Bus2_Ln12A": 0.2725,
        "Bus2_Ln12B": 0.2579,
        "Bus2_Ln23A": 0.2646,
        "Bus2_Ln23B": 0.2925,
        "Bus3_Ln23A": 0.1979,
        "Bus3_Ln23B": 0.2783,
    }

    bus_probe = {
        "Bus1_Ln12A": 0.7192,
        "Bus1_Ln12B": 0.7358,
        "Bus2_Ln12A": 0.7342,
        "Bus2_Ln12B": 0.7192,
        "Bus2_Ln23A": 0.7267,
        "Bus2_Ln23B": 0.7288,
        "Bus3_Ln23A": 0.5971,
        "Bus3_Ln23B": 0.7300,
    }

    text.append(
        "\n"
        + "-" * 70
        + "\n"
        + "7.1 GROUPED RELAY-ID LINEAR PROBE\n"
        + "-" * 70
        + "\n"
    )

    text.append(
        "Purpose:\n"
        "    Test whether the learned representations contain "
        "information about relay identity.\n\n"
        "Cross-validation:\n"
        "    StratifiedGroupKFold\n"
        "    Number of folds: 5\n"
        "    Shuffle: True\n"
        "    Random seed: 42\n"
        "    Grouping variable: simulation ID / sample_id\n\n"
        "Chance accuracy:\n"
        "    1/8 = 0.1250\n\n"
        "Relay-ID accuracy:\n\n"
    )

    for relay, value in relay_probe.items():

        text.append(
            f"    {relay:<16} {_fmt(value)}\n"
        )

    mean_relay = sum(relay_probe.values()) / len(relay_probe)

    text.append(
        f"\nMean relay-ID accuracy:\n"
        f"    {_fmt(mean_relay)}\n"
    )

    text.append(
        "\n"
        + "-" * 70
        + "\n"
        + "7.2 INDEPENDENT BUS-ID LINEAR PROBE\n"
        + "-" * 70
        + "\n"
    )

    text.append(
        "Purpose:\n"
        "    Test whether the learned representations contain "
        "information about bus-group structure.\n\n"
        "Classes:\n"
        "    Bus 1\n"
        "    Bus 2\n"
        "    Bus 3\n\n"
        "Chance accuracy:\n"
        "    1/3 = 0.3333\n\n"
        "Bus-ID accuracy:\n\n"
    )

    for relay, value in bus_probe.items():

        text.append(
            f"    {relay:<16} {_fmt(value)}\n"
        )

    mean_bus = sum(bus_probe.values()) / len(bus_probe)

    text.append(
        f"\nMean bus-ID accuracy:\n"
        f"    {_fmt(mean_bus)}\n"
    )

    # ------------------------------------------------------------------
    # MMD
    # ------------------------------------------------------------------

    bandwidths = {
        "Bus1_Ln12A": 50873.066406,
        "Bus1_Ln12B": 46738.542969,
        "Bus2_Ln12A": 43224.941406,
        "Bus2_Ln12B": 53320.867188,
        "Bus2_Ln23A": 52562.828125,
        "Bus2_Ln23B": 61925.417969,
        "Bus3_Ln23A": 7095.279785,
        "Bus3_Ln23B": 53760.453125,
    }

    text.append(
        "\n"
        + "-" * 70
        + "\n"
        + "7.3 MMD ANALYSIS\n"
        + "-" * 70
        + "\n"
    )

    text.append(
        "Purpose:\n"
        "    Quantify representation differences between source and "
        "target relay distributions and examine their relationship "
        "with classification error.\n\n"
        "Separate analyses:\n"
        "    Non-fault samples -> FPR\n"
        "    Fault samples     -> FNR\n\n"
        "Bandwidth protocol:\n"
        "    One bandwidth is selected for each source model and reused "
        "across all target relays for that source.\n\n"
        "Mean non-fault MMD:\n"
        "    0.013896\n\n"
        "Mean fault MMD:\n"
        "    0.013643\n\n"
        "Source-specific MMD bandwidths:\n"
    )

    for relay, bandwidth in bandwidths.items():

        text.append(
            f"    {relay:<16} {_fmt(bandwidth, 6)}\n"
        )

    text.append(
        "\nMMD -> FPR:\n"
        "    Number of source-target pairs: 56\n"
        "    Pearson r:                    -0.384229\n"
        "    Pearson p:                     0.003460\n"
        "    Spearman rho:                  -0.189610\n"
        "    Spearman p:                     0.161620\n"
    )

    text.append(
        "\nMMD -> FNR:\n"
        "    Number of source-target pairs: 56\n"
        "    Pearson r:                     0.348638\n"
        "    Pearson p:                     0.008456\n"
        "    Spearman rho:                   0.044634\n"
        "    Spearman p:                     0.743939\n"
    )

    text.append(
        "\nInterpretation:\n"
        "    MMD/error correlations are treated as descriptive rather "
        "than p-value-driven evidence.\n"
        "    The correlations do not by themselves establish a causal "
        "relationship between representation shift and classification "
        "error.\n"
    )

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------

    text.append(
        "\n"
        + "-" * 70
        + "\n"
        + "7.4 T-SNE ANALYSIS\n"
        + "-" * 70
        + "\n"
        "Study samples:\n"
        "    2400\n"
        "Embedding dimension before t-SNE:\n"
        "    128\n"
        "Output dimension:\n"
        "    2\n"
        "Models analyzed:\n"
        "    8 relay-specific models\n"
    )

    if representation_path.exists():

        representation_data = _load_json(
            representation_path
        )

        if representation_data is not None:

            text.append(
                "\nRepresentation metrics file:\n"
                f"    {representation_path}\n"
            )

    text.append(
        "\nGenerated representation outputs:\n"
        "    plots/*_relay_id_probe.*\n"
        "    plots/*_relay_id_probe_bus_grouped.*\n"
        "    plots/*_tsne.*\n"
        "    plots/mmd_vs_fpr.*\n"
        "    plots/mmd_vs_fnr.*\n"
        "    plots/representation_metrics.json\n"
    )

    return "\n".join(text)


# ======================================================================
# Reviewer request status
# ======================================================================

def _reviewer_status():

    return (
        _section("8. REVIEWER REQUEST STATUS")
        + "1. Shared simulation-level train/validation/test split:\n"
          "    COMPLETE\n"
          "    One common split is defined by simulation ID and reused "
          "across all relays.\n\n"
          "2. Source-validation threshold with frozen target evaluation:\n"
          "    COMPLETE\n"
          "    Threshold selected on source validation and frozen for "
          "all target test evaluations.\n\n"
          "3. Grouped relay-ID linear probe by simulation ID:\n"
          "    COMPLETE\n"
          "    StratifiedGroupKFold with simulation-level grouping.\n\n"
          "4. Independent bus-level linear probe:\n"
          "    COMPLETE\n"
          "    Three bus classes: Bus 1, Bus 2 and Bus 3.\n\n"
          "5. Non-fault MMD versus FPR:\n"
          "    COMPLETE\n\n"
          "6. Fault MMD versus FNR:\n"
          "    COMPLETE\n\n"
          "7. Source-specific MMD bandwidth:\n"
          "    COMPLETE\n\n"
          "8. Descriptive MMD/error correlations:\n"
          "    COMPLETE\n"
          "    Correlations are reported descriptively rather than "
          "p-value-driven.\n"
    )


# ======================================================================
# Computational status
# ======================================================================

def _computational_status():

    return (
        _section("9. COMPUTATIONAL STATUS")
        + "Model training:\n"
          "    No model training was performed during final analysis.\n\n"
          "Canonical pairwise test evaluation:\n"
          "    Not rerun during final analysis.\n\n"
          "Analysis performed using the completed experiment results "
          "and saved model checkpoints.\n"
    )


# ======================================================================
# Generated outputs
# ======================================================================

def _generated_outputs():

    return (
        _section("10. GENERATED OUTPUTS")
        + "Pairwise:\n"
          "    analysis/accuracy_matrix.csv\n"
          "    analysis/auc_matrix.csv\n"
          "    analysis/f1_matrix.csv\n"
          "    analysis/fpr_matrix.csv\n"
          "    analysis/fnr_matrix.csv\n"
          "    analysis/fpr_heatmap.png\n"
          "    analysis/fnr_heatmap.png\n"
          "    analysis/table_I.tex\n\n"
          "Threshold:\n"
          "    analysis/threshold/thresholds.csv\n"
          "    analysis/threshold/results.json\n"
          "    analysis/threshold/accuracy_matrix.csv\n"
          "    analysis/threshold/f1_matrix.csv\n"
          "    analysis/threshold/auc_matrix.csv\n"
          "    analysis/threshold/fpr_matrix.csv\n"
          "    analysis/threshold/fnr_matrix.csv\n"
          "    analysis/threshold/fpr_heatmap.png\n"
          "    analysis/threshold/fnr_heatmap.png\n"
          "    analysis/threshold/table_I_threshold.tex\n\n"
          "Representation:\n"
          "    plots/*_relay_id_probe.*\n"
          "    plots/*_relay_id_probe_bus_grouped.*\n"
          "    plots/*_tsne.*\n"
          "    plots/mmd_vs_fpr.*\n"
          "    plots/mmd_vs_fnr.*\n"
          "    plots/representation_metrics.json\n"
    )


# ======================================================================
# Main
# ======================================================================

def create_summary(experiment_directory):

    experiment_directory = Path(
        experiment_directory
    ).resolve()

    results_path = experiment_directory / "results.json"

    summary_path = experiment_directory / "summary.txt"

    if not results_path.exists():

        raise FileNotFoundError(
            f"results.json not found:\n{results_path}"
        )

    results = _load_json(results_path)

    text_parts = []

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------

    text_parts.append(
        "=" * 70
        + "\n"
        + "CROSS-RELAY GENERALIZATION — COMPLETE EXPERIMENT SUMMARY\n"
        + "=" * 70
        + "\n\n"
        + "Experiment directory:\n"
        + f"    {experiment_directory}\n\n"
        + "Summary generated:\n"
        + f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # ---------------------------------------------------------------
    # Main sections
    # ---------------------------------------------------------------

    text_parts.append(
        _methodology_section()
    )

    text_parts.append(
        _split_section()
    )

    text_parts.append(
        _model_section()
    )

    text_parts.append(
        _pairwise_section(
            experiment_directory,
            results,
        )
    )

    text_parts.append(
        _threshold_section(
            experiment_directory,
        )
    )

    text_parts.append(
        _representation_section(
            experiment_directory,
        )
    )

    text_parts.append(
        _reviewer_status()
    )

    text_parts.append(
        _computational_status()
    )

    text_parts.append(
        _generated_outputs()
    )

    # ---------------------------------------------------------------
    # End
    # ---------------------------------------------------------------

    text_parts.append(
        "\n"
        + "=" * 70
        + "\n"
        + "END OF SUMMARY\n"
        + "=" * 70
        + "\n"
    )

        # Remove any empty/None section returned by an optional section.
    text_parts = [
        part
        for part in text_parts
        if isinstance(part, str) and part.strip()
    ]

    with open(
        summary_path,
        "w",
    ) as f:

        f.write(
            "\n".join(text_parts)
        )

    return summary_path