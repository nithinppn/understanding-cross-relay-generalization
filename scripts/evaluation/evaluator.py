import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader

from scripts.datasets import RelayDataset


class Evaluator:

    def __init__(self, cfg):

        self.cfg = cfg

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def evaluate(
        self,
        model,
        X_local,
        y_local,
        indices,
    ):

        dataset = RelayDataset(
            X=X_local,
            y=y_local["y_fault_present"].astype(np.float32).values,
            indices=indices,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            persistent_workers=self.cfg.training.persistent_workers,
        )

        model = model.to(self.device)
        model.eval()

        logits_list = []
        predictions = []
        labels = []

        with torch.no_grad():

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(
                    self.device,
                    non_blocking=True,
                )

                logits = model(X_batch)

                # Keep raw logits for numerically stable ROC AUC.
                logits_list.extend(
                    logits.cpu().numpy()
                )

                # Fixed 0.5 probability threshold is equivalent
                # to a zero-logit threshold.
                preds = (logits >= 0).int()

                predictions.extend(
                    preds.cpu().numpy()
                )

                labels.extend(
                    y_batch.numpy()
                )

        return self._calculate_metrics(
            labels=labels,
            logits=logits_list,
            predictions=predictions,
        )

    @staticmethod
    def _calculate_metrics(
        labels,
        logits,
        predictions,
    ):

        logits = np.asarray(
            logits
        ).reshape(-1)

        predictions = np.asarray(
            predictions
        ).reshape(-1)

        labels = np.asarray(
            labels
        ).reshape(-1)

        # ============================================================
        # Confusion matrix
        # ============================================================

        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()

        # ============================================================
        # AUC
        # ============================================================

        if len(np.unique(labels)) > 1:

            auc = roc_auc_score(
                labels,
                logits,
            )

        else:

            auc = 0.0

        # ============================================================
        # False Positive Rate
        #
        # FPR = FP / (FP + TN)
        # ============================================================

        if (fp + tn) > 0:

            fpr = fp / (fp + tn)

        else:

            fpr = 0.0

        # ============================================================
        # False Negative Rate
        #
        # FNR = FN / (FN + TP)
        # ============================================================

        if (fn + tp) > 0:

            fnr = fn / (fn + tp)

        else:

            fnr = 0.0

        # ============================================================
        # Return metrics
        # ============================================================

        return {

            "accuracy": accuracy_score(
                labels,
                predictions,
            ),

            "f1": f1_score(
                labels,
                predictions,
                zero_division=0,
            ),

            "auc": auc,

            "fpr": fpr,

            "fnr": fnr,

            "tn": int(tn),

            "fp": int(fp),

            "fn": int(fn),

            "tp": int(tp),
        }