import time

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data import DataLoader

from scripts.datasets import RelayDataset


class Trainer:

    def __init__(self, cfg):

        self.cfg = cfg

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def fit(
        self,
        model,
        relay_name,
        X_local,
        y_local,
        train_indices,
        validation_indices,
    ):

        train_dataset = RelayDataset(
            X=X_local,
            y=y_local["y_fault_present"].astype(np.float32).values,
            indices=train_indices,
        )

        validation_dataset = RelayDataset(
            X=X_local,
            y=y_local["y_fault_present"].astype(np.float32).values,
            indices=validation_indices,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            persistent_workers=self.cfg.training.persistent_workers,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            pin_memory=self.cfg.training.pin_memory,
            persistent_workers=self.cfg.training.persistent_workers,
        )

        model = model.to(self.device)

        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        history = {
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "validation_f1": [],
        }

        best_epoch = 0
        best_validation_loss = float("inf")

        start = time.time()

        for epoch in range(self.cfg.training.epochs):

            model.train()

            train_loss = 0.0

            for X_batch, y_batch in train_loader:

                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                logits = model(X_batch)

                loss = criterion(logits, y_batch)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()

            validation_loss = 0.0

            predictions = []
            labels = []

            with torch.no_grad():

                for X_batch, y_batch in validation_loader:

                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    logits = model(X_batch)

                    loss = criterion(logits, y_batch)

                    validation_loss += loss.item()

                    predictions.extend(
                        (
                            torch.sigmoid(logits)
                            >= 0.5
                        ).cpu().numpy()
                    )

                    labels.extend(
                        y_batch.cpu().numpy()
                    )

            validation_loss /= len(validation_loader)

            validation_accuracy = accuracy_score(
                labels,
                predictions,
            )

            validation_f1 = f1_score(
                labels,
                predictions,
                zero_division=0,
            )

            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)
            history["validation_f1"].append(validation_f1)

            print(
                f"[{relay_name}] "
                f"Epoch {epoch+1:03d} | "
                f"Train {train_loss:.4f} | "
                f"Val {validation_loss:.4f} | "
                f"Acc {validation_accuracy:.4f} | "
                f"F1 {validation_f1:.4f}"
            )

            if validation_loss < best_validation_loss:

                best_validation_loss = validation_loss
                best_epoch = epoch + 1

            elif (
                epoch + 1 - best_epoch
                >= self.cfg.training.patience
            ):

                print(
                    f"Early stopping at epoch {epoch+1}"
                )

                break

        training_time = time.time() - start

        return {

            "model": model,

            "relay": relay_name,

            "best_epoch": best_epoch,

            "best_validation_loss": best_validation_loss,

            "validation_accuracy":
                history["validation_accuracy"][best_epoch - 1],

            "validation_f1":
                history["validation_f1"][best_epoch - 1],

            "training_time":
                training_time,

            "history": history,
        }