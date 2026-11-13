import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from scripts.utils.config import SENSOR_MAP


class Pairwise:

    def prepare_datasets(
        self,
        y_local,
        seed: int,
    ):
        # ============================================================
        # 1. Remove switching events globally
        # ============================================================

        valid_mask = (
            y_local["event_family"] != "switching"
        )

        valid_indices = np.where(valid_mask)[0]

        y_valid = y_local.iloc[valid_indices]

        # ============================================================
        # 2. Create ONE global simulation-level split
        # ============================================================

        simulation_ids = (
            y_valid["sample_id"]
            .drop_duplicates()
            .values
        )

        # ------------------------------------------------------------
        # First split:
        #   85% train+validation
        #   15% test
        # ------------------------------------------------------------

        gss_test = GroupShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=seed,
        )

        dummy_X = np.zeros(len(y_valid))
        dummy_y = np.zeros(len(y_valid))

        train_val_idx, test_idx = next(
            gss_test.split(
                dummy_X,
                dummy_y,
                groups=y_valid["sample_id"].values,
            )
        )

        train_val_simulation_ids = (
            y_valid.iloc[train_val_idx]["sample_id"]
            .drop_duplicates()
            .values
        )

        test_simulation_ids = (
            y_valid.iloc[test_idx]["sample_id"]
            .drop_duplicates()
            .values
        )

        # ------------------------------------------------------------
        # Second split:
        #   70% train
        #   15% validation
        #
        # 0.1764706 of the remaining 85% ≈ 15% overall
        # ------------------------------------------------------------

        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=0.17647058823529413,
            random_state=seed,
        )

        train_local_idx, val_local_idx = next(
            gss_val.split(
                np.zeros(len(train_val_simulation_ids)),
                np.zeros(len(train_val_simulation_ids)),
                groups=train_val_simulation_ids,
            )
        )

        train_simulation_ids = (
            train_val_simulation_ids[train_local_idx]
        )

        validation_simulation_ids = (
            train_val_simulation_ids[val_local_idx]
        )

        # ============================================================
        # 3. Sanity check global split
        # ============================================================

        train_set = set(train_simulation_ids)
        validation_set = set(validation_simulation_ids)
        test_set = set(test_simulation_ids)

        assert train_set.isdisjoint(validation_set)
        assert train_set.isdisjoint(test_set)
        assert validation_set.isdisjoint(test_set)

        assert (
            len(train_set)
            + len(validation_set)
            + len(test_set)
            == len(simulation_ids)
        )

        print(
            "\nGLOBAL SIMULATION SPLIT"
        )

        print(
            f"Train simulations: "
            f"{len(train_set)}"
        )

        print(
            f"Validation simulations: "
            f"{len(validation_set)}"
        )

        print(
            f"Test simulations: "
            f"{len(test_set)}"
        )

        # ============================================================
        # 4. Apply the SAME split to every relay
        # ============================================================

        relay_datasets = {}

        for helper_name, friendly_name in SENSOR_MAP.items():

            # --------------------------------------------------------
            # Relay-specific samples
            # --------------------------------------------------------

            relay_mask = (
                (y_local["sensor_name"] == helper_name)
                &
                (y_local["event_family"] != "switching")
            )

            relay_indices = np.where(
                relay_mask
            )[0]

            relay_sample_ids = (
                y_local.iloc[relay_indices]["sample_id"]
                .values
            )

            # --------------------------------------------------------
            # Apply GLOBAL simulation split
            # --------------------------------------------------------

            train_mask = np.isin(
                relay_sample_ids,
                train_simulation_ids,
            )

            validation_mask = np.isin(
                relay_sample_ids,
                validation_simulation_ids,
            )

            test_mask = np.isin(
                relay_sample_ids,
                test_simulation_ids,
            )

            train_indices = relay_indices[
                train_mask
            ]

            validation_indices = relay_indices[
                validation_mask
            ]

            test_indices = relay_indices[
                test_mask
            ]

            # --------------------------------------------------------
            # Sanity checks
            # --------------------------------------------------------

            assert len(
                set(
                    relay_sample_ids[train_mask]
                )
                &
                set(
                    relay_sample_ids[validation_mask]
                )
            ) == 0

            assert len(
                set(
                    relay_sample_ids[train_mask]
                )
                &
                set(
                    relay_sample_ids[test_mask]
                )
            ) == 0

            assert len(
                set(
                    relay_sample_ids[validation_mask]
                )
                &
                set(
                    relay_sample_ids[test_mask]
                )
            ) == 0

            # --------------------------------------------------------
            # Store relay datasets
            # --------------------------------------------------------

            relay_datasets[friendly_name] = {

                "train_indices":
                    train_indices,

                "validation_indices":
                    validation_indices,

                "test_indices":
                    test_indices,

                "all_indices":
                    relay_indices,

            }

            print(
                f"{friendly_name:20s} | "
                f"train={len(train_indices):8d} | "
                f"val={len(validation_indices):8d} | "
                f"test={len(test_indices):8d}"
            )

        return relay_datasets