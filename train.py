from __future__ import annotations
import os
import argparse
from create_splits_seq import create_splits
from dataset_modules.dataset_generic import (
    Generic_MIL_Dataset,
)
from utils.train_utils import MILTrainingConfig, TrainingResults, run_training


def setup_dataset(config: MILTrainingConfig) -> Generic_MIL_Dataset:
    """Setup dataset based on configuration"""
    print(f"\nLoad Dataset: {config.data_set_name}")

    if not config.data_root_dir or not config.data_set_name:
        raise ValueError("data_root_dir and data_set_name must be provided")

    data_dir = os.path.join(config.data_root_dir, config.data_set_name)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    if config.task == "task_1_tumor_vs_normal":
        create_splits(
            task="task_1_tumor_vs_normal",
            label_frac=config.label_frac,
            seed=config.seed,
            k=config.k,
            val_frac=0.15,
            test_frac=0.15,
        )
        dataset = Generic_MIL_Dataset(
            csv_path="dataset_csv/tumor_vs_normal_dummy_clean.csv",
            data_dir=data_dir,
            shuffle=False,
            seed=config.seed,
            print_info=True,
            label_dict={"normal_tissue": 0, "tumor_tissue": 1},
            patient_strat=False,
            ignore=[],
        )

    elif config.task == "task_2_tumor_subtyping":
        create_splits(
            task="task_2_tumor_subtyping",
            label_frac=config.label_frac,
            seed=config.seed,
            k=config.k,
            val_frac=0.15,
            test_frac=0.15,
        )
        dataset = Generic_MIL_Dataset(
            csv_path="dataset_csv/tumor_subtyping_dummy_clean.csv",
            data_dir=data_dir,
            shuffle=False,
            seed=config.seed,
            print_info=True,
            label_dict={"subtype_1": 0, "subtype_2": 1, "subtype_3": 2},
            patient_strat=False,
            ignore=[],
        )

        if config.model_type in ["clam_sb", "clam_mb"]:
            assert (
                config.subtyping
            ), "Subtyping must be enabled for CLAM models with tumor subtyping task"
    else:
        raise NotImplementedError(f"Task {config.task} not implemented")

    return dataset


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with type hints for all arguments"""
    parser = argparse.ArgumentParser(description="Configurations for WSI Training")

    # Data parameters
    parser.add_argument(
        "--data_root_dir", type=str, default=None, help="data directory"
    )
    parser.add_argument("--data_set_name", type=str, default=None, help="dataset name")
    parser.add_argument("--embed_dim", type=int, default=1024)

    # Training parameters
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="maximum number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--label_frac",
        type=float,
        default=1.0,
        help="fraction of training labels (default: 1.0)",
    )
    parser.add_argument(
        "--reg", type=float, default=1e-5, help="weight decay (default: 1e-5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed for reproducible experiment (default: 1)",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="number of folds (default: 10)"
    )
    parser.add_argument(
        "--k_start", type=int, default=-1, help="start fold (default: -1, last fold)"
    )
    parser.add_argument(
        "--k_end", type=int, default=-1, help="end fold (default: -1, first fold)"
    )

    # Results and logging
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="results directory (default: ./results)",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="manually specify the set of splits to use, "
        + "instead of infering from the task and label_frac argument (default: None)",
    )
    parser.add_argument(
        "--log_data",
        action="store_true",
        default=False,
        help="log data using tensorboard",
    )
    parser.add_argument(
        "--testing", action="store_true", default=False, help="debugging tool"
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="enable early stopping",
    )

    # Model parameters
    parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
    parser.add_argument("--drop_out", type=float, default=0.25, help="dropout")
    parser.add_argument(
        "--bag_loss",
        type=str,
        choices=["svm", "ce"],
        default="ce",
        help="slide-level classification loss function (default: ce)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["clam_sb", "clam_mb", "mil"],
        default="clam_sb",
        help="type of model (default: clam_sb, clam w/ single attention branch)",
    )
    parser.add_argument(
        "--exp_code", type=str, help="experiment code for saving results"
    )
    parser.add_argument(
        "--weighted_sample",
        action="store_true",
        default=False,
        help="enable weighted sampling",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "big"],
        default="small",
        help="size of model, does not affect mil",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["task_1_tumor_vs_normal", "task_2_tumor_subtyping"],
        help="task to perform",
    )

    # CLAM specific options
    parser.add_argument(
        "--no_inst_cluster",
        action="store_true",
        default=False,
        help="disable instance-level clustering",
    )
    parser.add_argument(
        "--inst_loss",
        type=str,
        choices=["svm", "ce", None],
        default=None,
        help="instance-level clustering loss function (default: None)",
    )
    parser.add_argument(
        "--subtyping", action="store_true", default=False, help="subtyping problem"
    )
    parser.add_argument(
        "--bag_weight",
        type=float,
        default=0.7,
        help="clam: weight coefficient for bag-level loss (default: 0.7)",
    )
    parser.add_argument(
        "--B",
        type=int,
        default=8,
        help="numbr of positive/negative patches to sample for clam",
    )

    # MLflow model registration options
    parser.add_argument(
        "--registered_model_name",
        type=str,
        default=None,
        help="Name for registering the model in MLflow Model Registry",
    )
    parser.add_argument(
        "--no_register_model",
        action="store_true",
        default=False,
        help="Disable model registration in MLflow",
    )

    return parser


def main() -> TrainingResults:
    """Main function for command line execution"""
    parser = create_parser()
    args = parser.parse_args()

    # Convert args to dictionary and create config
    config_dict = vars(args)
    config_dict["register_best_model"] = not args.no_register_model

    config = MILTrainingConfig(**config_dict)
    dataset = setup_dataset(config)
    # Run training
    results = run_training(config, dataset)
    print("Finished!")
    print("End script")

    return results


if __name__ == "__main__":
    main()
