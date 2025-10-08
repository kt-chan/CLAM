from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import pdb
import torch
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader, sampler

# internal imports
from create_splits_seq import create_splits
from dataset_modules.dataset_generic import (
    Generic_MIL_Dataset,
    Generic_WSI_Classification_Dataset,
)
from models.model_clam import CLAM_MB, CLAM_SB
from utils.file_utils import load_pkl, save_pkl
from utils.train_utils import train
from utils.utils import seed_torch


class TrainingResults(TypedDict):
    """Type definition for training results dictionary"""
    test_auc: List[float]
    val_auc: List[float]
    test_acc: List[float]
    val_acc: List[float]
    final_df: pd.DataFrame
    best_model: Optional[torch.nn.Module]
    mlflow_run_id: str


class ModelSettings(TypedDict):
    """Type definition for model settings dictionary"""
    num_splits: int
    k_start: int
    k_end: int
    task: str
    max_epochs: int
    results_dir: str
    lr: float
    experiment: str
    reg: float
    label_frac: float
    bag_loss: str
    seed: int
    model_type: str
    model_size: str
    use_drop_out: float
    weighted_sample: bool
    opt: str
    split_dir: str
    registered_model_name: Optional[str]
    register_best_model: bool
    bag_weight: Optional[float]
    inst_loss: Optional[str]
    B: Optional[int]


class MILTrainingConfig:
    """Configuration class for MIL training with strong typing"""

    def __init__(self, **kwargs: Any) -> None:
        # Data parameters
        self.data_root_dir: Optional[str] = kwargs.get("data_root_dir")
        self.data_set_name: Optional[str] = kwargs.get("data_set_name")
        self.embed_dim: int = kwargs.get("embed_dim", 1024)

        # Training parameters
        self.max_epochs: int = kwargs.get("max_epochs", 200)
        self.lr: float = kwargs.get("lr", 1e-4)
        self.label_frac: float = kwargs.get("label_frac", 1.0)
        self.reg: float = kwargs.get("reg", 1e-5)
        self.seed: int = kwargs.get("seed", 1)
        self.opt: str = kwargs.get("opt", "adam")
        self.drop_out: float = kwargs.get("drop_out", 0.25)
        self.bag_loss: str = kwargs.get("bag_loss", "ce")

        # Cross-validation parameters
        self.k: int = kwargs.get("k", 10)
        self.k_start: int = kwargs.get("k_start", -1)
        self.k_end: int = kwargs.get("k_end", -1)

        # Model parameters
        self.model_type: str = kwargs.get("model_type", "clam_sb")
        self.model_size: str = kwargs.get("model_size", "small")
        self.task: Optional[str] = kwargs.get("task")

        # CLAM specific parameters
        self.no_inst_cluster: bool = kwargs.get("no_inst_cluster", False)
        self.inst_loss: Optional[str] = kwargs.get("inst_loss")
        self.subtyping: bool = kwargs.get("subtyping", False)
        self.bag_weight: float = kwargs.get("bag_weight", 0.7)
        self.B: int = kwargs.get("B", 8)

        # Other parameters
        self.results_dir: str = kwargs.get("results_dir", "./results")
        self.split_dir: Optional[str] = kwargs.get("split_dir")
        self.log_data: bool = kwargs.get("log_data", False)
        self.testing: bool = kwargs.get("testing", False)
        self.early_stopping: bool = kwargs.get("early_stopping", False)
        self.weighted_sample: bool = kwargs.get("weighted_sample", False)
        self.exp_code: Optional[str] = kwargs.get("exp_code")

        # MLflow model registration parameters
        self.registered_model_name: Optional[str] = kwargs.get("registered_model_name")
        self.register_best_model: bool = kwargs.get("register_best_model", True)

        # Derived attributes (set by _setup_derived_attributes)
        self.start_fold: int = 0
        self.end_fold: int = self.k
        self.n_classes: Optional[int] = None

        self._setup_derived_attributes()

    def _setup_derived_attributes(self) -> None:
        """Setup attributes that depend on other parameters"""
        # Set fold ranges
        self.start_fold = 0 if self.k_start == -1 else self.k_start
        self.end_fold = self.k if self.k_end == -1 else self.k_end

        # Set n_classes based on task
        if self.task == "task_1_tumor_vs_normal":
            self.n_classes = 2
        elif self.task == "task_2_tumor_subtyping":
            self.n_classes = 3
        else:
            self.n_classes = None

        # Setup results directory
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_dir = os.path.join(
            self.results_dir, f"{self.exp_code}"
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # Setup split directory
        if self.split_dir is None:
            self.split_dir = os.path.join(
                "splits", f"{self.task}_{int(self.label_frac * 100)}"
            )
        else:
            self.split_dir = os.path.join("splits", self.split_dir)

        # Setup default registered model name
        if self.registered_model_name is None and self.exp_code:
            self.registered_model_name = f"{self.model_type}_{self.exp_code}"

    def get_settings(self) -> ModelSettings:
        """Return settings dictionary for logging"""
        settings: ModelSettings = {
            "num_splits": self.k,
            "k_start": self.k_start,
            "k_end": self.k_end,
            "task": self.task or "",
            "max_epochs": self.max_epochs,
            "results_dir": self.results_dir,
            "lr": self.lr,
            "experiment": self.exp_code or "",
            "reg": self.reg,
            "label_frac": self.label_frac,
            "bag_loss": self.bag_loss,
            "seed": self.seed,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "use_drop_out": self.drop_out,
            "weighted_sample": self.weighted_sample,
            "opt": self.opt,
            "split_dir": self.split_dir,
            "registered_model_name": self.registered_model_name,
            "register_best_model": self.register_best_model,
            "bag_weight": None,
            "inst_loss": None,
            "B": None,
        }

        if self.model_type in ["clam_sb", "clam_mb"]:
            settings.update({
                "bag_weight": self.bag_weight,
                "inst_loss": self.inst_loss,
                "B": self.B,
            })

        return settings


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
            assert config.subtyping, "Subtyping must be enabled for CLAM models with tumor subtyping task"
    else:
        raise NotImplementedError(f"Task {config.task} not implemented")

    return dataset


def reconstruct_clam_model(config: MILTrainingConfig, model_path: str) -> torch.nn.Module:
    """
    Reconstructs a CLAM model from a configuration and a saved state_dict.
    """
    if config.n_classes is None:
        raise ValueError("n_classes must be set in config before reconstructing model")

    # Create model architecture using the config
    if config.model_type == "clam_sb":
        model = CLAM_SB(
            size_arg=config.model_size,
            dropout=config.drop_out,
            n_classes=config.n_classes,
            subtyping=config.subtyping,
            embed_dim=config.embed_dim,
        )
    elif config.model_type == "clam_mb":
        model = CLAM_MB(
            size_arg=config.model_size,
            dropout=config.drop_out,
            n_classes=config.n_classes,
            subtyping=config.subtyping,
            embed_dim=config.embed_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Load the saved state_dict
    state_dict = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def log_best_model_to_mlflow(
    config: MILTrainingConfig,
    all_val_auc: List[float],
    all_test_auc: List[float],
    folds: np.ndarray,
    dataset: Generic_MIL_Dataset,
) -> Optional[torch.nn.Module]:
    """
    Identify the best model from cross-validation and log it to MLflow
    using a manually inferred signature.
    """
    if not config.register_best_model:
        print("Model registration is disabled. Skipping...")
        return None

    # Find the best fold based on validation AUC
    best_fold_idx = np.argmax(all_val_auc)
    best_fold = folds[best_fold_idx]
    best_val_auc = all_val_auc[best_fold_idx]
    best_test_auc = all_test_auc[best_fold_idx]

    print(f"Best model from fold {best_fold} with validation AUC: {best_val_auc:.4f}")

    # Load the best model results
    best_model_checkpoint_path = os.path.join(
        config.results_dir, f"s_{best_fold}_checkpoint.pt"
    )

    if not os.path.exists(best_model_checkpoint_path):
        print(f"Warning: Best model checkpoint not found at {best_model_checkpoint_path}")
        return None

    try:
        # Define explicit requirements
        pip_reqs = [
            "timm==0.9.8",
            "torch==2.8.0+cu128",
            "torchvision==0.23.0+cu128",
            "torchaudio==2.8.0+cu128",
            f"mlflow=={mlflow.__version__}",
            "numpy",
            "pandas",
        ]

        # Create dummy input for signature inference
        L = 500  # number of instances (patches)
        D = config.embed_dim
        dummy_input_np = np.random.randn(L, D).astype(np.float32)
        dummy_input_torch = torch.from_numpy(dummy_input_np)

        # Reconstruct the model architecture and load weights
        best_model = reconstruct_clam_model(config, best_model_checkpoint_path)

        # Generate signature by passing dummy input through the model
        with torch.no_grad():
            output = best_model(dummy_input_torch)
            Y_prob = output[0] if isinstance(output, tuple) else output

        Y_prob_np = Y_prob.cpu().numpy()
        signature = infer_signature(model_input=dummy_input_np, model_output=Y_prob_np)

        # Log the reconstructed model using explicit signature
        print("Logging best model to MLflow...")
        mlflow.pytorch.log_model(
            best_model,
            name="models",
            registered_model_name=config.registered_model_name,
            pip_requirements=pip_reqs,
            signature=signature,
        )

        print(f"✅ Successfully registered model: {config.registered_model_name}")
        print(f"   - Best validation AUC: {best_val_auc:.4f}")
        print(f"   - Best test AUC: {best_test_auc:.4f}")
        print(f"   - From fold: {best_fold}")

        # Log additional metrics for the best model
        mlflow.set_tag("best_model_fold", str(best_fold))
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_metric("best_test_auc", best_test_auc)

        return best_model

    except Exception as e:
        print(f"Error logging model to MLflow: {e}")
        return None


def run_training(config: MILTrainingConfig) -> TrainingResults:
    """
    Main training function.
    Starts the main MLflow run and coordinates cross-validation.
    """
    if not config.exp_code:
        raise ValueError("exp_code must be provided in config")

    # Setup MLflow experiment
    experiment_name = f"Train_{config.exp_code}"
    mlflow.set_experiment(experiment_name)

    run_name = f"CV_Seed{config.seed}_k{config.k}"
    with mlflow.start_run(run_name=run_name) as run:
        # Setup device and seed
        seed_torch(config.seed)

        # Setup dataset
        dataset = setup_dataset(config)

        # Verify split directory exists
        if not os.path.isdir(config.split_dir):
            raise FileNotFoundError(f"Split directory {config.split_dir} does not exist")

        # Log settings
        settings = config.get_settings()
        mlflow.log_params(settings)

        # Save settings to log file
        log_file = os.path.join(
            config.results_dir, f"experiment_{config.exp_code}.txt"
        )
        with open(log_file, "w") as f:
            print(settings, file=f)

        print("################# Settings ###################")
        for key, val in settings.items():
            print(f"{key}:  {val}")

        # Run training across folds
        all_test_auc: List[float] = []
        all_val_auc: List[float] = []
        all_test_acc: List[float] = []
        all_val_acc: List[float] = []

        folds = np.arange(config.start_fold, config.end_fold)
        for i in folds:
            seed_torch(config.seed)
            train_dataset, val_dataset, test_dataset = dataset.return_splits(
                from_id=False,
                csv_path=os.path.join(config.split_dir, f"splits_{i}.csv"),
            )

            datasets = (train_dataset, val_dataset, test_dataset)
            results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, config)

            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
            all_test_acc.append(test_acc)
            all_val_acc.append(val_acc)

            # Save results for this fold
            filename = os.path.join(config.results_dir, f"split_{i}_results.pkl")
            save_pkl(filename, results)

        # Save and log final summary
        final_df = pd.DataFrame({
            "folds": folds,
            "test_auc": all_test_auc,
            "val_auc": all_val_auc,
            "test_acc": all_test_acc,
            "val_acc": all_val_acc,
        })

        save_name = (
            f"summary_partial_{config.start_fold}_{config.end_fold}.csv"
            if len(folds) != config.k
            else "summary.csv"
        )
        final_summary_path = os.path.join(config.results_dir, save_name)
        final_df.to_csv(final_summary_path)

        # Log final cross-validation statistics
        mlflow.set_tag(
            "Training Info",
            f"CLAM model training with {config.data_set_name} data"
        )
        mlflow.log_artifact(final_summary_path)

        mlflow.log_metric("CV_Test_AUC_Mean", float(np.mean(all_test_auc)))
        mlflow.log_metric("CV_Test_AUC_Std", float(np.std(all_test_auc)))
        mlflow.log_metric("CV_Test_Accuracy_Mean", float(np.mean(all_test_acc)))
        mlflow.log_metric("CV_Test_Accuracy_Std", float(np.std(all_test_acc)))
        mlflow.log_metric("CV_Val_AUC_Mean", float(np.mean(all_val_auc)))
        mlflow.log_metric("CV_Val_AUC_Std", float(np.std(all_val_auc)))

        # Register the best model
        best_model = log_best_model_to_mlflow(
            config, all_val_auc, all_test_auc, folds, dataset
        )

        return TrainingResults(
            test_auc=all_test_auc,
            val_auc=all_val_auc,
            test_acc=all_test_acc,
            val_acc=all_val_acc,
            final_df=final_df,
            best_model=best_model,
            mlflow_run_id=run.info.run_id,
        )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with type hints for all arguments"""
    parser = argparse.ArgumentParser(description="Configurations for WSI Training")

    # Data parameters
    parser.add_argument("--data_root_dir", type=str, default=None, help="data directory")
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
    parser.add_argument("--k", type=int, default=10, help="number of folds (default: 10)")
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
    parser.add_argument("--exp_code", type=str, help="experiment code for saving results")
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

    # Run training
    results = run_training(config)
    print("Finished!")
    print("End script")

    return results


if __name__ == "__main__":
    main()