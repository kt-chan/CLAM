from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from models.model_clam import CLAM_MB, CLAM_SB
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.train_utils import train
from dataset_modules.dataset_generic import (
    Generic_WSI_Classification_Dataset,
    Generic_MIL_Dataset,
)
from create_splits_seq import create_splits

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types.schema import TensorSpec

# NOTE: The CLAMForMLflow class is removed as it's no longer needed.


class MILTrainingConfig:
    """Configuration class for MIL training"""

    def __init__(self, **kwargs):
        # Data parameters
        self.data_root_dir = kwargs.get("data_root_dir", None)
        self.data_set_name = kwargs.get("data_set_name", None)
        self.embed_dim = kwargs.get("embed_dim", 1024)

        # Training parameters
        self.max_epochs = kwargs.get("max_epochs", 200)
        self.lr = kwargs.get("lr", 1e-4)
        self.label_frac = kwargs.get("label_frac", 1.0)
        self.reg = kwargs.get("reg", 1e-5)
        self.seed = kwargs.get("seed", 1)
        self.opt = kwargs.get("opt", "adam")
        self.drop_out = kwargs.get("drop_out", 0.25)
        self.bag_loss = kwargs.get("bag_loss", "ce")

        # Cross-validation parameters
        self.k = kwargs.get("k", 10)
        self.k_start = kwargs.get("k_start", -1)
        self.k_end = kwargs.get("k_end", -1)

        # Model parameters
        self.model_type = kwargs.get("model_type", "clam_sb")
        self.model_size = kwargs.get("model_size", "small")
        self.task = kwargs.get("task", None)

        # CLAM specific parameters
        self.no_inst_cluster = kwargs.get("no_inst_cluster", False)
        self.inst_loss = kwargs.get("inst_loss", None)
        self.subtyping = kwargs.get("subtyping", False)
        self.bag_weight = kwargs.get("bag_weight", 0.7)
        self.B = kwargs.get("B", 8)

        # Other parameters
        self.results_dir = kwargs.get("results_dir", "./results")
        self.split_dir = kwargs.get("split_dir", None)
        self.log_data = kwargs.get("log_data", False)
        self.testing = kwargs.get("testing", False)
        self.early_stopping = kwargs.get("early_stopping", False)
        self.weighted_sample = kwargs.get("weighted_sample", False)
        self.exp_code = kwargs.get("exp_code", None)

        # MLflow model registration parameters
        self.registered_model_name = kwargs.get("registered_model_name", None)
        self.register_best_model = kwargs.get("register_best_model", True)

        # Derived attributes
        self._setup_derived_attributes()

    def _setup_derived_attributes(self):
        """Setup attributes that depend on other parameters"""
        if self.k_start == -1:
            self.start_fold = 0
        else:
            self.start_fold = self.k_start

        if self.k_end == -1:
            self.end_fold = self.k
        else:
            self.end_fold = self.k_end

        # Set n_classes based on task
        if self.task == "task_1_tumor_vs_normal":
            self.n_classes = 2
        elif self.task == "task_2_tumor_subtyping":
            self.n_classes = 3
        else:
            self.n_classes = None

        # Setup results directory
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)

        self.results_dir = os.path.join(
            self.results_dir, str(self.exp_code) + "_s{}".format(self.seed)
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # Setup split directory
        if self.split_dir is None:
            self.split_dir = os.path.join(
                "splits", self.task + "_{}".format(int(self.label_frac * 100))
            )
        else:
            self.split_dir = os.path.join("splits", self.split_dir)

        # Setup default registered model name
        if self.registered_model_name is None:
            self.registered_model_name = f"{self.model_type}_{self.exp_code}"

    def get_settings(self):
        """Return settings dictionary for logging"""
        settings = {
            "num_splits": self.k,
            "k_start": self.k_start,
            "k_end": self.k_end,
            "task": self.task,
            "max_epochs": self.max_epochs,
            "results_dir": self.results_dir,
            "lr": self.lr,
            "experiment": self.exp_code,
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
        }

        if self.model_type in ["clam_sb", "clam_mb"]:
            settings.update(
                {
                    "bag_weight": self.bag_weight,
                    "inst_loss": self.inst_loss,
                    "B": self.B,
                }
            )

        return settings


def setup_dataset(config: MILTrainingConfig):
    """Setup dataset based on configuration"""
    print(f"\nLoad Dataset: {config.data_set_name}")

    data_dir = os.path.join(config.data_root_dir, config.data_set_name)
    assert os.path.isdir(data_dir), f"Data directory {data_dir} does not exist"

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
            assert config.subtyping
    else:
        raise NotImplementedError(f"Task {config.task} not implemented")

    return dataset


def reconstruct_clam_model(config, model_path):
    """
    Reconstructs a CLAM model from a configuration and a saved state_dict.
    """
    # --- Step 1: Re-create the model architecture using the config ---
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

    # --- Step 2: Load the saved state_dict into the new model ---
    state_dict = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)

    # --- Step 3: Set the model to evaluation mode ---
    model.eval()

    return model


def log_best_model_to_mlflow(config, all_val_auc, all_test_auc, folds, dataset):
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
        print(f"Warning: Best model results not found at {best_model_checkpoint_path}")
        return None

    try:

        # ====================================================================
        # MLFLOW INTEGRATION: Log the best model using explicit signature
        # ====================================================================

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

        # L = number of instances (e.g., 500 patches)
        # D = embedding dimension (config.embed_dim, e.g., 1024)
        L = 500
        D = config.embed_dim

        # 1. FIX DTYPE: Create a dummy feature tensor on CPU with float32 dtype (fixes Double vs. Float)
        dummy_input_np = np.random.randn(L, D).astype(np.float32)
        dummy_input_torch = torch.from_numpy(dummy_input_np)

        # 2. Reconstruct the model architecture and load the weights
        best_model = reconstruct_clam_model(config, best_model_checkpoint_path)

        # 3. GENERATE SIGNATURE: Pass the dummy input through the model's forward
        # and capture the desired output (Y_prob) to define the signature.
        with torch.no_grad():
            output = best_model(dummy_input_torch)

            if isinstance(output, tuple):
                # If the model returns the full tuple (Y_prob, Y_hat, A_raw, results_dict)
                Y_prob = output[0]
            else:
                # If the model simplifies the return to just the prediction tensor (Y_prob)
                Y_prob = output

        # Convert the output tensor to numpy for infer_signature
        Y_prob_np = Y_prob.cpu().numpy()

        # Infer the signature using the single input and the single output tensor (Y_prob)
        signature = infer_signature(model_input=dummy_input_np, model_output=Y_prob_np)

        # 4. Log the reconstructed model using the explicit signature
        print("Logging best model to MLflow...")
        mlflow.pytorch.log_model(
            best_model,
            name="models",
            registered_model_name=config.registered_model_name,
            pip_requirements=pip_reqs,
            signature=signature,  # Pass the explicit signature
        )

        print("Model successfully logged.")
        print(f"✅ Successfully registered model: {config.registered_model_name}")
        print(f"   - Best validation AUC: {best_val_auc:.4f}")
        print(f"   - Best test AUC: {best_test_auc:.4f}")
        print(f"   - From fold: {best_fold}")

        # Log additional metrics for the best model
        mlflow.set_tag("best_model_fold", best_fold)
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_metric("best_test_auc", best_test_auc)

        return best_model

    except Exception as e:
        print(f"Error logging model to MLflow: {e}")
        return None


def run_training(config: MILTrainingConfig):
    """
    Main training function.
    Starts the main MLflow run and coordinates cross-validation.
    """

    # ====================================================================
    # MLFLOW INTEGRATION: Start the main run for cross-validation
    # ====================================================================
    # Set the MLflow experiment name
    experiment_name = f"Train_{config.exp_code}"
    mlflow.set_experiment(experiment_name)

    run_name = f"CV_Seed{config.seed}_k{config.k}"
    with mlflow.start_run(run_name=run_name) as run:

        # Setup device and seed
        seed_torch(config.seed)

        # Setup dataset
        dataset = setup_dataset(config)

        # Verify split directory exists
        assert os.path.isdir(
            config.split_dir
        ), f"Split directory {config.split_dir} does not exist"

        # Log settings
        settings = config.get_settings()

        # ====================================================================
        # MLFLOW INTEGRATION: Log all parameters
        # ====================================================================
        mlflow.log_params(settings)

        log_file = os.path.join(
            config.results_dir, "experiment_{}.txt".format(config.exp_code)
        )
        with open(log_file, "w") as f:
            print(settings, file=f)

        print("################# Settings ###################")
        for key, val in settings.items():
            print("{}:  {}".format(key, val))

        # Run training across folds
        all_test_auc = []
        all_val_auc = []
        all_test_acc = []
        all_val_acc = []

        folds = np.arange(config.start_fold, config.end_fold)
        for i in folds:
            seed_torch(config.seed)
            train_dataset, val_dataset, test_dataset = dataset.return_splits(
                from_id=False,
                csv_path=os.path.join(config.split_dir, "splits_{}.csv".format(i)),
            )

            datasets = (train_dataset, val_dataset, test_dataset)
            # NOTE: train is the core_utils.train function which now handles nested MLflow runs
            results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, config)

            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
            all_test_acc.append(test_acc)
            all_val_acc.append(val_acc)

            # Save results for this fold
            filename = os.path.join(
                config.results_dir, "split_{}_results.pkl".format(i)
            )
            save_pkl(filename, results)

        # Save and log final summary
        final_df = pd.DataFrame(
            {
                "folds": folds,
                "test_auc": all_test_auc,
                "val_auc": all_val_auc,
                "test_acc": all_test_acc,
                "val_acc": all_val_acc,
            }
        )

        if len(folds) != config.k:
            save_name = "summary_partial_{}_{}.csv".format(
                config.start_fold, config.end_fold
            )
        else:
            save_name = "summary.csv"

        final_summary_path = os.path.join(config.results_dir, save_name)
        final_df.to_csv(final_summary_path)

        # ====================================================================
        # MLFLOW INTEGRATION: Log final cross-validation statistics
        # ====================================================================
        mlflow.set_tag(
            "Training Info", f"CLAM model training with {config.data_set_name} data"
        )
        mlflow.log_artifact(final_summary_path)

        mlflow.log_metric("CV_Test_AUC_Mean", np.mean(all_test_auc))
        mlflow.log_metric("CV_Test_AUC_Std", np.std(all_test_auc))
        mlflow.log_metric("CV_Test_Accuracy_Mean", np.mean(all_test_acc))
        mlflow.log_metric("CV_Test_Accuracy_Std", np.std(all_test_acc))
        mlflow.log_metric("CV_Val_AUC_Mean", np.mean(all_val_auc))
        mlflow.log_metric("CV_Val_AUC_Std", np.std(all_val_auc))

        # ====================================================================
        # MLFLOW INTEGRATION: Register the best model
        # ====================================================================
        best_model = log_best_model_to_mlflow(
            config, all_val_auc, all_test_auc, folds, dataset
        )

        return {
            "test_auc": all_test_auc,
            "val_auc": all_val_auc,
            "test_acc": all_test_acc,
            "val_acc": all_val_acc,
            "final_df": final_df,
            "best_model": best_model,
            "mlflow_run_id": run.info.run_id,
        }


def seed_torch(seed=7):
    """Set random seed for reproducibility"""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_parser():
    """Create argument parser"""
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
        "--task", type=str, choices=["task_1_tumor_vs_normal", "task_2_tumor_subtyping"]
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


def main():
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
