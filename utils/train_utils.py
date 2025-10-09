from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

from dataset_modules.dataset_generic import Generic_MIL_Dataset, save_splits
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_mil import MIL_fc, MIL_fc_mc
from utils.file_utils import save_pkl
from utils.utils import calculate_error, get_optim, get_split_loader, seed_torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== TYPE DEFINITIONS ====================
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


class TrainingResults(TypedDict):
    """Type definition for training results"""

    test_auc: List[float]
    val_auc: List[float]
    test_acc: List[float]
    val_acc: List[float]
    final_df: pd.DataFrame
    best_model: Optional[torch.nn.Module]
    mlflow_run_id: str


class FoldResults(TypedDict):
    """Results for a single fold"""

    results_dict: Dict[str, Any]
    test_auc: float
    val_auc: float
    test_acc: float
    val_acc: float


class MILModelConfig(TypedDict, total=False):
    """Type definition for model configuration"""

    dropout: float
    n_classes: int
    embed_dim: int
    size_arg: str
    subtyping: bool
    k_sample: int


# ==================== CORE CONFIGURATION CLASS ====================
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

        # Setup directories
        os.makedirs(self.results_dir, exist_ok=True)
        if self.exp_code:
            self.results_dir = os.path.join(self.results_dir, self.exp_code)
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

        # Add CLAM-specific settings
        if self.model_type in ["clam_sb", "clam_mb"]:
            settings.update(
                {
                    "bag_weight": self.bag_weight,
                    "inst_loss": self.inst_loss,
                    "B": self.B,
                }
            )

        return settings


# ==================== UTILITY CLASSES ====================
class AccuracyLogger:
    """Accuracy logger for tracking classification performance per class"""

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.initialize()

    def initialize(self) -> None:
        """Reset all counters"""
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def log(self, Y_hat: int, Y: int) -> None:
        """Log single prediction"""
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += Y_hat == Y

    def log_batch(self, Y_hat: np.ndarray, Y: np.ndarray) -> None:
        """Log batch of predictions"""
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c: int) -> Tuple[Optional[float], int, int]:
        """
        Get accuracy summary for class c

        Returns:
            Tuple of (accuracy, correct_count, total_count)
        """
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        acc = float(correct) / count if count > 0 else None
        return acc, correct, count


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after given patience"""

    def __init__(
        self, patience: int = 20, stop_epoch: int = 50, verbose: bool = False
    ) -> None:
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        model: nn.Module,
        ckpt_name: str = "checkpoint.pt",
    ) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(
        self, val_loss: float, model: nn.Module, ckpt_name: str
    ) -> None:
        """Saves model when validation loss decreases"""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


# ==================== FACTORY CLASSES ====================
class ModelFactory:
    """Factory class for creating models and loss functions"""

    @staticmethod
    def create_loss_function(bag_loss: str, n_classes: int) -> nn.Module:
        """Create loss function based on configuration"""
        if bag_loss == "svm":
            from topk.svm import SmoothTop1SVM

            loss_fn = SmoothTop1SVM(n_classes=n_classes)
            if device.type == "cuda":
                loss_fn = loss_fn.cuda()
        else:
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    @staticmethod
    def create_instance_loss(inst_loss: Optional[str]) -> nn.Module:
        """Create instance loss function for CLAM models"""
        if inst_loss == "svm":
            from topk.svm import SmoothTop1SVM

            instance_loss_fn = SmoothTop1SVM(n_classes=2)
            if device.type == "cuda":
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        return instance_loss_fn

    @staticmethod
    def create_model(config: MILTrainingConfig) -> nn.Module:
        """Create model based on configuration"""
        if config.n_classes is None:
            raise ValueError("n_classes must be set before creating model")

        model_dict: MILModelConfig = {
            "dropout": config.drop_out,
            "n_classes": config.n_classes,
            "embed_dim": config.embed_dim,
        }

        if config.model_size is not None and config.model_type != "mil":
            model_dict["size_arg"] = config.model_size

        # Create appropriate model
        if config.model_type in ["clam_sb", "clam_mb"]:
            model = ModelFactory._create_clam_model(config, model_dict)
        else:  # MIL model
            model = ModelFactory._create_mil_model(config, model_dict)

        return model.to(device)

    @staticmethod
    def _create_clam_model(
        config: MILTrainingConfig, model_dict: MILModelConfig
    ) -> nn.Module:
        """Create CLAM model"""
        if config.subtyping:
            model_dict["subtyping"] = True

        if config.B > 0:
            model_dict["k_sample"] = config.B

        instance_loss_fn = ModelFactory.create_instance_loss(config.inst_loss)

        if config.model_type == "clam_sb":
            return CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif config.model_type == "clam_mb":
            return CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError(f"Model type {config.model_type} not implemented")

    @staticmethod
    def _create_mil_model(
        config: MILTrainingConfig, model_dict: MILModelConfig
    ) -> nn.Module:
        """Create MIL model"""
        if config.n_classes > 2:
            return MIL_fc_mc(**model_dict)
        else:
            return MIL_fc(**model_dict)


class DataLoaderFactory:
    """Factory class for creating data loaders"""

    @staticmethod
    def create_data_loaders(
        config: MILTrainingConfig, train_split: Any, val_split: Any, test_split: Any
    ) -> Tuple[Any, Any, Any]:
        """Create data loaders for training, validation, and testing"""
        train_loader = get_split_loader(
            train_split,
            training=True,
            testing=config.testing,
            weighted=config.weighted_sample,
        )
        val_loader = get_split_loader(val_split, testing=config.testing)
        test_loader = get_split_loader(test_split, testing=config.testing)
        return train_loader, val_loader, test_loader


# ==================== METRICS AND LOGGING ====================
class MetricsCalculator:
    """Utility class for calculating and logging metrics"""

    @staticmethod
    def calculate_auc(
        n_classes: int, labels: np.ndarray, probabilities: np.ndarray
    ) -> float:
        """Calculate AUC score based on number of classes"""
        if n_classes == 2:
            return roc_auc_score(labels, probabilities[:, 1])
        else:
            aucs = []
            binary_labels = label_binarize(labels, classes=list(range(n_classes)))
            for class_idx in range(n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(
                        binary_labels[:, class_idx], probabilities[:, class_idx]
                    )
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float("nan"))
            return np.nanmean(np.array(aucs))

    @staticmethod
    def log_training_metrics(
        epoch: int,
        train_loss: float,
        train_error: float,
        train_inst_loss: float,
        acc_logger: AccuracyLogger,
        n_classes: int,
        writer: Optional[Any] = None,
    ) -> None:
        """Log training metrics to MLflow and TensorBoard"""
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_error", train_error, step=epoch)
        mlflow.log_metric("train_clustering_loss", train_inst_loss, step=epoch)

        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print(f"class {i}: acc {acc}, correct {correct}/{count}")
            if writer and acc is not None:
                writer.add_scalar(f"train/class_{i}_acc", acc, epoch)
            if acc is not None:
                mlflow.log_metric(f"train_class_{i}_acc", acc, step=epoch)

        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/error", train_error, epoch)
            writer.add_scalar("train/clustering_loss", train_inst_loss, epoch)

    @staticmethod
    def log_validation_metrics(
        epoch: int,
        val_loss: float,
        val_error: float,
        auc: float,
        acc_logger: AccuracyLogger,
        n_classes: int,
        writer: Optional[Any] = None,
    ) -> None:
        """Log validation metrics to MLflow and TensorBoard"""
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_error", val_error, step=epoch)
        mlflow.log_metric("val_auc", auc, step=epoch)

        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print(f"class {i}: acc {acc}, correct {correct}/{count}")
            if writer and acc is not None:
                writer.add_scalar(f"val/class_{i}_acc", acc, epoch)
            if acc is not None:
                mlflow.log_metric(f"val_class_{i}_acc", acc, step=epoch)

        if writer:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/auc", auc, epoch)
            writer.add_scalar("val/error", val_error, epoch)

    @staticmethod
    def log_final_metrics(
        val_error: float,
        val_auc: float,
        test_error: float,
        test_auc: float,
        acc_logger: AccuracyLogger,
        n_classes: int,
    ) -> None:
        """Log final metrics to MLflow"""
        mlflow.log_metric("final_val_loss", val_error)
        mlflow.log_metric("final_val_auc", val_auc)
        mlflow.log_metric("final_test_loss", test_error)
        mlflow.log_metric("final_test_auc", test_auc)
        mlflow.log_metric("final_val_acc", 1 - val_error)
        mlflow.log_metric("final_test_acc", 1 - test_error)

        for i in range(n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print(f"class {i}: acc {acc}, correct {correct}/{count}")
            if acc is not None:
                mlflow.log_metric(f"final_test_class_{i}_acc", acc)


# ==================== ABSTRACT TRAINING STRATEGIES ====================
class TrainingStrategy(ABC):
    """Abstract base class for training strategies"""

    @abstractmethod
    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        n_classes: int,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        """Train for one epoch"""
        pass

    @abstractmethod
    def validate(
        self,
        cur: int,
        epoch: int,
        model: nn.Module,
        loader: Any,
        n_classes: int,
        early_stopping: Optional[EarlyStopping] = None,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        results_dir: Optional[str] = None,
    ) -> bool:
        """Validate the model"""
        pass


class StandardMILTrainingStrategy(TrainingStrategy):
    """Training strategy for standard MIL models"""

    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        n_classes: int,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        """Training loop for standard MIL models"""
        model.train()
        acc_logger = AccuracyLogger(n_classes=n_classes)
        train_loss = 0.0
        train_error = 0.0

        print("\n")
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)

            # Forward pass
            logits, Y_prob, Y_hat, _, _ = model(data)

            # Calculate loss
            acc_logger.log(Y_hat, label)
            loss = (
                loss_fn(logits, label)
                if loss_fn
                else nn.CrossEntropyLoss()(logits, label)
            )
            loss_value = loss.item()

            train_loss += loss_value
            if (batch_idx + 1) % 20 == 0:
                print(
                    f"batch {batch_idx}, loss: {loss_value:.4f}, "
                    f"label: {label.item()}, bag_size: {data.size(0)}"
                )

            error = calculate_error(Y_hat, label)
            train_error += error

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Calculate epoch statistics
        train_loss /= len(loader)
        train_error /= len(loader)

        print(
            f"Epoch: {epoch}, train_loss: {train_loss:.4f}, train_error: {train_error:.4f}"
        )

        # Log training metrics
        MetricsCalculator.log_training_metrics(
            epoch, train_loss, train_error, 0.0, acc_logger, n_classes, writer
        )

    def validate(
        self,
        cur: int,
        epoch: int,
        model: nn.Module,
        loader: Any,
        n_classes: int,
        early_stopping: Optional[EarlyStopping] = None,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        results_dir: Optional[str] = None,
    ) -> bool:
        """Validation loop for standard MIL models"""
        model.eval()
        acc_logger = AccuracyLogger(n_classes=n_classes)
        val_loss = 0.0
        val_error = 0.0

        probabilities = np.zeros((len(loader), n_classes))
        labels = np.zeros(len(loader))

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                data, label = data.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )

                logits, Y_prob, Y_hat, _, _ = model(data)

                acc_logger.log(Y_hat, label)
                loss = (
                    loss_fn(logits, label)
                    if loss_fn
                    else nn.CrossEntropyLoss()(logits, label)
                )

                probabilities[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()

                val_loss += loss.item()
                error = calculate_error(Y_hat, label)
                val_error += error

        val_error /= len(loader)
        val_loss /= len(loader)

        # Calculate AUC
        auc = MetricsCalculator.calculate_auc(n_classes, labels, probabilities)

        # Log validation metrics
        MetricsCalculator.log_validation_metrics(
            epoch, val_loss, val_error, auc, acc_logger, n_classes, writer
        )

        print(
            f"\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}"
        )

        # Early stopping
        if early_stopping and results_dir:
            early_stopping(
                epoch,
                val_loss,
                model,
                ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"),
            )
            if early_stopping.early_stop:
                print("Early stopping")
                return True

        return False


class CLAMTrainingStrategy(TrainingStrategy):
    """Training strategy for CLAM models with instance-level clustering"""

    def __init__(self, bag_weight: float = 0.7):
        self.bag_weight = bag_weight

    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        n_classes: int,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        """Training loop for CLAM models with instance-level clustering"""
        model.train()
        acc_logger = AccuracyLogger(n_classes=n_classes)
        inst_logger = AccuracyLogger(n_classes=n_classes)

        train_loss = 0.0
        train_error = 0.0
        train_inst_loss = 0.0
        inst_count = 0

        print("\n")
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)

            # Forward pass
            logits, Y_prob, Y_hat, _, instance_dict = model(
                data, label=label, instance_eval=True
            )

            # Calculate losses
            acc_logger.log(Y_hat, label)
            loss = (
                loss_fn(logits, label)
                if loss_fn
                else nn.CrossEntropyLoss()(logits, label)
            )
            loss_value = loss.item()

            instance_loss = instance_dict["instance_loss"]
            inst_count += 1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value

            total_loss = self.bag_weight * loss + (1 - self.bag_weight) * instance_loss

            # Log instance predictions
            inst_preds = instance_dict["inst_preds"]
            inst_labels = instance_dict["inst_labels"]
            inst_logger.log_batch(inst_preds, inst_labels)

            train_loss += loss_value
            if (batch_idx + 1) % 20 == 0:
                print(
                    f"batch {batch_idx}, loss: {loss_value:.4f}, "
                    f"instance_loss: {instance_loss_value:.4f}, "
                    f"weighted_loss: {total_loss.item():.4f}, "
                    f"label: {label.item()}, bag_size: {data.size(0)}"
                )

            error = calculate_error(Y_hat, label)
            train_error += error

            # Backward pass
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Calculate epoch statistics
        train_loss /= len(loader)
        train_error /= len(loader)

        # Log instance metrics
        if inst_count > 0:
            train_inst_loss /= inst_count
            print("\n")
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                print(f"class {i} clustering acc {acc}: correct {correct}/{count}")
                if acc is not None:
                    mlflow.log_metric(
                        f"train_inst_cluster_class_{i}_acc", acc, step=epoch
                    )

        print(
            f"Epoch: {epoch}, train_loss: {train_loss:.4f}, "
            f"train_clustering_loss: {train_inst_loss:.4f}, train_error: {train_error:.4f}"
        )

        # Log training metrics
        MetricsCalculator.log_training_metrics(
            epoch,
            train_loss,
            train_error,
            train_inst_loss,
            acc_logger,
            n_classes,
            writer,
        )

    def validate(
        self,
        cur: int,
        epoch: int,
        model: nn.Module,
        loader: Any,
        n_classes: int,
        early_stopping: Optional[EarlyStopping] = None,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        results_dir: Optional[str] = None,
    ) -> bool:
        """Validation loop for CLAM models with instance-level clustering"""
        model.eval()
        acc_logger = AccuracyLogger(n_classes=n_classes)
        inst_logger = AccuracyLogger(n_classes=n_classes)
        val_loss = 0.0
        val_error = 0.0
        val_inst_loss = 0.0
        inst_count = 0

        probabilities = np.zeros((len(loader), n_classes))
        labels = np.zeros(len(loader))

        with torch.inference_mode():
            for batch_idx, (data, label) in enumerate(loader):
                data, label = data.to(device), label.to(device)

                logits, Y_prob, Y_hat, _, instance_dict = model(
                    data, label=label, instance_eval=True
                )

                acc_logger.log(Y_hat, label)
                loss = (
                    loss_fn(logits, label)
                    if loss_fn
                    else nn.CrossEntropyLoss()(logits, label)
                )
                val_loss += loss.item()

                instance_loss = instance_dict["instance_loss"]
                inst_count += 1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value

                inst_preds = instance_dict["inst_preds"]
                inst_labels = instance_dict["inst_labels"]
                inst_logger.log_batch(inst_preds, inst_labels)

                probabilities[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()

                error = calculate_error(Y_hat, label)
                val_error += error

        val_error /= len(loader)
        val_loss /= len(loader)

        # Calculate AUC
        auc = MetricsCalculator.calculate_auc(n_classes, labels, probabilities)

        # Log validation metrics
        MetricsCalculator.log_validation_metrics(
            epoch, val_loss, val_error, auc, acc_logger, n_classes, writer
        )

        # Log instance metrics
        if inst_count > 0:
            val_inst_loss /= inst_count
            mlflow.log_metric("val_inst_loss", val_inst_loss, step=epoch)

            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                print(f"class {i} clustering acc {acc}: correct {correct}/{count}")
                if acc is not None:
                    mlflow.log_metric(
                        f"val_inst_cluster_class_{i}_acc", acc, step=epoch
                    )

        print(
            f"\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}"
        )

        if writer:
            writer.add_scalar("val/inst_loss", val_inst_loss, epoch)

        # Early stopping
        if early_stopping and results_dir:
            early_stopping(
                epoch,
                val_loss,
                model,
                ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"),
            )
            if early_stopping.early_stop:
                print("Early stopping")
                return True

        return False


class TrainingStrategyFactory:
    """Factory for creating training strategies"""

    @staticmethod
    def create_strategy(config: MILTrainingConfig) -> TrainingStrategy:
        """Create appropriate training strategy based on configuration"""
        if config.model_type in ["clam_sb", "clam_mb"] and not config.no_inst_cluster:
            return CLAMTrainingStrategy(bag_weight=config.bag_weight)
        else:
            return StandardMILTrainingStrategy()


# ==================== MODEL EVALUATOR ====================
class ModelEvaluator:
    """Handles model evaluation and summary statistics"""

    @staticmethod
    def summary(
        model: nn.Module, loader: Any, n_classes: int
    ) -> Tuple[Dict[str, Any], float, float, AccuracyLogger]:
        """Generate summary statistics for model performance"""
        acc_logger = AccuracyLogger(n_classes=n_classes)
        model.eval()
        test_error = 0.0

        all_probs = np.zeros((len(loader), n_classes))
        all_labels = np.zeros(len(loader))

        slide_ids = loader.dataset.slide_data["slide_id"]
        patient_results = {}

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            with torch.inference_mode():
                logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()

            patient_results.update(
                {
                    slide_id: {
                        "slide_id": np.array(slide_id),
                        "prob": probs,
                        "label": label.item(),
                    }
                }
            )

            error = calculate_error(Y_hat, label)
            test_error += error

        test_error /= len(loader)

        # Calculate AUC
        auc = MetricsCalculator.calculate_auc(n_classes, all_labels, all_probs)

        return patient_results, test_error, auc, acc_logger


# ==================== SINGLE FOLD TRAINER ====================
class FoldTrainer:
    """Handles training for a single fold"""

    def __init__(self, config: MILTrainingConfig, fold_id: int, folds: int):
        self.config = config
        self.fold_id = fold_id
        self.folds = folds
        self.writer = None
        self._setup_writer()

    def _setup_writer(self) -> None:
        """Setup TensorBoard writer if logging is enabled"""
        if self.config.log_data:
            from tensorboardX import SummaryWriter

            writer_dir = os.path.join(self.config.results_dir, str(self.fold_id))
            os.makedirs(writer_dir, exist_ok=True)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)

    def setup_data_splits(self, datasets: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
        """Setup and log data splits"""
        print("\nInit train/val/test splits...", end=" ")
        train_split, val_split, test_split = datasets

        # Save splits
        split_csv_path = os.path.join(
            self.config.results_dir, f"splits_{self.fold_id}.csv"
        )
        save_splits(datasets, ["train", "val", "test"], split_csv_path)
        mlflow.log_artifact(split_csv_path)
        print("Done!")

        print(f"Training on {len(train_split)} samples")
        print(f"Validating on {len(val_split)} samples")
        print(f"Testing on {len(test_split)} samples")

        return train_split, val_split, test_split

    def train(self, datasets: Tuple[Any, Any, Any]) -> FoldResults:
        """Train for a single fold with MLflow tracking"""
        with mlflow.start_run(run_name=f"Fold {self.fold_id}", nested=True):
            # Log hyperparameters
            mlflow.log_params(
                {
                    "fold": self.fold_id,
                    "max_epochs": self.config.max_epochs,
                    "lr": self.config.lr,
                    "model_type": self.config.model_type,
                    "bag_loss": self.config.bag_loss,
                    "drop_out": self.config.drop_out,
                    "n_classes": self.config.n_classes,
                }
            )

            print(f"\nTraining Fold {self.fold_id}/{self.folds}!")

            # Setup data
            train_split, val_split, test_split = self.setup_data_splits(datasets)

            # Setup components
            loss_fn = ModelFactory.create_loss_function(
                self.config.bag_loss, self.config.n_classes
            )
            model = ModelFactory.create_model(self.config)
            optimizer = get_optim(model, self.config)
            train_loader, val_loader, test_loader = (
                DataLoaderFactory.create_data_loaders(
                    self.config, train_split, val_split, test_split
                )
            )

            early_stopping = (
                EarlyStopping(patience=20, stop_epoch=50, verbose=True)
                if self.config.early_stopping
                else None
            )

            # Get training strategy
            strategy = TrainingStrategyFactory.create_strategy(self.config)

            # Training loop
            stop_training = self._run_training_loop(
                model,
                train_loader,
                val_loader,
                strategy,
                loss_fn,
                optimizer,
                early_stopping,
            )

            # Load best/final model
            ckpt_path = os.path.join(
                self.config.results_dir, f"s_{self.fold_id}_checkpoint.pt"
            )
            if self.config.early_stopping and early_stopping and stop_training:
                model.load_state_dict(torch.load(ckpt_path))
            else:
                torch.save(model.state_dict(), ckpt_path)

            mlflow.log_artifact(ckpt_path)

            # Final evaluation
            return self._final_evaluation(model, val_loader, test_loader)

    def _run_training_loop(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        strategy: TrainingStrategy,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        early_stopping: Optional[EarlyStopping],
    ) -> bool:
        """Run the main training loop"""
        stop_training = False

        for epoch in range(self.config.max_epochs):
            print("#" * 128)
            print(
                f"Running training for fold:{self.fold_id}/{self.folds}, on epoch: {epoch}/{self.config.max_epochs} ..."
            )
            print("#" * 128)

            # Train epoch
            strategy.train_epoch(
                epoch,
                model,
                train_loader,
                optimizer,
                self.config.n_classes,
                self.writer,
                loss_fn,
            )

            # Validate
            stop_training = strategy.validate(
                self.fold_id,
                epoch,
                model,
                val_loader,
                self.config.n_classes,
                early_stopping,
                self.writer,
                loss_fn,
                self.config.results_dir,
            )

            if stop_training:
                break

        return stop_training

    def _final_evaluation(
        self, model: nn.Module, val_loader: Any, test_loader: Any
    ) -> FoldResults:
        """Perform final evaluation and return results"""
        # Validation evaluation
        _, val_error, val_auc, _ = ModelEvaluator.summary(
            model, val_loader, self.config.n_classes
        )
        print(f"Val error: {val_error:.4f}, ROC AUC: {val_auc:.4f}")

        # Test evaluation
        results_dict, test_error, test_auc, acc_logger = ModelEvaluator.summary(
            model, test_loader, self.config.n_classes
        )
        print(f"Test error: {test_error:.4f}, ROC AUC: {test_auc:.4f}")

        # Log final metrics
        MetricsCalculator.log_final_metrics(
            val_error, val_auc, test_error, test_auc, acc_logger, self.config.n_classes
        )

        # Cleanup
        if self.writer:
            self.writer.close()

        return {
            "results_dict": results_dict,
            "test_auc": test_auc,
            "val_auc": val_auc,
            "test_acc": 1 - test_error,
            "val_acc": 1 - val_error,
        }


# ==================== MODEL MANAGER ====================
class ModelManager:
    """Manages model reconstruction and MLflow registration"""

    @staticmethod
    def reconstruct_clam_model(
        config: MILTrainingConfig, model_path: str
    ) -> torch.nn.Module:
        """Reconstructs a CLAM model from configuration and saved state_dict"""
        if config.n_classes is None:
            raise ValueError("n_classes must be set before reconstructing model")

        # Create model architecture
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

        # Load state_dict
        state_dict = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    @staticmethod
    def log_best_model_to_mlflow(
        config: MILTrainingConfig,
        all_val_auc: List[float],
        all_test_auc: List[float],
        folds: np.ndarray,
        dataset: Generic_MIL_Dataset,
    ) -> Optional[torch.nn.Module]:
        """Identify the best model from cross-validation and log it to MLflow"""
        if not config.register_best_model:
            print("Model registration is disabled. Skipping...")
            return None

        # Find the best fold
        best_fold_idx = np.argmax(all_val_auc)
        best_fold = folds[best_fold_idx]
        best_val_auc = all_val_auc[best_fold_idx]
        best_test_auc = all_test_auc[best_fold_idx]

        print(
            f"Best model from fold {best_fold} with validation AUC: {best_val_auc:.4f}"
        )

        # Load the best model
        best_model_path = os.path.join(
            config.results_dir, f"s_{best_fold}_checkpoint.pt"
        )

        if not os.path.exists(best_model_path):
            print(f"Warning: Best model checkpoint not found at {best_model_path}")
            return None

        try:
            # Define requirements
            pip_reqs = [
                f"torch=={torch.__version__}",
                f"mlflow=={mlflow.__version__}",
                "numpy",
                "pandas",
            ]

            # Create dummy input for signature inference
            L, D = 500, config.embed_dim  # instances and embedding dimension
            dummy_input_np = np.random.randn(L, D).astype(np.float32)
            dummy_input_torch = torch.from_numpy(dummy_input_np)

            # Reconstruct model
            best_model = ModelManager.reconstruct_clam_model(config, best_model_path)

            # Generate signature
            with torch.no_grad():
                output = best_model(dummy_input_torch)
                Y_prob = output[0] if isinstance(output, tuple) else output

            Y_prob_np = Y_prob.cpu().numpy()
            signature = infer_signature(
                model_input=dummy_input_np, model_output=Y_prob_np
            )

            # Log model to MLflow
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

            # Log additional metrics
            mlflow.set_tag("best_model_fold", str(best_fold))
            mlflow.log_metric("best_val_auc", best_val_auc)
            mlflow.log_metric("best_test_auc", best_test_auc)

            return best_model

        except Exception as e:
            print(f"Error logging model to MLflow: {e}")
            return None


# ==================== MAIN TRAINING ORCHESTRATOR ====================
class CrossValidationTrainer:
    """Main orchestrator for cross-validation training"""

    def __init__(self, config: MILTrainingConfig, dataset: Generic_MIL_Dataset):
        self.config = config
        self.dataset = dataset
        self._validate_config()
        # ✅ Set seed exactly ONCE
        print(f"Setting random seed: {config.seed}")
        seed_torch(config.seed)

    def _validate_config(self) -> None:
        """Validate configuration before training"""
        if not self.config.exp_code:
            raise ValueError("exp_code must be provided in config")

        if not os.path.isdir(self.config.split_dir):
            raise FileNotFoundError(
                f"Split directory {self.config.split_dir} does not exist"
            )

    def setup_mlflow_experiment(self) -> mlflow.ActiveRun:
        """Setup MLflow experiment and return the active run"""
        experiment_name = f"Train_{self.config.exp_code}"
        mlflow.set_experiment(experiment_name)

        run_name = f"CV_Seed{self.config.seed}_k{self.config.k}"
        return mlflow.start_run(run_name=run_name)

    def log_experiment_settings(self) -> None:
        """Log experiment settings to MLflow and file"""
        settings = self.config.get_settings()
        mlflow.log_params(settings)

        # Save settings to log file
        log_file = os.path.join(
            self.config.results_dir, f"experiment_{self.config.exp_code}.txt"
        )
        with open(log_file, "w") as f:
            print(settings, file=f)

        print("################# Settings ###################")
        for key, val in settings.items():
            print(f"{key}:  {val}")

    def run_cross_validation(self) -> TrainingResults:
        """Run complete cross-validation training"""
        with self.setup_mlflow_experiment() as run:
            # Log settings
            self.log_experiment_settings()

            # Run training across folds
            results = self._train_all_folds()

            # Log final results
            self._log_final_results(results)

            # Register best model
            best_model = ModelManager.log_best_model_to_mlflow(
                self.config,
                results["val_auc"],
                results["test_auc"],
                results["folds"],
                self.dataset,
            )

            return TrainingResults(
                test_auc=results["test_auc"],
                val_auc=results["val_auc"],
                test_acc=results["test_acc"],
                val_acc=results["val_acc"],
                final_df=results["final_df"],
                best_model=best_model,
                mlflow_run_id=run.info.run_id,
            )

    def _train_all_folds(self) -> Dict[str, Any]:
        """Train all folds and collect results"""
        all_test_auc: List[float] = []
        all_val_auc: List[float] = []
        all_test_acc: List[float] = []
        all_val_acc: List[float] = []

        folds = np.arange(self.config.start_fold, self.config.end_fold)

        for fold_idx in folds:
            # Get dataset splits for this fold
            train_dataset, val_dataset, test_dataset = self.dataset.return_splits(
                from_id=False,
                csv_path=os.path.join(self.config.split_dir, f"splits_{fold_idx}.csv"),
            )

            datasets = (train_dataset, val_dataset, test_dataset)

            # Train this fold
            fold_trainer = FoldTrainer(self.config, fold_idx, folds.size)
            fold_results = fold_trainer.train(datasets)

            # Collect results
            all_test_auc.append(fold_results["test_auc"])
            all_val_auc.append(fold_results["val_auc"])
            all_test_acc.append(fold_results["test_acc"])
            all_val_acc.append(fold_results["val_acc"])

            # Save results for this fold
            filename = os.path.join(
                self.config.results_dir, f"split_{fold_idx}_results.pkl"
            )
            save_pkl(filename, fold_results["results_dict"])

        return {
            "folds": folds,
            "test_auc": all_test_auc,
            "val_auc": all_val_auc,
            "test_acc": all_test_acc,
            "val_acc": all_val_acc,
            "final_df": self._create_summary_dataframe(
                folds, all_test_auc, all_val_auc, all_test_acc, all_val_acc
            ),
        }

    def _create_summary_dataframe(
        self,
        folds: np.ndarray,
        all_test_auc: List[float],
        all_val_auc: List[float],
        all_test_acc: List[float],
        all_val_acc: List[float],
    ) -> pd.DataFrame:
        """Create and save summary dataframe"""
        final_df = pd.DataFrame(
            {
                "folds": folds,
                "test_auc": all_test_auc,
                "val_auc": all_val_auc,
                "test_acc": all_test_acc,
                "val_acc": all_val_acc,
            }
        )

        save_name = (
            f"summary_partial_{self.config.start_fold}_{self.config.end_fold}.csv"
            if len(folds) != self.config.k
            else "summary.csv"
        )
        final_summary_path = os.path.join(self.config.results_dir, save_name)
        final_df.to_csv(final_summary_path)

        return final_df

    def _log_final_results(self, results: Dict[str, Any]) -> None:
        """Log final cross-validation results to MLflow"""
        mlflow.set_tag(
            "Training Info",
            f"CLAM model training with {self.config.data_set_name} data",
        )

        # Log summary artifact
        final_summary_path = os.path.join(
            self.config.results_dir,
            (
                "summary.csv"
                if len(results["folds"]) == self.config.k
                else f"summary_partial_{self.config.start_fold}_{self.config.end_fold}.csv"
            ),
        )
        mlflow.log_artifact(final_summary_path)

        # Log final metrics
        mlflow.log_metric("CV_Test_AUC_Mean", float(np.mean(results["test_auc"])))
        mlflow.log_metric("CV_Test_AUC_Std", float(np.std(results["test_auc"])))
        mlflow.log_metric("CV_Test_Accuracy_Mean", float(np.mean(results["test_acc"])))
        mlflow.log_metric("CV_Test_Accuracy_Std", float(np.std(results["test_acc"])))
        mlflow.log_metric("CV_Val_AUC_Mean", float(np.mean(results["val_auc"])))
        mlflow.log_metric("CV_Val_AUC_Std", float(np.std(results["val_auc"])))


# ==================== PUBLIC INTERFACE ====================
def run_training(
    config: MILTrainingConfig, dataset: Generic_MIL_Dataset
) -> TrainingResults:
    """Main training function with cross-validation"""
    trainer = CrossValidationTrainer(config, dataset)
    return trainer.run_cross_validation()
