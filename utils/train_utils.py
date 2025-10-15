from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import auc as calc_auc
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import label_binarize

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

from dataset_modules.dataset_generic import Generic_MIL_Dataset, save_splits
from models.model_clam import CLAM_MB, CLAM_SB, CLAM_MB_Regression, CLAM_SB_Regression
from models.model_mil import MIL_fc, MIL_fc_mc
from utils.file_utils import save_pkl
from utils.utils import (
    calculate_error,
    get_optim,
    get_split_loader,
    seed_torch,
    TaskType,
)
from utils.utils import TaskType

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


# ==================== UPDATED METRICS LOGGER CLASSES ====================
class BaseMetricsLogger(ABC):
    """Base class for metrics loggers"""

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        """Reset all counters and metrics"""
        pass

    @abstractmethod
    def log(
        self, Y_hat: Union[int, float, torch.Tensor], Y: Union[int, float, torch.Tensor]
    ) -> None:
        """Log single prediction"""
        pass

    @abstractmethod
    def log_batch(
        self, Y_hat: Union[np.ndarray, torch.Tensor], Y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Log batch of predictions"""
        pass

    @abstractmethod
    def get_summary(self, c: Optional[int] = None) -> Tuple[Optional[float], int, int]:
        """Get metrics summary"""
        pass

    @abstractmethod
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all computed metrics as dictionary"""
        pass


class ClassificationMetricsLogger(BaseMetricsLogger):
    """Metrics logger for classification tasks (backward compatible with original AccuracyLogger)"""

    def __init__(self, n_classes: int, task_type: TaskType = TaskType.BINARY):
        self.n_classes = n_classes
        super().__init__(task_type)

    def initialize(self) -> None:
        """Reset all counters"""
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def log(self, Y_hat: Union[int, torch.Tensor], Y: Union[int, torch.Tensor]) -> None:
        """Log single prediction"""
        # Detach tensors from computation graph before conversion
        if isinstance(Y_hat, torch.Tensor):
            Y_hat = (
                Y_hat.detach().item()
                if Y_hat.dim() == 0
                else int(Y_hat.detach().cpu().numpy())
            )
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().item() if Y.dim() == 0 else int(Y.detach().cpu().numpy())

        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += Y_hat == Y

    def log_batch(
        self, Y_hat: Union[np.ndarray, torch.Tensor], Y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Log batch of predictions"""
        if isinstance(Y_hat, torch.Tensor):
            Y_hat = Y_hat.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()

        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c: Optional[int] = None) -> Tuple[Optional[float], int, int]:
        """
        Get accuracy summary for class c

        Args:
            c: Class index. If None, returns overall accuracy.

        Returns:
            Tuple of (accuracy, correct_count, total_count)
        """
        if c is not None:
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            acc = float(correct) / count if count > 0 else None
            return acc, correct, count
        else:
            # Return overall accuracy
            total_count = sum(item["count"] for item in self.data)
            total_correct = sum(item["correct"] for item in self.data)
            acc = float(total_correct) / total_count if total_count > 0 else None
            return acc, total_correct, total_count

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all computed metrics as dictionary"""
        metrics = {}
        total_count = sum(item["count"] for item in self.data)
        total_correct = sum(item["correct"] for item in self.data)

        if total_count > 0:
            metrics["overall_accuracy"] = float(total_correct) / total_count

        for i in range(self.n_classes):
            acc, correct, count = self.get_summary(i)
            if acc is not None:
                metrics[f"class_{i}_accuracy"] = acc
                metrics[f"class_{i}_count"] = count

        return metrics


class RegressionMetricsLogger(BaseMetricsLogger):
    """Metrics logger for regression tasks with support for multi-output regression"""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION):
        super().__init__(task_type)

    def initialize(self) -> None:
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.total_count = 0

    def log(
        self,
        Y_hat: Union[float, torch.Tensor, np.ndarray],
        Y: Union[float, torch.Tensor, np.ndarray],
    ) -> None:
        """Log single prediction - handle multi-output regression"""
        # Convert tensors to numpy arrays, detaching from computation graph if needed
        if isinstance(Y_hat, torch.Tensor):
            Y_hat = Y_hat.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()

        # Ensure we're working with numpy arrays
        Y_hat = np.asarray(Y_hat)
        Y = np.asarray(Y)

        # Handle multi-output case (like [M_primary, M_secondary])
        if Y_hat.ndim == 1 and Y_hat.size > 1:
            # This is a multi-output prediction for a single sample
            # We'll flatten it and treat each output as a separate prediction
            for i in range(Y_hat.size):
                self.predictions.append(float(Y_hat[i]))
                self.targets.append(float(Y[i]))
                self.total_count += 1
        else:
            # Single output or already flattened
            # Flatten any multi-dimensional arrays
            Y_hat_flat = Y_hat.flatten()
            Y_flat = Y.flatten()

            for pred, target in zip(Y_hat_flat, Y_flat):
                self.predictions.append(float(pred))
                self.targets.append(float(target))
                self.total_count += 1

    def log_batch(
        self, Y_hat: Union[np.ndarray, torch.Tensor], Y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Log batch of predictions - handle multi-output regression"""
        if isinstance(Y_hat, torch.Tensor):
            Y_hat = Y_hat.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()

        # Ensure we're working with numpy arrays
        Y_hat = np.asarray(Y_hat)
        Y = np.asarray(Y)

        # Handle different dimensionalities
        if Y_hat.ndim == 1:
            # Single output, single sample or batch of single outputs
            Y_hat_flat = Y_hat.flatten()
            Y_flat = Y.flatten()
        elif Y_hat.ndim == 2:
            if Y_hat.shape[1] == 1:
                # Batch of single outputs: (batch_size, 1)
                Y_hat_flat = Y_hat.flatten()
                Y_flat = Y.flatten()
            else:
                # Multi-output regression: (batch_size, n_outputs)
                # We'll flatten all outputs together
                Y_hat_flat = Y_hat.reshape(-1)
                Y_flat = Y.reshape(-1)
        else:
            # Higher dimensions - flatten completely
            Y_hat_flat = Y_hat.reshape(-1)
            Y_flat = Y.reshape(-1)

        # Add all predictions and targets
        self.predictions.extend(Y_hat_flat.astype(float).tolist())
        self.targets.extend(Y_flat.astype(float).tolist())
        self.total_count += len(Y_hat_flat)

    def get_summary(self, c: Optional[int] = None) -> Tuple[Optional[float], int, int]:
        """
        Get regression metrics summary

        Args:
            c: If provided, returns metrics for specific output index in multi-output case

        Returns:
            Tuple of (MAE, total_count, total_count)
        """
        if self.total_count == 0:
            return None, 0, 0

        if c is not None:
            # For multi-output case, filter predictions and targets for specific output
            # This assumes predictions are stored as [output1_sample1, output2_sample1, output1_sample2, output2_sample2, ...]
            # We need to know the number of outputs to do this properly
            # For now, return overall MAE
            pass

        mae = mean_absolute_error(self.targets, self.predictions)
        return mae, self.total_count, self.total_count

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all computed regression metrics as dictionary"""
        if self.total_count == 0:
            return {}

        # Ensure targets and predictions are NumPy arrays for calculation
        targets_np = np.array(self.targets)
        predictions_np = np.array(self.predictions)

        metrics = {
            "mae": mean_absolute_error(targets_np, predictions_np),
            "mse": mean_squared_error(targets_np, predictions_np),
            "rmse": np.sqrt(mean_squared_error(targets_np, predictions_np)),
            "total_count": self.total_count,
        }

        # Additional regression metrics
        if self.total_count > 1:
            # R2 calculation denominator (Total Sum of Squares: SStotal)
            ss_total = np.sum((targets_np - np.mean(targets_np)) ** 2)

            # Fix for divided by zero: check if the denominator is zero
            if ss_total == 0:
                # If targets are all the same, R2 is typically 1.0 if predictions match
                # and the model is trivial, or excluded/set to 0.0.
                # Here we set it to 1.0 only if the model is perfect (i.e., MSE is 0)
                # otherwise we might set it to 0 or nan. Given the context, 1.0 is safest
                # if the constant prediction is the target.

                # Check if the model also predicted the constant value perfectly
                if np.sum((targets_np - predictions_np) ** 2) == 0:
                    metrics["r2"] = 1.0
                else:
                    # If the data is constant, but the predictions are wrong
                    # R2 is often considered undefined or 0.0, or handled by the caller.
                    # We will set it to 0.0 (as the simple model of predicting the mean
                    # performs better than our model).
                    metrics["r2"] = 0.0
            else:
                # Numerator (Residual Sum of Squares: SSres)
                ss_residual = np.sum((targets_np - predictions_np) ** 2)

                metrics["r2"] = 1.0 - (ss_residual / ss_total)

        return metrics


class MetricsLoggerFactory:
    """Factory for creating appropriate metrics logger based on task type"""

    @staticmethod
    def create_logger(task_type: TaskType, n_classes: int = -1) -> BaseMetricsLogger:
        """Create appropriate metrics logger based on task type"""
        if task_type == TaskType.REGRESSION:
            return RegressionMetricsLogger(task_type)
        else:  # BINARY or MULTICLASS
            if n_classes <= 0:
                raise ValueError(
                    f"n_classes must be positive for classification, got {n_classes}"
                )
            return ClassificationMetricsLogger(n_classes, task_type)


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
        self.task = kwargs.get("task")

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
            self.task: TaskType = TaskType.BINARY
            self.n_classes = 2
        elif self.task == "task_2_tumor_subtyping":
            self.task: TaskType = TaskType.MULTICLASS
            self.n_classes = 3
        elif self.task == "task_3_tumor_count":
            self.task: TaskType = TaskType.REGRESSION
            self.n_classes = -1
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


# ==================== UPDATED METRICS CALCULATOR ====================
class MetricsCalculator:
    """Utility class for calculating and logging metrics"""

    @staticmethod
    def calculate_classification_metrics(
        n_classes: int, labels: np.ndarray, probabilities: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics (AUC, etc.)"""
        metrics = {}

        if n_classes == 2:
            metrics["auc"] = roc_auc_score(labels, probabilities[:, 1])
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
            metrics["auc"] = np.nanmean(np.array(aucs))

        return metrics

    @staticmethod
    def calculate_regression_metrics(
        labels: np.ndarray, predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        metrics = {
            "mae": mean_absolute_error(labels, predictions),
            "mse": mean_squared_error(labels, predictions),
            "rmse": np.sqrt(mean_squared_error(labels, predictions)),
        }

        # Calculate R-squared
        if len(labels) > 1:
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return metrics

    @staticmethod
    def log_training_metrics(
        epoch: int,
        train_loss: float,
        train_error: float,
        train_inst_loss: float,
        metrics_logger: BaseMetricsLogger,
        n_classes: int,
        writer: Optional[Any] = None,
    ) -> None:
        """Log training metrics to MLflow and TensorBoard"""
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_error", train_error, step=epoch)

        if train_inst_loss > 0:  # Only log if instance loss is used
            mlflow.log_metric("train_clustering_loss", train_inst_loss, step=epoch)

        # Log task-specific metrics
        all_metrics = metrics_logger.get_all_metrics()
        for metric_name, metric_value in all_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value, step=epoch)
            if writer:
                writer.add_scalar(f"train/{metric_name}", metric_value, epoch)

        # For classification, also log per-class accuracy (backward compatibility)
        if isinstance(metrics_logger, ClassificationMetricsLogger):
            for i in range(n_classes):
                acc, correct, count = metrics_logger.get_summary(i)
                if acc is not None:
                    mlflow.log_metric(f"train_class_{i}_acc", acc, step=epoch)
                    if writer:
                        writer.add_scalar(f"train/class_{i}_acc", acc, epoch)

        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/error", train_error, epoch)
            if train_inst_loss > 0:
                writer.add_scalar("train/clustering_loss", train_inst_loss, epoch)

    @staticmethod
    def log_validation_metrics(
        epoch: int,
        val_loss: float,
        val_error: float,
        auc: Optional[float],
        metrics_logger: BaseMetricsLogger,
        n_classes: int,
        writer: Optional[Any] = None,
        task_type: TaskType = TaskType.BINARY,
    ) -> None:
        """Log validation metrics to MLflow and TensorBoard"""
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_error", val_error, step=epoch)

        if auc is not None and task_type != TaskType.REGRESSION:
            mlflow.log_metric("val_auc", auc, step=epoch)

        # Log task-specific metrics
        all_metrics = metrics_logger.get_all_metrics()
        for metric_name, metric_value in all_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)
            if writer:
                writer.add_scalar(f"val/{metric_name}", metric_value, epoch)

        # For classification, also log per-class accuracy (backward compatibility)
        if isinstance(metrics_logger, ClassificationMetricsLogger):
            for i in range(n_classes):
                acc, correct, count = metrics_logger.get_summary(i)
                if acc is not None:
                    mlflow.log_metric(f"val_class_{i}_acc", acc, step=epoch)
                    if writer:
                        writer.add_scalar(f"val/class_{i}_acc", acc, epoch)

        if writer:
            writer.add_scalar("val/loss", val_loss, epoch)
            if auc is not None:
                writer.add_scalar("val/auc", auc, epoch)
            writer.add_scalar("val/error", val_error, epoch)

    @staticmethod
    def log_final_metrics(
        val_error: float,
        val_auc: Optional[float],
        test_error: float,
        test_auc: Optional[float],
        metrics_logger: BaseMetricsLogger,
        n_classes: int,
        task_type: TaskType = TaskType.BINARY,
    ) -> None:
        """Log final metrics to MLflow"""
        mlflow.log_metric("final_val_loss", val_error)
        mlflow.log_metric("final_test_loss", test_error)

        if val_auc is not None and task_type != TaskType.REGRESSION:
            mlflow.log_metric("final_val_auc", val_auc)
        if test_auc is not None and task_type != TaskType.REGRESSION:
            mlflow.log_metric("final_test_auc", test_auc)

        if task_type != TaskType.REGRESSION:
            mlflow.log_metric("final_val_acc", 1 - val_error)
            mlflow.log_metric("final_test_acc", 1 - test_error)

        # Log task-specific final metrics
        all_metrics = metrics_logger.get_all_metrics()
        for metric_name, metric_value in all_metrics.items():
            mlflow.log_metric(f"final_test_{metric_name}", metric_value)

        # For classification, also log per-class accuracy (backward compatibility)
        if isinstance(metrics_logger, ClassificationMetricsLogger):
            for i in range(n_classes):
                acc, correct, count = metrics_logger.get_summary(i)
                if acc is not None:
                    mlflow.log_metric(f"final_test_class_{i}_acc", acc)


# ==================== FACTORY CLASSES ====================
class ModelFactory:
    """Factory class for creating models and loss functions"""

    @staticmethod
    def create_loss_function(
        bag_loss: str, n_classes: int, task: TaskType
    ) -> nn.Module:
        """Create loss function based on configuration and task type"""
        if task == TaskType.REGRESSION:
            # Regression-specific loss functions
            if bag_loss == "mse":
                loss_fn = nn.MSELoss()
            elif bag_loss == "l1":
                loss_fn = nn.L1Loss()
            elif bag_loss == "smooth_l1":
                loss_fn = nn.SmoothL1Loss()
            elif bag_loss == "huber":
                loss_fn = nn.HuberLoss()
            else:
                # Default to MSE for regression
                loss_fn = nn.MSELoss()
                print(f"Using default MSE loss for regression (requested: {bag_loss})")
        else:
            # Classification loss functions
            if bag_loss == "svm":
                from topk.svm import SmoothTop1SVM

                loss_fn = SmoothTop1SVM(n_classes=n_classes)
                if device.type == "cuda":
                    loss_fn = loss_fn.cuda()
            else:
                loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    @staticmethod
    def create_instance_loss(inst_loss: Optional[str], task: TaskType) -> nn.Module:
        """Create instance loss function for CLAM models"""
        if task == TaskType.REGRESSION:
            # For regression, instance loss is still classification-based (pattern detection)
            if inst_loss == "svm":
                from topk.svm import SmoothTop1SVM

                instance_loss_fn = SmoothTop1SVM(n_classes=2)
                if device.type == "cuda":
                    instance_loss_fn = instance_loss_fn.cuda()
            else:
                instance_loss_fn = nn.CrossEntropyLoss()
        else:
            # Classification task
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
        if config.model_type in ["clam_sb", "clam_mb", "clam_sbr", "clam_mbr"]:
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

        instance_loss_fn = ModelFactory.create_instance_loss(
            config.inst_loss, config.task
        )

        if config.task == TaskType.BINARY or config.task == TaskType.MULTICLASS:
            if config.model_type == "clam_sb":
                return CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            elif config.model_type == "clam_mb":
                return CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif config.task == TaskType.REGRESSION:
            # Add regression-specific parameters
            regression_params = {
                "min_score": 3.0,  # Default Gleason min
                "max_score": 5.0,  # Default Gleason max
            }
            model_dict.update(regression_params)

            if config.model_type == "clam_sb" or config.model_type == "clam_sbr":
                return CLAM_SB_Regression(
                    **model_dict, instance_loss_fn=instance_loss_fn
                )
            elif config.model_type == "clam_mb" or config.model_type == "clam_mbr":
                return CLAM_MB_Regression(
                    **model_dict, instance_loss_fn=instance_loss_fn
                )
        else:
            raise NotImplementedError(f"Model type {config.model_type} not implemented")

    @staticmethod
    def _create_mil_model(
        config: MILTrainingConfig, model_dict: MILModelConfig
    ) -> nn.Module:
        """Create MIL model"""
        if config.task == TaskType.REGRESSION:
            # For regression MIL models
            if hasattr(config, "regression_outputs"):
                output_dim = config.regression_outputs
            else:
                output_dim = 1  # Default to single output regression

            if config.n_classes > 2:
                # Multi-output regression
                return MIL_fc_mc(**model_dict)
            else:
                # Single output regression
                return MIL_fc(**model_dict)
        else:
            # Classification MIL models
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

        print(f"Creating data loaders for task: {config.task}")

        train_loader = get_split_loader(
            train_split,
            training=True,
            testing=config.testing,
            weighted=config.weighted_sample,
            task=config.task,
        )
        val_loader = get_split_loader(
            val_split, testing=config.testing, task=config.task
        )
        test_loader = get_split_loader(
            test_split, testing=config.testing, task=config.task
        )
        return train_loader, val_loader, test_loader

    @staticmethod
    def create_loader_for_split(
        config: MILTrainingConfig,
        split_dataset: Any,
        training: bool = False,
        testing: bool = False,
        weighted: bool = False,
    ) -> Any:
        """Create a data loader for a specific split using task from config"""
        return get_split_loader(
            split_dataset,
            training=training,
            testing=testing,
            weighted=weighted,
            task=config.task,
        )


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
        task_type: TaskType,
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
        task_type: TaskType,
        n_classes: int,
        early_stopping: Optional[EarlyStopping] = None,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        results_dir: Optional[str] = None,
        **kwargs,
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
        task_type: TaskType,
        n_classes: int,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        """Training loop for standard MIL models"""
        model.train()
        task_type = kwargs.get("task_type", TaskType.BINARY)
        metrics_logger = MetricsLoggerFactory.create_logger(task_type, n_classes)
        train_loss = 0.0
        train_error = 0.0

        print("\n")
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)

            # Forward pass
            logits, Y_prob, Y_hat, _, _ = model(data)

            # Calculate loss
            metrics_logger.log(Y_hat, label)
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
            epoch, train_loss, train_error, 0.0, metrics_logger, n_classes, writer
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
        **kwargs,
    ) -> bool:
        """Validation loop for standard MIL models"""
        model.eval()
        task_type = kwargs.get("task_type", TaskType.BINARY)
        metrics_logger = MetricsLoggerFactory.create_logger(task_type, n_classes)
        val_loss = 0.0
        val_error = 0.0

        probabilities = (
            np.zeros((len(loader), n_classes))
            if task_type != TaskType.REGRESSION
            else np.zeros(len(loader))
        )
        labels = np.zeros(len(loader))
        predictions = np.zeros(len(loader))

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                data, label = data.to(device, non_blocking=True), label.to(
                    device, non_blocking=True
                )

                logits, Y_prob, Y_hat, _, _ = model(data)

                metrics_logger.log(Y_hat, label)
                loss = (
                    loss_fn(logits, label)
                    if loss_fn
                    else nn.CrossEntropyLoss()(logits, label)
                )

                if task_type != TaskType.REGRESSION:
                    probabilities[batch_idx] = Y_prob.detach().cpu().numpy()
                else:
                    probabilities[batch_idx] = (
                        Y_hat.detach().cpu().numpy()
                    )  # For regression, store predictions
                    predictions[batch_idx] = Y_hat.detach().cpu().numpy()

                labels[batch_idx] = label.item()

                val_loss += loss.item()
                error = calculate_error(Y_hat, label)
                val_error += error

        val_error /= len(loader)
        val_loss /= len(loader)

        # Calculate AUC for classification tasks
        auc = None
        if task_type != TaskType.REGRESSION:
            auc = MetricsCalculator.calculate_classification_metrics(
                n_classes, labels, probabilities
            )["auc"]
        else:
            # For regression, calculate regression metrics
            regression_metrics = MetricsCalculator.calculate_regression_metrics(
                labels, predictions
            )
            for metric_name, metric_value in regression_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)

        # Log validation metrics
        MetricsCalculator.log_validation_metrics(
            epoch,
            val_loss,
            val_error,
            auc,
            metrics_logger,
            n_classes,
            writer,
            task_type,
        )

        print(
            f"\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}"
            + (f", auc: {auc:.4f}" if auc is not None else "")
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
        # Removed alpha parameter - instance loss is handled internally by the model

    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        task_type: TaskType,
        n_classes: int,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        """Train one epoch for CLAM models"""
        model.train()
        metrics_logger = MetricsLoggerFactory.create_logger(task_type, n_classes)

        train_loss = 0.0
        train_error = 0.0
        train_inst_loss = 0.0
        inst_count = 0

        print(f"\nTraining Epoch: {epoch}")

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits, Y_prob, Y_hat, _, results_dict = model(
                data, label=label, instance_eval=True
            )

            # Calculate bag loss based on task type
            if task_type == TaskType.REGRESSION:
                # For regression, use the scaled values (Y_prob) for loss calculation
                bag_loss = loss_fn(Y_prob, label)
            else:
                # For classification, use logits
                bag_loss = loss_fn(logits, label)

            # Extract instance loss (constraint loss is no longer returned)
            instance_loss = results_dict.get("instance_loss", 0.0)

            # Ensure Y_hat is available
            if Y_hat is None:
                Y_hat = Y_prob

            # Calculate total loss - instance loss is already included in the model's forward pass
            total_loss = bag_loss + instance_loss

            # Update metrics
            train_loss += total_loss.item()

            # Convert losses to scalars safely
            if isinstance(instance_loss, torch.Tensor):
                instance_loss_value = instance_loss.item()
            else:
                instance_loss_value = float(instance_loss)

            train_inst_loss += instance_loss_value
            inst_count += 1

            error = calculate_error(Y_hat, label, task_type)
            train_error += error

            # Log batch metrics
            metrics_logger.log_batch(Y_hat, label)

            # Backward pass
            total_loss.backward()
            optimizer.step()

        # Calculate epoch averages
        train_loss /= len(loader)
        train_error /= len(loader)
        train_inst_loss = train_inst_loss / inst_count if inst_count > 0 else 0.0

        # Log using MetricsCalculator
        MetricsCalculator.log_training_metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_error=train_error,
            train_inst_loss=train_inst_loss,
            metrics_logger=metrics_logger,
            n_classes=n_classes,
            writer=writer,
        )

    def validate(
        self,
        cur: int,
        epoch: int,
        model: nn.Module,
        loader: Any,
        task_type: TaskType,
        n_classes: int,
        early_stopping: Optional[EarlyStopping] = None,
        writer: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        results_dir: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Validate CLAM model"""
        model.eval()
        metrics_logger = MetricsLoggerFactory.create_logger(task_type, n_classes)

        val_loss = 0.0
        val_error = 0.0
        val_inst_loss = 0.0
        inst_count = 0

        # For metrics calculation
        all_probs = []
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                data, label = data.to(device), label.to(device)

                # Forward pass
                logits, Y_prob, Y_hat, _, results_dict = model(
                    data, label=label, instance_eval=True
                )

                # Calculate bag loss based on task type
                if task_type == TaskType.REGRESSION:
                    bag_loss = loss_fn(Y_prob, label)
                else:
                    bag_loss = loss_fn(logits, label)

                # Extract instance loss (constraint loss is no longer returned)
                instance_loss = results_dict.get("instance_loss", 0.0)

                # Ensure Y_hat is available
                if Y_hat is None:
                    Y_hat = Y_prob

                # Calculate total loss - instance loss is already included
                total_loss = bag_loss + instance_loss

                # Convert losses to scalars safely
                if isinstance(instance_loss, torch.Tensor):
                    instance_loss_value = instance_loss.item()
                else:
                    instance_loss_value = float(instance_loss)

                val_loss += total_loss.item()
                val_inst_loss += instance_loss_value
                inst_count += 1

                error = calculate_error(Y_hat, label, task_type)
                val_error += error

                # Store probabilities and labels for metrics calculation
                if task_type != TaskType.REGRESSION:
                    all_probs.append(Y_prob.detach().cpu().numpy())
                else:
                    all_predictions.append(Y_hat.detach().cpu().numpy())

                all_labels.append(label.detach().cpu().numpy())

                # Log batch metrics
                metrics_logger.log_batch(Y_hat, label)

        # Calculate metrics
        val_loss /= len(loader)
        val_error /= len(loader)
        val_inst_loss = val_inst_loss / inst_count if inst_count > 0 else 0.0

        # Prepare data for metrics calculation
        all_labels_np = np.concatenate(all_labels)

        # Calculate task-specific metrics
        auc = None
        regression_metrics = None

        if task_type != TaskType.REGRESSION:
            # Classification metrics
            all_probs_np = np.concatenate(all_probs)
            if n_classes == 2:
                auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
            else:
                auc = roc_auc_score(
                    all_labels_np, all_probs_np, multi_class="ovr", average="macro"
                )
        else:
            # Regression metrics
            all_predictions_np = np.concatenate(all_predictions)
            regression_metrics = MetricsCalculator.calculate_regression_metrics(
                all_labels_np, all_predictions_np
            )

            # Log regression metrics
            for metric_name, metric_value in regression_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)
                if writer:
                    writer.add_scalar(f"val/{metric_name}", metric_value, epoch)

        # Print validation results
        print_str = (
            f"Val Loss: {val_loss:.4f}, Val Instance Loss: {val_inst_loss:.4f}, "
            f"Val Error: {val_error:.4f}"
        )
        if auc is not None:
            print_str += f", Val AUC: {auc:.4f}"
        print(print_str)

        # Log validation metrics using MetricsCalculator
        MetricsCalculator.log_validation_metrics(
            epoch=epoch,
            val_loss=val_loss,
            val_error=val_error,
            auc=auc,
            metrics_logger=metrics_logger,
            n_classes=n_classes,
            writer=writer,
            task_type=task_type,
        )

        # Early stopping
        if early_stopping and results_dir:
            # For regression, use val_loss for early stopping
            # Alternatively, you could use a specific regression metric
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
        if (
            config.model_type in ["clam_sb", "clam_mb", "clam_sbr", "clam_mbr"]
            and not config.no_inst_cluster
        ):
            return CLAMTrainingStrategy(bag_weight=config.bag_weight)
        else:
            return StandardMILTrainingStrategy()


# ==================== UPDATED MODEL EVALUATOR ====================
class ModelEvaluator:
    """Handles model evaluation and summary statistics"""

    @staticmethod
    def summary(
        model: nn.Module,
        loader: Any,
        n_classes: int,
        task_type: TaskType = TaskType.BINARY,
    ) -> Tuple[Dict[str, Any], float, float, BaseMetricsLogger]:
        """Generate summary statistics for model performance"""
        metrics_logger = MetricsLoggerFactory.create_logger(task_type, n_classes)
        model.eval()
        test_error = 0.0

        # Initialize arrays based on task type
        if task_type != TaskType.REGRESSION:
            all_probs = np.zeros((len(loader), n_classes))
            all_labels = np.zeros(len(loader))
            all_predictions = np.zeros(len(loader))
        else:
            # For regression with 2D output [primary, secondary]
            all_probs = np.zeros((len(loader), 2))  # Store both primary and secondary
            all_labels = np.zeros((len(loader), 2))  # Store both labels if available
            all_predictions = np.zeros((len(loader), 2))

        slide_ids = loader.dataset.slide_data["slide_id"]
        patient_results = {}

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            with torch.inference_mode():
                logits, Y_prob, Y_hat, _, _ = model(data)

            metrics_logger.log(Y_hat, label)

            if task_type != TaskType.REGRESSION:
                probs = Y_prob.cpu().numpy()
                all_probs[batch_idx] = probs
                all_predictions[batch_idx] = Y_hat.cpu().numpy()

                # Handle label conversion safely for classification
                label_np = label.cpu().numpy()
                if label_np.size == 1:
                    all_labels[batch_idx] = label_np.item()
                else:
                    # For multi-element labels, take the first one or use appropriate indexing
                    all_labels[batch_idx] = label_np.flatten()[0]

                patient_results[slide_id] = {
                    "slide_id": np.array(slide_id),
                    "prob": probs,
                    "label": all_labels[batch_idx],
                }
            else:
                # Handle regression with 2D output
                preds = Y_hat.cpu().numpy()

                # Ensure preds is 2D [primary, secondary]
                if preds.ndim == 1:
                    preds = preds.reshape(1, -1)

                # Store predictions
                all_probs[batch_idx] = preds[0]  # Store as 1D array of length 2
                all_predictions[batch_idx] = preds[0]

                # Handle label formatting for regression - FIXED for 2D arrays
                label_np = label.cpu().numpy()

                # Handle different label formats for regression
                if label_np.ndim == 0:
                    # Scalar label
                    all_labels[batch_idx] = [label_np.item(), label_np.item()]
                elif label_np.size == 1:
                    # 1D array with single element
                    all_labels[batch_idx] = [label_np.item(), label_np.item()]
                else:
                    # Multi-element array - extract first two elements safely
                    label_flat = label_np.flatten()
                    if len(label_flat) >= 2:
                        all_labels[batch_idx] = [label_flat[0], label_flat[1]]
                    else:
                        # Fallback: duplicate the first element
                        all_labels[batch_idx] = [label_flat[0], label_flat[0]]

                patient_results[slide_id] = {
                    "slide_id": np.array(slide_id),
                    "prob": preds[0],  # Store as 1D array [primary, secondary]
                    "label": all_labels[batch_idx],
                }

            error = calculate_error(Y_hat, label, task_type)
            test_error += error

        test_error /= len(loader)

        # Calculate metrics based on task type
        auc = None
        if task_type != TaskType.REGRESSION:
            auc = MetricsCalculator.calculate_classification_metrics(
                n_classes, all_labels, all_probs
            )["auc"]
        else:
            # For regression, calculate metrics for primary and secondary separately
            regression_metrics = MetricsCalculator.calculate_regression_metrics(
                all_labels, all_predictions
            )
            # Log regression metrics
            for metric_name, metric_value in regression_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

        return patient_results, test_error, auc, metrics_logger


# ==================== UPDATED MODEL MANAGER ====================
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
        if config.task == TaskType.BINARY and config.model_type == "clam_sb":
            model = CLAM_SB(
                size_arg=config.model_size,
                dropout=config.drop_out,
                n_classes=config.n_classes,
                subtyping=config.subtyping,
                embed_dim=config.embed_dim,
            )
        elif config.task == TaskType.MULTICLASS and config.model_type == "clam_mb":
            model = CLAM_MB(
                size_arg=config.model_size,
                dropout=config.drop_out,
                n_classes=config.n_classes,
                subtyping=config.subtyping,
                embed_dim=config.embed_dim,
            )
        elif config.task == TaskType.REGRESSION and (
            config.model_type == "clam_sb" or config.model_type == "clam_sbr"
        ):
            model = CLAM_SB_Regression(
                size_arg=config.model_size,
                dropout=config.drop_out,
                n_classes=config.n_classes,
                subtyping=config.subtyping,
                embed_dim=config.embed_dim,
            )
        elif config.task == TaskType.REGRESSION and (
            config.model_type == "clam_mb" or config.model_type == "clam_mbr"
        ):
            model = CLAM_MB_Regression(
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
        all_val_metrics: List[float],
        all_test_metrics: List[float],
        folds: np.ndarray,
        dataset: Generic_MIL_Dataset,
    ) -> Optional[torch.nn.Module]:
        """Identify the best model from cross-validation and log it to MLflow"""
        if not config.register_best_model:
            print("Model registration is disabled. Skipping...")
            return None

        # Find the best fold based on task type
        if config.task == TaskType.REGRESSION:
            # For regression, lower MAE is better
            best_fold_idx = np.argmin(all_val_metrics)
            best_val_metric = all_val_metrics[best_fold_idx]
            best_test_metric = all_test_metrics[best_fold_idx]
            metric_name = "MAE"
        else:
            # For classification, higher AUC is better
            best_fold_idx = np.argmax(all_val_metrics)
            best_val_metric = all_val_metrics[best_fold_idx]
            best_test_metric = all_test_metrics[best_fold_idx]
            metric_name = "AUC"

        best_fold = folds[best_fold_idx]

        print(
            f"Best model from fold {best_fold} with validation {metric_name}: {best_val_metric:.4f}"
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
            print(f"   - Best validation {metric_name}: {best_val_metric:.4f}")
            print(f"   - Best test {metric_name}: {best_test_metric:.4f}")
            print(f"   - From fold: {best_fold}")

            # Log additional metrics
            mlflow.set_tag("best_model_fold", str(best_fold))
            mlflow.log_metric(f"best_val_{metric_name.lower()}", best_val_metric)
            mlflow.log_metric(f"best_test_{metric_name.lower()}", best_test_metric)

            return best_model

        except Exception as e:
            print(f"Error logging model to MLflow: {e}")
            return None


# ==================== UPDATED SINGLE FOLD TRAINER ====================
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
                    "task_type": (
                        self.config.task.value
                        if hasattr(self.config.task, "value")
                        else str(self.config.task)
                    ),
                }
            )

            print(f"\nTraining Fold {self.fold_id}/{self.folds}!")

            # Setup data
            train_split, val_split, test_split = self.setup_data_splits(datasets)

            # Setup components
            loss_fn = ModelFactory.create_loss_function(
                self.config.bag_loss, self.config.n_classes, self.config.task
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
                self.config.task,
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
                self.config.task,
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

        # Check task type
        is_regression = self.config.task == TaskType.REGRESSION

        # --- Validation evaluation ---
        val_results, val_error, val_auc, val_metrics_logger = ModelEvaluator.summary(
            model, val_loader, self.config.n_classes, self.config.task
        )

        # --- Test evaluation ---
        test_results, test_error, test_auc, test_metrics_logger = (
            ModelEvaluator.summary(
                model, test_loader, self.config.n_classes, self.config.task
            )
        )

        # Get comprehensive metrics
        val_metrics = val_metrics_logger.get_all_metrics()
        test_metrics = test_metrics_logger.get_all_metrics()

        # Select the primary metric to print based on task
        if is_regression:
            val_mae = val_metrics.get("mae", val_error)
            test_mae = test_metrics.get("mae", test_error)

            val_metric_str = f"MAE: {val_mae:.4f}, Error (Loss): {val_error:.4f}"
            test_metric_str = f"MAE: {test_mae:.4f}, Error (Loss): {test_error:.4f}"

            # Log additional regression metrics
            val_mse = val_metrics.get("mse", 0.0)
            val_r2 = val_metrics.get("r2", 0.0)
            test_mse = test_metrics.get("mse", 0.0)
            test_r2 = test_metrics.get("r2", 0.0)

            print(f"Val {val_metric_str}, MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
            print(f"Test {test_metric_str}, MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        else:
            val_metric_str = f"Error: {val_error:.4f}" + (
                f", ROC AUC: {val_auc:.4f}" if val_auc is not None else ""
            )
            test_metric_str = f"Error: {test_error:.4f}" + (
                f", ROC AUC: {test_auc:.4f}" if test_auc is not None else ""
            )
            print(f"Val {val_metric_str}")
            print(f"Test {test_metric_str}")

        # --- Log final metrics ---
        MetricsCalculator.log_final_metrics(
            val_error,
            val_auc,
            test_error,
            test_auc,
            test_metrics_logger,
            self.config.n_classes,
            self.config.task,
        )

        # --- Cleanup ---
        if self.writer:
            self.writer.close()

        # --- Return Results ---
        if is_regression:
            return {
                "results_dict": test_results,
                "test_mae": test_mae,
                "val_mae": val_mae,
                "test_mse": test_mse,
                "val_mse": val_mse,
                "test_r2": test_r2,
                "val_r2": val_r2,
                "test_auc": 0.0,  # Placeholder for compatibility
                "val_auc": 0.0,  # Placeholder for compatibility
                "test_acc": test_mae,  # Repurpose acc field for MAE
                "val_acc": val_mae,  # Repurpose acc field for MAE
            }
        else:
            return {
                "results_dict": test_results,
                "test_auc": test_auc if test_auc is not None else 0.0,
                "val_auc": val_auc if val_auc is not None else 0.0,
                "test_acc": 1.0 - test_error,  # Convert error to accuracy
                "val_acc": 1.0 - val_error,  # Convert error to accuracy
                "test_mae": 0.0,  # Placeholder for compatibility
                "val_mae": 0.0,  # Placeholder for compatibility
                "test_mse": 0.0,  # Placeholder for compatibility
                "val_mse": 0.0,  # Placeholder for compatibility
                "test_r2": 0.0,  # Placeholder for compatibility
                "val_r2": 0.0,  # Placeholder for compatibility
            }


# ==================== UPDATED CROSS-VALIDATION TRAINER ====================
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

            # Register best model with appropriate metrics
            if self.config.task == TaskType.REGRESSION:
                val_metrics = results["val_mae"]
                test_metrics = results["test_mae"]
            else:
                val_metrics = results["val_auc"]
                test_metrics = results["test_auc"]

            best_model = ModelManager.log_best_model_to_mlflow(
                self.config,
                val_metrics,
                test_metrics,
                results["folds"],
                self.dataset,
            )

            return TrainingResults(
                test_auc=(
                    results["test_auc"]
                    if self.config.task != TaskType.REGRESSION
                    else None
                ),
                val_auc=(
                    results["val_auc"]
                    if self.config.task != TaskType.REGRESSION
                    else None
                ),
                test_acc=(
                    results["test_acc"]
                    if self.config.task != TaskType.REGRESSION
                    else None
                ),
                val_acc=(
                    results["val_acc"]
                    if self.config.task != TaskType.REGRESSION
                    else None
                ),
                test_mae=(
                    results["test_mae"]
                    if self.config.task == TaskType.REGRESSION
                    else None
                ),
                val_mae=(
                    results["val_mae"]
                    if self.config.task == TaskType.REGRESSION
                    else None
                ),
                test_mse=(
                    results["test_mse"]
                    if self.config.task == TaskType.REGRESSION
                    else None
                ),
                val_mse=(
                    results["val_mse"]
                    if self.config.task == TaskType.REGRESSION
                    else None
                ),
                test_r2=(
                    results["test_r2"]
                    if self.config.task == TaskType.REGRESSION
                    else None
                ),
                val_r2=(
                    results["val_r2"]
                    if self.config.task == TaskType.REGRESSION
                    else None
                ),
                final_df=results["final_df"],
                best_model=best_model,
                mlflow_run_id=run.info.run_id,
            )

    def _train_all_folds(self) -> Dict[str, Any]:
        """Train all folds and collect results"""
        # Initialize lists for metrics based on task type
        if self.config.task == TaskType.REGRESSION:
            all_test_mae: List[float] = []
            all_val_mae: List[float] = []
            all_test_mse: List[float] = []
            all_val_mse: List[float] = []
            all_test_r2: List[float] = []
            all_val_r2: List[float] = []
            all_test_auc: List[float] = []  # Placeholder
            all_val_auc: List[float] = []  # Placeholder
            all_test_acc: List[float] = []  # Repurposed as MAE
            all_val_acc: List[float] = []  # Repurposed as MAE
        else:
            all_test_auc: List[float] = []
            all_val_auc: List[float] = []
            all_test_acc: List[float] = []
            all_val_acc: List[float] = []
            all_test_mae: List[float] = []  # Placeholder
            all_val_mae: List[float] = []  # Placeholder
            all_test_mse: List[float] = []  # Placeholder
            all_val_mse: List[float] = []  # Placeholder
            all_test_r2: List[float] = []  # Placeholder
            all_val_r2: List[float] = []  # Placeholder

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

            # Collect results based on task type
            if self.config.task == TaskType.REGRESSION:
                all_test_mae.append(fold_results["test_mae"])
                all_val_mae.append(fold_results["val_mae"])
                all_test_mse.append(fold_results["test_mse"])
                all_val_mse.append(fold_results["val_mse"])
                all_test_r2.append(fold_results["test_r2"])
                all_val_r2.append(fold_results["val_r2"])
                all_test_acc.append(fold_results["test_acc"])  # MAE
                all_val_acc.append(fold_results["val_acc"])  # MAE
                all_test_auc.append(0.0)  # Placeholder
                all_val_auc.append(0.0)  # Placeholder
            else:
                all_test_auc.append(fold_results["test_auc"])
                all_val_auc.append(fold_results["val_auc"])
                all_test_acc.append(fold_results["test_acc"])
                all_val_acc.append(fold_results["val_acc"])
                all_test_mae.append(0.0)  # Placeholder
                all_val_mae.append(0.0)  # Placeholder
                all_test_mse.append(0.0)  # Placeholder
                all_val_mse.append(0.0)  # Placeholder
                all_test_r2.append(0.0)  # Placeholder
                all_val_r2.append(0.0)  # Placeholder

            # Save results for this fold
            filename = os.path.join(
                self.config.results_dir, f"split_{fold_idx}_results.pkl"
            )
            save_pkl(filename, fold_results["results_dict"])

        # Prepare results dictionary
        if self.config.task == TaskType.REGRESSION:
            return {
                "folds": folds,
                "test_mae": all_test_mae,
                "val_mae": all_val_mae,
                "test_mse": all_test_mse,
                "val_mse": all_val_mse,
                "test_r2": all_test_r2,
                "val_r2": all_val_r2,
                "test_auc": all_test_auc,
                "val_auc": all_val_auc,
                "test_acc": all_test_acc,
                "val_acc": all_val_acc,
                "final_df": self._create_summary_dataframe_regression(
                    folds,
                    all_test_mae,
                    all_val_mae,
                    all_test_mse,
                    all_val_mse,
                    all_test_r2,
                    all_val_r2,
                ),
            }
        else:
            return {
                "folds": folds,
                "test_auc": all_test_auc,
                "val_auc": all_val_auc,
                "test_acc": all_test_acc,
                "val_acc": all_val_acc,
                "test_mae": all_test_mae,
                "val_mae": all_val_mae,
                "test_mse": all_test_mse,
                "val_mse": all_val_mse,
                "test_r2": all_test_r2,
                "val_r2": all_val_r2,
                "final_df": self._create_summary_dataframe_classification(
                    folds, all_test_auc, all_val_auc, all_test_acc, all_val_acc
                ),
            }

    def _create_summary_dataframe_classification(
        self,
        folds: np.ndarray,
        all_test_auc: List[float],
        all_val_auc: List[float],
        all_test_acc: List[float],
        all_val_acc: List[float],
    ) -> pd.DataFrame:
        """Create and save summary dataframe for classification tasks"""
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

    def _create_summary_dataframe_regression(
        self,
        folds: np.ndarray,
        all_test_mae: List[float],
        all_val_mae: List[float],
        all_test_mse: List[float],
        all_val_mse: List[float],
        all_test_r2: List[float],
        all_val_r2: List[float],
    ) -> pd.DataFrame:
        """Create and save summary dataframe for regression tasks"""
        final_df = pd.DataFrame(
            {
                "folds": folds,
                "test_mae": all_test_mae,
                "val_mae": all_val_mae,
                "test_mse": all_test_mse,
                "val_mse": all_val_mse,
                "test_r2": all_test_r2,
                "val_r2": all_val_r2,
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

        # Log metrics based on task type
        if self.config.task == TaskType.REGRESSION:
            self._log_regression_metrics(results)
        else:
            self._log_classification_metrics(results)

    def _log_classification_metrics(self, results: Dict[str, Any]) -> None:
        """Log classification metrics to MLflow"""
        mlflow.log_metric("CV_Test_AUC_Mean", float(np.mean(results["test_auc"])))
        mlflow.log_metric("CV_Test_AUC_Std", float(np.std(results["test_auc"])))
        mlflow.log_metric("CV_Test_Accuracy_Mean", float(np.mean(results["test_acc"])))
        mlflow.log_metric("CV_Test_Accuracy_Std", float(np.std(results["test_acc"])))
        mlflow.log_metric("CV_Val_AUC_Mean", float(np.mean(results["val_auc"])))
        mlflow.log_metric("CV_Val_AUC_Std", float(np.std(results["val_auc"])))

        print("\n################# Final Classification Results ###################")
        print(
            f"CV Test AUC: {np.mean(results['test_auc']):.4f} ± {np.std(results['test_auc']):.4f}"
        )
        print(
            f"CV Test Accuracy: {np.mean(results['test_acc']):.4f} ± {np.std(results['test_acc']):.4f}"
        )
        print(
            f"CV Val AUC: {np.mean(results['val_auc']):.4f} ± {np.std(results['val_auc']):.4f}"
        )

    def _log_regression_metrics(self, results: Dict[str, Any]) -> None:
        """Log regression metrics to MLflow"""
        # MAE Metrics
        mlflow.log_metric("CV_Test_MAE_Mean", float(np.mean(results["test_mae"])))
        mlflow.log_metric("CV_Test_MAE_Std", float(np.std(results["test_mae"])))
        mlflow.log_metric("CV_Val_MAE_Mean", float(np.mean(results["val_mae"])))
        mlflow.log_metric("CV_Val_MAE_Std", float(np.std(results["val_mae"])))

        # MSE Metrics
        mlflow.log_metric("CV_Test_MSE_Mean", float(np.mean(results["test_mse"])))
        mlflow.log_metric("CV_Test_MSE_Std", float(np.std(results["test_mse"])))
        mlflow.log_metric("CV_Val_MSE_Mean", float(np.mean(results["val_mse"])))
        mlflow.log_metric("CV_Val_MSE_Std", float(np.std(results["val_mse"])))

        # R² Metrics
        mlflow.log_metric("CV_Test_R2_Mean", float(np.mean(results["test_r2"])))
        mlflow.log_metric("CV_Test_R2_Std", float(np.std(results["test_r2"])))
        mlflow.log_metric("CV_Val_R2_Mean", float(np.mean(results["val_r2"])))
        mlflow.log_metric("CV_Val_R2_Std", float(np.std(results["val_r2"])))

        print("\n################# Final Regression Results ###################")
        print(
            f"CV Test MAE: {np.mean(results['test_mae']):.4f} ± {np.std(results['test_mae']):.4f}"
        )
        print(
            f"CV Test MSE: {np.mean(results['test_mse']):.4f} ± {np.std(results['test_mse']):.4f}"
        )
        print(
            f"CV Test R²: {np.mean(results['test_r2']):.4f} ± {np.std(results['test_r2']):.4f}"
        )
        print(
            f"CV Val MAE: {np.mean(results['val_mae']):.4f} ± {np.std(results['val_mae']):.4f}"
        )


# ==================== PUBLIC INTERFACE ====================
def run_training(
    config: MILTrainingConfig, dataset: Generic_MIL_Dataset
) -> TrainingResults:
    """Main training function with cross-validation"""
    trainer = CrossValidationTrainer(config, dataset)
    return trainer.run_cross_validation()
