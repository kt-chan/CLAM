import numpy as np
import torch
import os
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
import json

import mlflow
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import label_binarize

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from utils.utils import *
from utils.train_utils import (
    ModelManager,
    MetricsLoggerFactory,
    BaseMetricsLogger,
    MetricsCalculator,
    ModelEvaluator,
    TaskType,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalConfig:
    """Configuration class for evaluation with comprehensive type hints."""

    def __init__(self, **kwargs) -> None:
        # Data parameters
        self.data_root_dir = kwargs.get("data_root_dir")
        self.data_set_name = kwargs.get("data_set_name")

        # Results and paths
        self.results_dir = kwargs.get("results_dir", "./results")
        self.save_exp_code = kwargs.get("save_exp_code")
        self.models_exp_code = kwargs.get("models_exp_code")
        self.splits_dir = kwargs.get("splits_dir")

        # Model parameters
        self.model_size = kwargs.get("model_size", "small")
        self.model_type = kwargs.get("model_type", "clam_sb")
        self.drop_out = kwargs.get("drop_out", 0.25)
        self.embed_dim = kwargs.get("embed_dim", 1024)

        # Evaluation parameters
        self.k = kwargs.get("k", 10)
        self.k_start = kwargs.get("k_start", -1)
        self.k_end = kwargs.get("k_end", -1)
        self.fold = kwargs.get("fold", -1)
        self.micro_average = kwargs.get("micro_average", False)
        self.split = kwargs.get("split", "test")
        self.task = kwargs.get("task")
        self.subtyping = kwargs.get("subtyping", False)

        # MLflow model registration parameters
        self.registered_model_name = kwargs.get("registered_model_name")
        self.register_best_model = kwargs.get("register_best_model", True)
        self.detailed_metrics = kwargs.get("detailed_metrics", True)
        self.seed = kwargs.get("seed", 1)
        # Setup derived attributes
        self._setup_derived_attributes()

    def _setup_derived_attributes(self) -> None:
        """Setup derived attributes and paths."""
        # Set n_classes and task_type based on task
        self._setup_task_configuration()

        # Setup directories
        self._setup_directories()

        # Setup fold ranges
        self._setup_fold_configuration()

    def _setup_task_configuration(self) -> None:
        """Configure task-specific parameters."""
        if self.task == "task_1_tumor_vs_normal":
            self.task = TaskType.BINARY
            self.n_classes = 2
        elif self.task == "task_2_tumor_subtyping":
            self.task = TaskType.MULTICLASS
            self.n_classes = 3
        elif self.task == "task_3_tumor_count":
            self.task = TaskType.REGRESSION
            self.n_classes = -1
        else:
            self.task = None
            self.n_classes = None

    def _setup_directories(self) -> None:
        """Setup and validate directory paths."""
        self.save_dir = os.path.join("./eval_results", f"EVAL_{self.save_exp_code}")
        self.models_dir = os.path.join(self.results_dir, str(self.models_exp_code))

        # Setup default registered model name
        if self.registered_model_name is None and self.models_exp_code:
            self.registered_model_name = f"{self.model_type}_{self.models_exp_code}"

        os.makedirs(self.save_dir, exist_ok=True)

        # Setup splits directory
        if self.splits_dir is None:
            self.splits_dir = self.models_dir

        # Validate directories
        self._validate_directories()

    def _validate_directories(self) -> None:
        """Validate that required directories exist."""
        if not os.path.isdir(self.models_dir):
            raise FileNotFoundError(
                f"Models directory {self.models_dir} does not exist"
            )
        if not os.path.isdir(self.splits_dir):
            raise FileNotFoundError(
                f"Splits directory {self.splits_dir} does not exist"
            )

    def _setup_fold_configuration(self) -> None:
        """Setup fold ranges for evaluation."""
        self.start_fold = 0 if self.k_start == -1 else self.k_start
        self.end_fold = self.k if self.k_end == -1 else self.k_end

        # Setup folds to evaluate
        if self.fold == -1:
            self.folds = range(self.start_fold, self.end_fold)
        else:
            self.folds = range(self.fold, self.fold + 1)

    def get_settings(self) -> Dict[str, Any]:
        """Return settings dictionary for logging."""
        return {
            "task": self.task,
            "split": self.split,
            "save_dir": self.save_dir,
            "models_dir": self.models_dir,
            "models_exp_code": self.models_exp_code,
            "model_type": self.model_type,
            "drop_out": self.drop_out,
            "model_size": self.model_size,
            "k_folds_evaluated": list(self.folds),
            "detailed_metrics": self.detailed_metrics,
        }


class ModelLoader:
    """Handles model loading and initialization."""

    @staticmethod
    def load_model(
        config: EvalConfig, ckpt_path: str, device: str = "cuda"
    ) -> torch.nn.Module:
        """Initialize model from checkpoint with proper configuration."""
        print(f"Loading model from checkpoint: {ckpt_path}")

        try:
            # Use ModelManager to reconstruct the model (same approach as train_utils.py)
            model = ModelManager.reconstruct_clam_model(config, ckpt_path)
            model = model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model with ModelManager: {e}")
            print("Falling back to direct model loading...")
            return ModelLoader._load_model_directly(config, ckpt_path, device)

    @staticmethod
    def _load_model_directly(
        config: EvalConfig, ckpt_path: str, device: str
    ) -> torch.nn.Module:
        """Fallback method for direct model loading."""
        model_dict = {
            "dropout": config.drop_out,
            "n_classes": config.n_classes,
            "embed_dim": config.embed_dim,
        }

        if config.model_size is not None and config.model_type in [
            "clam_sb",
            "clam_mb",
        ]:
            model_dict.update({"size_arg": config.model_size})

        # Create model based on type and task
        model = ModelLoader._create_model_by_type(config, model_dict)

        # Load state dict
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_clean = ModelLoader._clean_state_dict(ckpt)

        model.load_state_dict(ckpt_clean, strict=True)
        model = model.to(device)
        model.eval()

        return model

    @staticmethod
    def _create_model_by_type(
        config: EvalConfig, model_dict: Dict[str, Any]
    ) -> torch.nn.Module:
        """Create model instance based on configuration."""
        if config.task == TaskType.REGRESSION:
            return ModelLoader._create_regression_model(config, model_dict)
        else:
            return ModelLoader._create_classification_model(config, model_dict)

    @staticmethod
    def _create_regression_model(
        config: EvalConfig, model_dict: Dict[str, Any]
    ) -> torch.nn.Module:
        """Create regression model based on configuration."""
        if config.model_type in ["clam_sb", "clam_sbr"]:
            from models.model_clam import CLAM_SB_Regression

            return CLAM_SB_Regression(**model_dict)
        elif config.model_type in ["clam_mb", "clam_mbr"]:
            from models.model_clam import CLAM_MB_Regression

            return CLAM_MB_Regression(**model_dict)
        else:
            from models.model_mil import MIL_fc

            return MIL_fc(**model_dict)

    @staticmethod
    def _create_classification_model(
        config: EvalConfig, model_dict: Dict[str, Any]
    ) -> torch.nn.Module:
        """Create classification model based on configuration."""
        if config.model_type == "clam_sb":
            from models.model_clam import CLAM_SB

            return CLAM_SB(**model_dict)
        elif config.model_type == "clam_mb":
            from models.model_clam import CLAM_MB

            return CLAM_MB(**model_dict)
        else:
            if config.n_classes > 2:
                from models.model_mil import MIL_fc_mc

                return MIL_fc_mc(**model_dict)
            else:
                from models.model_mil import MIL_fc

                return MIL_fc(**model_dict)

    @staticmethod
    def _clean_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
        """Clean state dictionary by removing unnecessary keys."""
        ckpt_clean = {}
        for key in ckpt.keys():
            if "instance_loss_fn" in key:
                continue
            ckpt_clean.update({key.replace(".module", ""): ckpt[key]})
        return ckpt_clean


class MetricsCalculator:
    """Handles comprehensive metrics calculation for different task types."""

    @staticmethod
    def calculate_detailed_metrics(
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        all_probs: np.ndarray,
        n_classes: int,
        task_type: TaskType = TaskType.BINARY,
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        metrics: Dict[str, Any] = {}

        try:
            if task_type == TaskType.REGRESSION:
                MetricsCalculator._calculate_regression_metrics(
                    metrics, all_labels, all_preds
                )
            else:
                MetricsCalculator._calculate_classification_metrics(
                    metrics, all_labels, all_preds, all_probs, n_classes, task_type
                )
        except Exception as e:
            print(f"Warning: Error calculating detailed metrics: {e}")

        return metrics

    @staticmethod
    def _calculate_regression_metrics(
        metrics: Dict[str, Any], all_labels: np.ndarray, all_preds: np.ndarray
    ) -> None:
        """Calculate regression-specific metrics with multi-output support."""
        try:
            # Handle different dimensionalities
            if all_preds.ndim == 1:
                # Single output regression
                metrics["mae"] = float(mean_absolute_error(all_labels, all_preds))
                metrics["mse"] = float(mean_squared_error(all_labels, all_preds))
                metrics["rmse"] = float(np.sqrt(metrics["mse"]))
                metrics["r2"] = float(r2_score(all_labels, all_preds))

            elif all_preds.ndim == 2:
                # Multi-output regression
                n_outputs = all_preds.shape[1]

                # Calculate overall metrics (flatten if needed)
                flat_preds = all_preds.reshape(-1)
                if all_labels.ndim == 2 and all_labels.shape[1] == n_outputs:
                    flat_labels = all_labels.reshape(-1)
                else:
                    flat_labels = all_labels

                metrics["mae"] = float(mean_absolute_error(flat_labels, flat_preds))
                metrics["mse"] = float(mean_squared_error(flat_labels, flat_preds))
                metrics["rmse"] = float(np.sqrt(metrics["mse"]))
                metrics["r2"] = float(r2_score(flat_labels, flat_preds))

                # Calculate per-output metrics
                for i in range(n_outputs):
                    if all_labels.ndim == 2 and all_labels.shape[1] > i:
                        output_labels = all_labels[:, i]
                    else:
                        output_labels = all_labels

                    metrics[f"mae_output_{i}"] = float(
                        mean_absolute_error(output_labels, all_preds[:, i])
                    )
                    metrics[f"mse_output_{i}"] = float(
                        mean_squared_error(output_labels, all_preds[:, i])
                    )
                    metrics[f"rmse_output_{i}"] = float(
                        np.sqrt(metrics[f"mse_output_{i}"])
                    )

            else:
                # Higher dimensional - flatten
                flat_preds = all_preds.reshape(-1)
                flat_labels = all_labels.reshape(-1)
                metrics["mae"] = float(mean_absolute_error(flat_labels, flat_preds))
                metrics["mse"] = float(mean_squared_error(flat_labels, flat_preds))
                metrics["rmse"] = float(np.sqrt(metrics["mse"]))
                metrics["r2"] = float(r2_score(flat_labels, flat_preds))

        except Exception as e:
            print(f"Error calculating regression metrics: {e}")
            metrics["mae"] = 0.0
            metrics["mse"] = 0.0
            metrics["rmse"] = 0.0
            metrics["r2"] = 0.0

    @staticmethod
    def _calculate_classification_metrics(
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        all_probs: np.ndarray,
        n_classes: int,
        task_type: TaskType,
    ) -> None:
        """Calculate classification-specific metrics."""
        metrics["accuracy"] = float(np.mean(all_preds == all_labels))
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(all_labels, all_preds)
        )

        # Basic classification metrics
        MetricsCalculator._calculate_basic_classification_metrics(
            metrics, all_labels, all_preds, n_classes, task_type
        )

        # Per-class metrics
        MetricsCalculator._calculate_per_class_metrics(
            metrics, all_labels, all_preds, n_classes
        )

        # AUC and confidence metrics
        if all_probs is not None and all_probs.size > 0:
            MetricsCalculator._calculate_auc_metrics(
                metrics, all_labels, all_probs, n_classes, task_type
            )
            MetricsCalculator._calculate_confidence_metrics(
                metrics, all_labels, all_preds, all_probs
            )

    @staticmethod
    def _calculate_basic_classification_metrics(
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        n_classes: int,
        task_type: TaskType,
    ) -> None:
        """Calculate basic classification metrics."""
        average_methods = ["macro", "weighted"]

        for average in average_methods:
            metrics.update(
                {
                    f"precision_{average}": float(
                        precision_score(
                            all_labels, all_preds, average=average, zero_division=0
                        )
                    ),
                    f"recall_{average}": float(
                        recall_score(
                            all_labels, all_preds, average=average, zero_division=0
                        )
                    ),
                    f"f1_{average}": float(
                        f1_score(
                            all_labels, all_preds, average=average, zero_division=0
                        )
                    ),
                }
            )

        # For binary classification, also calculate binary metrics
        if task_type == TaskType.BINARY and n_classes == 2:
            metrics.update(
                {
                    "precision_binary": float(
                        precision_score(all_labels, all_preds, zero_division=0)
                    ),
                    "recall_binary": float(
                        recall_score(all_labels, all_preds, zero_division=0)
                    ),
                    "f1_binary": float(
                        f1_score(all_labels, all_preds, zero_division=0)
                    ),
                }
            )

    @staticmethod
    def _calculate_per_class_metrics(
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        n_classes: int,
    ) -> None:
        """Calculate per-class metrics for classification."""
        for class_idx in range(n_classes):
            class_mask = all_labels == class_idx
            if np.sum(class_mask) > 0:
                class_accuracy = float(np.mean(all_preds[class_mask] == class_idx))
                metrics[f"class_{class_idx}_accuracy"] = class_accuracy
                metrics[f"class_{class_idx}_support"] = int(np.sum(class_mask))

    @staticmethod
    def _calculate_auc_metrics(
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_probs: np.ndarray,
        n_classes: int,
        task_type: TaskType,
    ) -> None:
        """Calculate AUC metrics for classification."""
        try:
            if task_type == TaskType.BINARY:
                # Binary AUC
                if all_probs.shape[1] == 1:
                    # Single probability column
                    positive_probs = all_probs[:, 0]
                else:
                    # Two probability columns - use positive class (index 1)
                    positive_probs = (
                        all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs[:, 0]
                    )

                if len(np.unique(all_labels)) > 1:
                    metrics["auc"] = float(roc_auc_score(all_labels, positive_probs))

            elif task_type == TaskType.MULTICLASS and n_classes > 2:
                # Multi-class AUC (One-vs-Rest)
                y_true_bin = label_binarize(all_labels, classes=range(n_classes))

                # Calculate AUC for each class
                auc_scores = []
                for i in range(n_classes):
                    if len(np.unique(y_true_bin[:, i])) > 1:
                        class_auc = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
                        auc_scores.append(class_auc)
                        metrics[f"auc_class_{i}"] = float(class_auc)

                if auc_scores:
                    metrics["auc_macro"] = float(np.mean(auc_scores))

        except Exception as e:
            print(f"Could not calculate AUC metrics: {e}")

    @staticmethod
    def _calculate_confidence_metrics(
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        all_probs: np.ndarray,
    ) -> None:
        """Calculate confidence-based metrics."""
        try:
            max_probs = np.max(all_probs, axis=1)
            metrics["mean_confidence"] = float(np.mean(max_probs))
            metrics["confidence_std"] = float(np.std(max_probs))

            # Calibration metrics
            correct_predictions = all_preds == all_labels
            if np.sum(correct_predictions) > 0:
                metrics["avg_confidence_correct"] = float(
                    np.mean(max_probs[correct_predictions])
                )
            if np.sum(~correct_predictions) > 0:
                metrics["avg_confidence_incorrect"] = float(
                    np.mean(max_probs[~correct_predictions])
                )

        except Exception as e:
            print(f"Could not calculate confidence metrics: {e}")


class ArtifactGenerator:
    """Handles generation of visualization artifacts and detailed metrics."""

    @staticmethod
    def create_detailed_metrics_artifacts(
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        all_probs: np.ndarray,
        config: EvalConfig,
        fold: int,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Create detailed metrics and visualization artifacts."""
        detailed_metrics = {}
        artifacts = {}

        try:
            # Calculate comprehensive metrics
            detailed_metrics = MetricsCalculator.calculate_detailed_metrics(
                all_labels, all_preds, all_probs, config.n_classes, config.task
            )

            # Create visualization artifacts
            if config.task != TaskType.REGRESSION:
                ArtifactGenerator._create_classification_artifacts(
                    artifacts, all_labels, all_preds, all_probs, config, fold
                )
            else:
                ArtifactGenerator._create_regression_artifacts(
                    artifacts, all_labels, all_preds, config, fold
                )

        except Exception as e:
            print(f"Warning: Could not create detailed artifacts: {e}")

        return detailed_metrics, artifacts

    @staticmethod
    def _create_classification_artifacts(
        artifacts: Dict[str, str],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        all_probs: np.ndarray,
        config: EvalConfig,
        fold: int,
    ) -> None:
        """Create classification-specific visualization artifacts."""
        try:
            ArtifactGenerator._create_confusion_matrix(
                artifacts, all_labels, all_preds, config, fold
            )

            if config.task == TaskType.BINARY and all_probs is not None:
                ArtifactGenerator._create_roc_curve(
                    artifacts, all_labels, all_probs, config, fold
                )
            elif config.task == TaskType.MULTICLASS and all_probs is not None:
                ArtifactGenerator._create_multiclass_roc_curve(
                    artifacts, all_labels, all_probs, config, fold
                )
        except Exception as e:
            print(f"Warning: Could not create classification artifacts: {e}")

    @staticmethod
    def _create_regression_artifacts(
        artifacts: Dict[str, str],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        config: EvalConfig,
        fold: int,
    ) -> None:
        """Create regression-specific visualization artifacts."""
        try:
            ArtifactGenerator._create_regression_plot(
                artifacts, all_labels, all_preds, config, fold
            )
        except Exception as e:
            print(f"Warning: Could not create regression artifacts: {e}")

    @staticmethod
    def _create_confusion_matrix(
        artifacts: Dict[str, str],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        config: EvalConfig,
        fold: int,
    ) -> None:
        """Create confusion matrix plot."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - Fold {fold}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            cm_path = os.path.join(config.save_dir, f"confusion_matrix_fold_{fold}.png")
            plt.savefig(cm_path, bbox_inches="tight", dpi=300)
            plt.close()

            artifacts["confusion_matrix"] = os.path.normpath(cm_path)
        except Exception as e:
            print(f"Could not create confusion matrix: {e}")

    @staticmethod
    def _create_roc_curve(
        artifacts: Dict[str, str],
        all_labels: np.ndarray,
        all_probs: np.ndarray,
        config: EvalConfig,
        fold: int,
    ) -> None:
        """Create ROC curve plot for binary classification."""
        try:
            import matplotlib.pyplot as plt

            if all_probs.shape[1] == 1:
                positive_probs = all_probs[:, 0]
            else:
                positive_probs = all_probs[:, 1]

            fpr, tpr, _ = roc_curve(all_labels, positive_probs)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - Fold {fold}")
            plt.legend(loc="lower right")

            roc_path = os.path.join(config.save_dir, f"roc_curve_fold_{fold}.png")
            plt.savefig(roc_path, bbox_inches="tight", dpi=300)
            plt.close()

            artifacts["roc_curve"] = os.path.normpath(roc_path)
        except Exception as e:
            print(f"Could not create ROC curve: {e}")

    @staticmethod
    def _create_multiclass_roc_curve(
        artifacts: Dict[str, str],
        all_labels: np.ndarray,
        all_probs: np.ndarray,
        config: EvalConfig,
        fold: int,
    ) -> None:
        """Create multi-class ROC curve."""
        try:
            import matplotlib.pyplot as plt

            n_classes = all_probs.shape[1]
            y_true_bin = label_binarize(all_labels, classes=range(n_classes))

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot all ROC curves
            plt.figure(figsize=(10, 8))
            colors = ["blue", "red", "green", "orange", "purple", "brown"][:n_classes]

            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=2,
                    label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
                )

            plt.plot([0, 1], [0, 1], "k--", lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Multi-class ROC Curve - Fold {fold}")
            plt.legend(loc="lower right")

            roc_path = os.path.join(
                config.save_dir, f"multiclass_roc_curve_fold_{fold}.png"
            )
            plt.savefig(roc_path, bbox_inches="tight", dpi=300)
            plt.close()

            artifacts["multiclass_roc_curve"] = os.path.normpath(roc_path)
        except Exception as e:
            print(f"Could not create multi-class ROC curve: {e}")

    @staticmethod
    def _create_regression_plot(
        artifacts: Dict[str, str],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        config: EvalConfig,
        fold: int,
    ) -> None:
        """Create regression scatter plot."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))

            if all_preds.ndim == 1:
                # Single output regression
                plt.scatter(all_labels, all_preds, alpha=0.5)
                min_val = min(np.min(all_labels), np.min(all_preds))
                max_val = max(np.max(all_labels), np.max(all_preds))
                plt.plot([min_val, max_val], [min_val, max_val], "r--")
                plt.xlabel("True Values")
                plt.ylabel("Predicted Values")
            else:
                # Multi-output regression - plot first two outputs
                n_outputs = min(2, all_preds.shape[1])
                fig, axes = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))
                if n_outputs == 1:
                    axes = [axes]

                for i in range(n_outputs):
                    if all_labels.ndim == 2 and all_labels.shape[1] > i:
                        true_vals = all_labels[:, i]
                    else:
                        true_vals = all_labels

                    axes[i].scatter(true_vals, all_preds[:, i], alpha=0.5)
                    min_val = min(np.min(true_vals), np.min(all_preds[:, i]))
                    max_val = max(np.max(true_vals), np.max(all_preds[:, i]))
                    axes[i].plot([min_val, max_val], [min_val, max_val], "r--")
                    axes[i].set_xlabel(f"True Values Output {i}")
                    axes[i].set_ylabel(f"Predicted Values Output {i}")

                plt.tight_layout()

            plt.title(f"Regression Plot - Fold {fold}")

            reg_path = os.path.join(config.save_dir, f"regression_plot_fold_{fold}.png")
            plt.savefig(reg_path, bbox_inches="tight", dpi=300)
            plt.close()

            artifacts["regression_plot"] = os.path.normpath(reg_path)
        except Exception as e:
            print(f"Could not create regression plot: {e}")


class ResultsProcessor:
    """Processes and formats evaluation results."""

    @staticmethod
    def create_results_dataframe(
        patient_results: Dict[str, Any], task_type: TaskType, n_classes: int
    ) -> pd.DataFrame:
        """Create results DataFrame from patient results."""
        slide_ids = list(patient_results.keys())

        if task_type == TaskType.REGRESSION:
            return ResultsProcessor._create_regression_dataframe(
                patient_results, slide_ids
            )
        else:
            return ResultsProcessor._create_classification_dataframe(
                patient_results, slide_ids, n_classes
            )

    @staticmethod
    def _create_regression_dataframe(
        patient_results: Dict[str, Any], slide_ids: List[str]
    ) -> pd.DataFrame:
        """Create DataFrame for regression results with multi-output support."""
        all_labels = [patient_results[slide_id]["label"] for slide_id in slide_ids]
        all_preds = [patient_results[slide_id]["prob"] for slide_id in slide_ids]

        # Check if we have multi-output regression
        first_pred = all_preds[0] if all_preds else None
        has_multiple_outputs = (
            isinstance(first_pred, (list, np.ndarray))
            and len(np.array(first_pred).flatten()) > 1
        )

        if has_multiple_outputs:
            # Multi-output regression
            df_data = {"slide_id": slide_ids}

            # Handle predictions
            pred_arrays = [np.array(pred).flatten() for pred in all_preds]
            max_outputs = max(len(pred) for pred in pred_arrays) if pred_arrays else 1

            for i in range(max_outputs):
                df_data[f"pred_output_{i}"] = [
                    pred[i] if i < len(pred) else float("nan") for pred in pred_arrays
                ]

            # Handle labels
            first_label = all_labels[0] if all_labels else None
            if (
                isinstance(first_label, (list, np.ndarray))
                and len(np.array(first_label).flatten()) > 1
            ):
                # Multi-output labels
                label_arrays = [np.array(label).flatten() for label in all_labels]
                for i in range(max_outputs):
                    df_data[f"true_output_{i}"] = [
                        label[i] if i < len(label) else float("nan")
                        for label in label_arrays
                    ]
            else:
                # Single output labels - repeat for all outputs
                for i in range(max_outputs):
                    df_data[f"true_output_{i}"] = all_labels

        else:
            # Single output regression
            df_data = {
                "slide_id": slide_ids,
                "true_label": all_labels,
                "pred_label": all_preds,
            }

        return pd.DataFrame(df_data)

    @staticmethod
    def _create_classification_dataframe(
        patient_results: Dict[str, Any], slide_ids: List[str], n_classes: int
    ) -> pd.DataFrame:
        """Create DataFrame for classification results with robust probability handling."""
        all_labels = []
        all_preds = []
        all_probs = []

        for slide_id in slide_ids:
            result = patient_results[slide_id]
            all_labels.append(result["label"])
            prob_value = result["prob"]

            # Get prediction
            try:
                if isinstance(prob_value, (list, np.ndarray)):
                    prob_array = np.array(prob_value, dtype=float)
                    all_preds.append(np.argmax(prob_array))
                else:
                    # Scalar - assume it's positive class probability for binary
                    all_preds.append(1 if float(prob_value) > 0.5 else 0)
            except Exception as e:
                print(f"Error getting prediction for {slide_id}: {e}")
                all_preds.append(0)

            all_probs.append(prob_value)

        df_data = {
            "slide_id": slide_ids,
            "true_label": all_labels,
            "pred_label": all_preds,
        }

        # Add probability columns for each class
        for c in range(n_classes):
            prob_column = []
            for prob in all_probs:
                try:
                    prob_array = np.array(prob, dtype=float).flatten()

                    if len(prob_array) == 1 and n_classes == 2:
                        # Binary classification with single probability
                        if c == 0:
                            prob_column.append(1 - float(prob_array[0]))
                        else:
                            prob_column.append(float(prob_array[0]))
                    elif len(prob_array) > c:
                        prob_column.append(float(prob_array[c]))
                    else:
                        prob_column.append(0.0)
                except:
                    prob_column.append(0.0)

            df_data[f"prob_class_{c}"] = prob_column

        return pd.DataFrame(df_data)


class MLflowLogger:
    """Handles MLflow logging for evaluation results."""

    @staticmethod
    def log_evaluation_metrics(
        config: EvalConfig,
        test_error: float,
        auc_score: Optional[float],
        metrics_logger: BaseMetricsLogger,
        primary_metric: float,
        fold: Optional[int] = None,
    ) -> None:
        """Log evaluation metrics to MLflow."""
        # Log basic parameters
        MLflowLogger._log_basic_parameters(config, fold)

        # Log task-specific metrics
        if config.task == TaskType.REGRESSION:
            MLflowLogger._log_regression_metrics(
                test_error, metrics_logger, primary_metric
            )
        else:
            MLflowLogger._log_classification_metrics(
                test_error, auc_score, metrics_logger, config.n_classes
            )

    @staticmethod
    def _log_basic_parameters(config: EvalConfig, fold: Optional[int]) -> None:
        """Log basic evaluation parameters."""
        mlflow.log_param("model_type", config.model_type)
        mlflow.log_param("n_classes", config.n_classes)
        mlflow.log_param("task_type", config.task.value)
        if fold is not None:
            mlflow.log_param("fold", fold)

    @staticmethod
    def _log_regression_metrics(
        test_error: float, metrics_logger: BaseMetricsLogger, primary_metric: float
    ) -> None:
        """Log regression-specific metrics."""
        mlflow.log_metric("test_mae", primary_metric)

        # Log additional regression metrics
        regression_metrics = metrics_logger.get_all_metrics()
        for metric_name, metric_value in regression_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

    @staticmethod
    def _log_classification_metrics(
        test_error: float,
        auc_score: Optional[float],
        metrics_logger: BaseMetricsLogger,
        n_classes: int,
    ) -> None:
        """Log classification-specific metrics."""
        test_acc = 1.0 - test_error
        primary_metric = auc_score if auc_score is not None else test_acc

        mlflow.log_metric("test_error", test_error)
        mlflow.log_metric("test_accuracy", test_acc)
        if auc_score is not None:
            mlflow.log_metric("test_auc", auc_score)

        # Log class-wise accuracy
        if hasattr(metrics_logger, "get_summary"):
            for i in range(n_classes):
                acc, correct, count = metrics_logger.get_summary(i)
                if acc is not None:
                    mlflow.log_metric(f"test_class_{i}_acc", acc)
                    mlflow.log_metric(f"test_class_{i}_correct", correct)
                    mlflow.log_metric(f"test_class_{i}_total", count)

    @staticmethod
    def log_detailed_metrics(detailed_metrics: Dict[str, Any]) -> None:
        """Log detailed metrics to MLflow."""
        for metric_name, metric_value in detailed_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"test_{metric_name}", metric_value)

    @staticmethod
    def log_artifacts(
        artifacts: Dict[str, str], artifact_path: str = "evaluation_plots"
    ) -> None:
        """Log artifacts to MLflow."""
        for artifact_name, artifact_path in artifacts.items():
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path, artifact_path=artifact_path)


class EvaluationRunner:
    """Main orchestrator for running evaluations with robust task handling."""

    @staticmethod
    def evaluate(
        dataset: Any, config: EvalConfig, ckpt_path: str, fold: Optional[int] = None
    ) -> Tuple[
        torch.nn.Module,  # model
        Dict[str, Any],  # patient_results
        float,  # test_error
        float,  # primary_metric (auc_score or MAE)
        pd.DataFrame,  # results dataframe
        Optional[Dict[str, Any]],  # detailed_results
    ]:
        """Evaluates the model and logs final results to MLflow with detailed metrics."""
        eval_run_name = f"Fold_{fold}_k{config.k}"
        with mlflow.start_run(run_name=eval_run_name, nested=True) as run:

            # Log checkpoint path
            mlflow.log_param("eval_ckpt_path", ckpt_path)

            # Run evaluation pipeline
            return EvaluationRunner._run_evaluation_pipeline(
                dataset, config, ckpt_path, fold, eval_run_name
            )

    @staticmethod
    def _run_evaluation_pipeline(
        dataset: Any,
        config: EvalConfig,
        ckpt_path: str,
        fold: Optional[int],
        eval_run_name: str,
    ) -> Tuple[
        torch.nn.Module,
        Dict[str, Any],
        float,
        float,
        pd.DataFrame,
        Optional[Dict[str, Any]],
    ]:
        """Execute the complete evaluation pipeline."""
        # Load model and run evaluation
        model, patient_results, test_error, auc_score, metrics_logger = (
            EvaluationRunner._load_and_evaluate_model(config, dataset, ckpt_path)
        )

        # Calculate primary metric
        primary_metric = EvaluationRunner._calculate_primary_metric(
            config, test_error, auc_score, metrics_logger
        )

        # Log metrics to MLflow
        MLflowLogger.log_evaluation_metrics(
            config, test_error, auc_score, metrics_logger, primary_metric, fold
        )

        # Generate detailed results
        detailed_results = EvaluationRunner._generate_detailed_results(
            config, patient_results, fold
        )

        # Create results dataframe
        df = ResultsProcessor.create_results_dataframe(
            patient_results, config.task, config.n_classes
        )

        # Log artifacts
        EvaluationRunner._log_evaluation_artifacts(
            config, df, eval_run_name, detailed_results
        )

        # Print summary
        EvaluationRunner._print_evaluation_summary(
            config, test_error, auc_score, metrics_logger
        )

        return model, patient_results, test_error, primary_metric, df, detailed_results

    @staticmethod
    def _setup_mlflow_experiment(config: EvalConfig) -> mlflow.ActiveRun:
        # ✅ Set seed exactly ONCE
        print(f"Setting random seed: {config.seed}")
        seed_torch(config.seed)

        """Setup MLflow experiment and return the active run"""
        experiment_name = f"Eval_{config.models_exp_code}"
        mlflow.set_experiment(experiment_name)

        run_name = f"CV_Seed{config.seed}_k{config.k}"
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def _load_and_evaluate_model(
        config: EvalConfig, dataset: Any, ckpt_path: str
    ) -> Tuple[
        torch.nn.Module, Dict[str, Any], float, Optional[float], BaseMetricsLogger
    ]:
        """Load model and run evaluation."""
        print("Initializing model and loader...")
        model = ModelLoader.load_model(config, ckpt_path)
        loader = DataLoaderFactory.get_simple_loader(dataset, config.task)

        # Use ModelEvaluator for consistent evaluation
        patient_results, test_error, auc_score, metrics_logger = ModelEvaluator.summary(
            model, loader, config.n_classes, config.task
        )

        return model, patient_results, test_error, auc_score, metrics_logger

    @staticmethod
    def _calculate_primary_metric(
        config: EvalConfig,
        test_error: float,
        auc_score: Optional[float],
        metrics_logger: BaseMetricsLogger,
    ) -> float:
        """Calculate primary metric based on task type."""
        if config.task == TaskType.REGRESSION:
            return test_error  # test_error is MAE for regression
        else:
            test_acc = 1.0 - test_error
            return auc_score if auc_score is not None else test_acc

    @staticmethod
    def _generate_detailed_results(
        config: EvalConfig, patient_results: Dict[str, Any], fold: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Generate detailed results if enabled."""
        if not config.detailed_metrics:
            return None

        try:
            # Extract predictions and labels
            all_labels, all_preds, all_probs = EvaluationRunner._extract_predictions(
                config, patient_results
            )

            # Create detailed metrics and artifacts
            detailed_metrics, artifacts = (
                ArtifactGenerator.create_detailed_metrics_artifacts(
                    all_labels,
                    all_preds,
                    all_probs,
                    config,
                    fold if fold is not None else 0,
                )
            )

            # Save comprehensive metrics as JSON
            metrics_json_path = EvaluationRunner._save_comprehensive_metrics(
                config, detailed_metrics, fold
            )

            # Log to MLflow
            MLflowLogger.log_detailed_metrics(detailed_metrics)
            MLflowLogger.log_artifacts(artifacts)
            mlflow.log_artifact(metrics_json_path)

            return {
                "detailed_metrics": detailed_metrics,
                "artifacts": artifacts,
                "metrics_json_path": metrics_json_path,
            }

        except Exception as e:
            print(f"Warning: Could not generate detailed metrics: {e}")
            return None

    @staticmethod
    def _extract_predictions(
        config: EvalConfig, patient_results: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract predictions and labels from patient results with robust task handling."""
        all_labels = []
        all_preds = []
        all_probs = []

        print(
            f"Extracting predictions for task: {config.task}, n_classes: {config.n_classes}"
        )

        for i, (slide_id, result) in enumerate(patient_results.items()):
            if i < 3:  # Debug first 3 samples
                print(f"Sample {i}: label={result['label']}, prob={result['prob']}")

            label = result["label"]
            prob_value = result["prob"]

            try:
                if config.task == TaskType.REGRESSION:
                    processed_pred, processed_prob = (
                        EvaluationRunner._handle_regression(prob_value)
                    )
                    all_preds.append(processed_pred)
                    all_probs.append(processed_prob)

                elif config.task == TaskType.BINARY:
                    processed_pred, processed_prob = EvaluationRunner._handle_binary(
                        prob_value
                    )
                    all_preds.append(processed_pred)
                    all_probs.append(processed_prob)

                elif config.task == TaskType.MULTICLASS:
                    processed_pred, processed_prob = (
                        EvaluationRunner._handle_multiclass(
                            prob_value, config.n_classes
                        )
                    )
                    all_preds.append(processed_pred)
                    all_probs.append(processed_prob)

                else:
                    raise ValueError(f"Unknown task type: {config.task}")

                all_labels.append(label)

            except Exception as e:
                print(f"Error processing sample {slide_id}: {e}")
                print(f"  Label: {label}, Probability value: {prob_value}")
                continue

        # Convert to numpy arrays with proper formatting
        labels_array, preds_array, probs_array = (
            EvaluationRunner._format_arrays_for_metrics(
                all_labels, all_preds, all_probs, config.task
            )
        )

        print(
            f"Final shapes - labels: {labels_array.shape}, preds: {preds_array.shape}, probs: {probs_array.shape}"
        )

        return labels_array, preds_array, probs_array

    @staticmethod
    def _format_arrays_for_metrics(all_labels, all_preds, all_probs, task_type):
        """Format arrays appropriately for metrics calculation based on task type."""
        labels_array = np.array(all_labels)

        if task_type == TaskType.REGRESSION:
            # For regression, predictions are the actual values
            preds_array = np.array(all_preds)
            # Ensure proper shape for regression metrics
            if preds_array.ndim == 1:
                probs_array = preds_array.reshape(-1, 1)
            else:
                probs_array = preds_array
        else:
            # For classification
            preds_array = np.array(all_preds)
            # Handle probability arrays
            if (
                all_probs
                and hasattr(all_probs[0], "__len__")
                and not isinstance(all_probs[0], str)
            ):
                try:
                    probs_array = np.array(all_probs)
                    if probs_array.ndim == 1:
                        probs_array = probs_array.reshape(-1, 1)
                except:
                    probs_array = np.array(all_probs).reshape(-1, 1)
            else:
                probs_array = np.array(all_probs).reshape(-1, 1)

        return labels_array, preds_array, probs_array

    @staticmethod
    def _handle_regression(prob_value: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Handle regression predictions - can be single or multiple outputs."""
        try:
            if isinstance(prob_value, (list, np.ndarray)):
                prob_array = np.array(prob_value, dtype=float)

                if prob_array.ndim == 0:
                    # Scalar regression output
                    pred = np.array([float(prob_array)])
                else:
                    # Array output (single or multiple)
                    pred = prob_array.flatten()
            else:
                # Scalar value
                pred = np.array([float(prob_value)])

            return pred, pred.copy()

        except Exception as e:
            print(f"Error in regression handling: {e}, value: {prob_value}")
            return np.array([0.0]), np.array([0.0])

    @staticmethod
    def _handle_binary(prob_value: Any) -> Tuple[int, np.ndarray]:
        """Handle binary classification predictions."""
        try:
            if isinstance(prob_value, (list, np.ndarray)):
                prob_array = np.array(prob_value, dtype=float).flatten()
            else:
                prob_array = np.array([prob_value], dtype=float)

            # Handle different probability formats for binary classification
            if len(prob_array) == 1:
                # Single probability - assume it's for positive class
                prob_positive = float(prob_array[0])
                probabilities = np.array([1 - prob_positive, prob_positive])
                prediction = 1 if prob_positive > 0.5 else 0
            else:
                # Multiple probabilities - use first two
                probabilities = prob_array[:2]
                prediction = np.argmax(probabilities)

            # Normalize probabilities
            prob_sum = np.sum(probabilities)
            if not np.isclose(prob_sum, 1.0, atol=0.01) and prob_sum > 0:
                probabilities = probabilities / prob_sum

            return prediction, probabilities

        except Exception as e:
            print(f"Error in binary classification handling: {e}, value: {prob_value}")
            return 0, np.array([1.0, 0.0])

    @staticmethod
    def _handle_multiclass(prob_value: Any, n_classes: int) -> Tuple[int, np.ndarray]:
        """Handle multi-class classification predictions."""
        try:
            if isinstance(prob_value, (list, np.ndarray)):
                prob_array = np.array(prob_value, dtype=float).flatten()
            else:
                prob_array = np.array([prob_value], dtype=float)

            # Ensure we have the right number of probabilities
            if len(prob_array) == n_classes:
                probabilities = prob_array
            elif len(prob_array) > n_classes:
                probabilities = prob_array[:n_classes]
            else:
                probabilities = np.zeros(n_classes)
                probabilities[: len(prob_array)] = prob_array

            prediction = np.argmax(probabilities)

            # Normalize probabilities
            prob_sum = np.sum(probabilities)
            if not np.isclose(prob_sum, 1.0, atol=0.01) and prob_sum > 0:
                probabilities = probabilities / prob_sum

            return prediction, probabilities

        except Exception as e:
            print(
                f"Error in multi-class classification handling: {e}, value: {prob_value}"
            )
            probabilities = np.ones(n_classes) / n_classes
            return 0, probabilities

    @staticmethod
    def _save_comprehensive_metrics(
        config: EvalConfig, detailed_metrics: Dict[str, Any], fold: Optional[int]
    ) -> str:
        """Save comprehensive metrics to JSON file."""
        metrics_json_path = os.path.join(
            config.save_dir,
            f"comprehensive_metrics_fold_{fold if fold is not None else 0}.json",
        )
        metrics_json_path = os.path.normpath(metrics_json_path)
        with open(metrics_json_path, "w") as f:
            json.dump(detailed_metrics, f, indent=2)
        return metrics_json_path

    @staticmethod
    def _log_evaluation_artifacts(
        config: EvalConfig,
        df: pd.DataFrame,
        eval_run_name: str,
        detailed_results: Optional[Dict[str, Any]],
    ) -> None:
        """Log evaluation artifacts to MLflow."""
        # Log results DataFrame
        results_path = os.path.join(config.save_dir, f"{eval_run_name}_results.csv")
        df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        # Log detailed metrics JSON if available
        if detailed_results and "metrics_json_path" in detailed_results:
            mlflow.log_artifact(detailed_results["metrics_json_path"])

    @staticmethod
    def _print_evaluation_summary(
        config: EvalConfig,
        test_error: float,
        auc_score: Optional[float],
        metrics_logger: BaseMetricsLogger,
    ) -> None:
        """Print evaluation summary."""
        if config.task == TaskType.REGRESSION:
            regression_metrics = metrics_logger.get_all_metrics()
            print(f"Test MAE: {test_error:.4f}")
            if "r2" in regression_metrics:
                print(f"Test R²: {regression_metrics['r2']:.4f}")
        else:
            print(f"Test Error: {test_error:.4f}")
            print(f"Test Accuracy: {1.0 - test_error:.4f}")
            if auc_score is not None:
                print(f"Test AUC: {auc_score:.4f}")

    @staticmethod
    def _load_test_split(
        config: EvalConfig, fold: int, dataset: Generic_MIL_Dataset
    ) -> Any:
        """Load test dataset split for the given fold."""
        try:
            _, _, test_dataset = dataset.return_splits(
                from_id=False,
                csv_path=os.path.join(config.splits_dir, f"splits_{fold}.csv"),
            )
            return test_dataset
        except Exception as e:
            print(f"Error loading split for fold {fold}: {e}")
            return None

    @staticmethod
    def _get_checkpoint_path(config: EvalConfig, fold: int) -> Optional[str]:
        """Get checkpoint path for the given fold."""
        ckpt_path = os.path.join(config.models_dir, f"s_{fold}_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for fold {fold}: {ckpt_path}")
            return None
        return ckpt_path

    @staticmethod
    def _format_fold_results(
        results: Tuple[
            torch.nn.Module,
            Dict[str, Any],
            float,
            float,
            pd.DataFrame,
            Optional[Dict[str, Any]],
        ],
    ) -> Dict[str, Any]:
        """Format results for a single fold."""
        model, patient_results, test_error, primary_metric, df, detailed_results = (
            results
        )
        return {
            "model": model,
            "patient_results": patient_results,
            "test_error": test_error,
            "primary_metric": primary_metric,
            "dataframe": df,
            "detailed_results": detailed_results,
        }


def run_evaluation(config: EvalConfig, dataset: Any) -> Dict[int, Dict[str, Any]]:
    """Run comprehensive evaluation across multiple folds."""
    all_results: Dict[int, Dict[str, Any]] = {}

    with EvaluationRunner._setup_mlflow_experiment(config) as run:

        for fold in config.folds:
            print(f"\n{'='*50}")
            print(f"Evaluating Fold {fold}")
            print(f"{'='*50}")

            try:
                # Load dataset split for this fold
                test_dataset = EvaluationRunner._load_test_split(config, fold, dataset)
                if test_dataset is None:
                    continue

                # Find checkpoint for this fold
                ckpt_path = EvaluationRunner._get_checkpoint_path(config, fold)
                if ckpt_path is None:
                    continue

                # Run evaluation
                results = EvaluationRunner.evaluate(test_dataset, config, ckpt_path, fold)
                all_results[fold] = EvaluationRunner._format_fold_results(results)

            except Exception as e:
                print(f"Error evaluating fold {fold}: {e}")
                continue

    return all_results
