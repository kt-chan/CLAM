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

    # Type annotations for all attributes
    data_root_dir: Optional[str]
    data_set_name: Optional[str]
    results_dir: str
    save_exp_code: Optional[str]
    models_exp_code: Optional[str]
    splits_dir: Optional[str]
    model_size: str
    model_type: str
    drop_out: float
    embed_dim: int
    k: int
    k_start: int
    k_end: int
    fold: int
    micro_average: bool
    split: str
    task: Optional[str]
    subtyping: bool
    registered_model_name: Optional[str]
    register_best_model: bool
    detailed_metrics: bool
    n_classes: Optional[int]
    task: TaskType
    save_dir: str
    models_dir: str
    start_fold: int
    end_fold: int
    folds: range

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
                    metrics, all_labels, all_preds, all_probs, n_classes
                )
        except Exception as e:
            print(f"Warning: Error calculating detailed metrics: {e}")

        return metrics

    @staticmethod
    def _calculate_regression_metrics(
        metrics: Dict[str, Any], all_labels: np.ndarray, all_preds: np.ndarray
    ) -> None:
        """Calculate regression-specific metrics."""
        metrics["mae"] = float(mean_absolute_error(all_labels, all_preds))
        metrics["mse"] = float(mean_squared_error(all_labels, all_preds))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))

        # Handle multi-output regression (primary, secondary)
        if all_preds.ndim == 2 and all_preds.shape[1] == 2:
            metrics["primary_mae"] = float(
                mean_absolute_error(all_labels[:, 0], all_preds[:, 0])
            )
            metrics["secondary_mae"] = float(
                mean_absolute_error(all_labels[:, 1], all_preds[:, 1])
            )

        # R-squared calculation
        if len(all_labels) > 1:
            ss_res = np.sum((all_labels - all_preds) ** 2)
            ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
            metrics["r2"] = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    @staticmethod
    def _calculate_classification_metrics(
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        all_probs: np.ndarray,
        n_classes: int,
    ) -> None:
        """Calculate classification-specific metrics."""
        metrics["accuracy"] = float(np.mean(all_preds == all_labels))
        metrics["balanced_accuracy"] = float(
            balanced_accuracy_score(all_labels, all_preds)
        )

        # Precision, Recall, F1
        MetricsCalculator._calculate_basic_classification_metrics(
            metrics, all_labels, all_preds
        )

        # Per-class metrics
        MetricsCalculator._calculate_per_class_metrics(
            metrics, all_labels, all_preds, n_classes
        )

        # Confidence metrics (only for classification)
        if all_probs is not None and all_probs.size > 0:
            MetricsCalculator._calculate_confidence_metrics(
                metrics, all_labels, all_preds, all_probs
            )

    @staticmethod
    def _calculate_basic_classification_metrics(
        metrics: Dict[str, Any], all_labels: np.ndarray, all_preds: np.ndarray
    ) -> None:
        """Calculate basic classification metrics."""
        metrics.update(
            {
                "precision_macro": float(
                    precision_score(
                        all_labels, all_preds, average="macro", zero_division=0
                    )
                ),
                "recall_macro": float(
                    recall_score(
                        all_labels, all_preds, average="macro", zero_division=0
                    )
                ),
                "f1_macro": float(
                    f1_score(all_labels, all_preds, average="macro", zero_division=0)
                ),
                "precision_weighted": float(
                    precision_score(
                        all_labels, all_preds, average="weighted", zero_division=0
                    )
                ),
                "recall_weighted": float(
                    recall_score(
                        all_labels, all_preds, average="weighted", zero_division=0
                    )
                ),
                "f1_weighted": float(
                    f1_score(all_labels, all_preds, average="weighted", zero_division=0)
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

                # For binary classification, calculate class-specific precision/recall
                if n_classes == 2:
                    MetricsCalculator._calculate_binary_class_metrics(
                        metrics, all_labels, all_preds, class_idx
                    )

    @staticmethod
    def _calculate_binary_class_metrics(
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        class_idx: int,
    ) -> None:
        """Calculate binary classification metrics for a specific class."""
        binary_preds = (all_preds == class_idx).astype(int)
        binary_labels = (all_labels == class_idx).astype(int)

        if len(np.unique(binary_labels)) > 1:
            metrics.update(
                {
                    f"class_{class_idx}_precision": float(
                        precision_score(binary_labels, binary_preds, zero_division=0)
                    ),
                    f"class_{class_idx}_recall": float(
                        recall_score(binary_labels, binary_preds, zero_division=0)
                    ),
                    f"class_{class_idx}_f1": float(
                        f1_score(binary_labels, binary_preds, zero_division=0)
                    ),
                }
            )

    @staticmethod
    def _calculate_confidence_metrics(
        metrics: Dict[str, Any],
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        all_probs: np.ndarray,
    ) -> None:
        """Calculate confidence-based metrics."""
        max_probs = np.max(all_probs, axis=1)
        metrics["mean_confidence"] = float(np.mean(max_probs))
        metrics["confidence_std"] = float(np.std(max_probs))

        # Calibration metrics
        correct_predictions = all_preds == all_labels
        metrics["avg_confidence_correct"] = (
            float(np.mean(max_probs[correct_predictions]))
            if np.sum(correct_predictions) > 0
            else 0.0
        )
        metrics["avg_confidence_incorrect"] = (
            float(np.mean(max_probs[~correct_predictions]))
            if np.sum(~correct_predictions) > 0
            else 0.0
        )


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
        except Exception as e:
            print(f"Warning: Could not create classification artifacts: {e}")

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

            artifacts["confusion_matrix"] = cm_path
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

            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
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

            artifacts["roc_curve"] = roc_path
        except Exception as e:
            print(f"Could not create ROC curve: {e}")


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
        """Create DataFrame for regression results."""
        all_labels = [patient_results[slide_id]["label"] for slide_id in slide_ids]
        all_preds = [patient_results[slide_id]["prob"] for slide_id in slide_ids]

        # For regression with 2D output
        if isinstance(all_preds[0], (list, np.ndarray)) and len(all_preds[0]) == 2:
            df_data = {
                "slide_id": slide_ids,
                "true_primary": ResultsProcessor._extract_primary_labels(all_labels),
                "true_secondary": ResultsProcessor._extract_secondary_labels(
                    all_labels
                ),
                "pred_primary": [pred[0] for pred in all_preds],
                "pred_secondary": [pred[1] for pred in all_preds],
            }
        else:
            df_data = {
                "slide_id": slide_ids,
                "true_label": all_labels,
                "pred_label": all_preds,
            }

        return pd.DataFrame(df_data)

    @staticmethod
    def _extract_primary_labels(all_labels: List[Any]) -> List[float]:
        """Extract primary labels from label data."""
        return [
            label[0] if isinstance(label, (list, np.ndarray)) else float(label)
            for label in all_labels
        ]

    @staticmethod
    def _extract_secondary_labels(all_labels: List[Any]) -> List[float]:
        """Extract secondary labels from label data."""
        return [
            label[1] if isinstance(label, (list, np.ndarray)) else float(label)
            for label in all_labels
        ]

    @staticmethod
    def _create_classification_dataframe(
        patient_results: Dict[str, Any], slide_ids: List[str], n_classes: int
    ) -> pd.DataFrame:
        """Create DataFrame for classification results."""
        all_labels = [patient_results[slide_id]["label"] for slide_id in slide_ids]
        all_preds = [
            np.argmax(patient_results[slide_id]["prob"]) for slide_id in slide_ids
        ]
        all_probs = [patient_results[slide_id]["prob"] for slide_id in slide_ids]

        df_data = {
            "slide_id": slide_ids,
            "Y": all_labels,
            "Y_hat": all_preds,
        }

        # Add probability columns for each class
        for c in range(n_classes):
            df_data[f"p_{c}"] = [prob[c] for prob in all_probs]

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
    """Main orchestrator for running evaluations."""

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
        """
        Evaluates the model and logs final results to MLflow with detailed metrics.
        """
        experiment_name = f"Eval_{config.models_exp_code}"
        mlflow.set_experiment(experiment_name)
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
        """Extract predictions and labels from patient results."""
        all_labels = []
        all_preds = []
        all_probs = []

        for slide_id, result in patient_results.items():
            all_labels.append(result["label"])
            if config.task == TaskType.REGRESSION:
                all_preds.append(
                    result["prob"]
                )  # For regression, prob contains predictions
                all_probs.append(result["prob"])
            else:
                all_preds.append(np.argmax(result["prob"]))
                all_probs.append(result["prob"])

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    @staticmethod
    def _save_comprehensive_metrics(
        config: EvalConfig, detailed_metrics: Dict[str, Any], fold: Optional[int]
    ) -> str:
        """Save comprehensive metrics to JSON file."""
        metrics_json_path = os.path.join(
            config.save_dir,
            f"comprehensive_metrics_fold_{fold if fold is not None else 0}.json",
        )
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


    def _load_test_split(config: EvalConfig, fold: int, dataset: Generic_MIL_Dataset) -> Any:
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


    def _get_checkpoint_path(config: EvalConfig, fold: int) -> Optional[str]:
        """Get checkpoint path for the given fold."""
        ckpt_path = os.path.join(config.models_dir, f"s_{fold}_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for fold {fold}: {ckpt_path}")
            return None
        return ckpt_path


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
        model, patient_results, test_error, primary_metric, df, detailed_results = results
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

