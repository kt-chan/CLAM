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

# Import from your existing utilities
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
    """Configuration class for evaluation with strong type hints."""

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
        assert os.path.isdir(
            self.models_dir
        ), f"Models directory {self.models_dir} does not exist"
        assert os.path.isdir(
            self.splits_dir
        ), f"Splits directory {self.splits_dir} does not exist"

        # Setup fold ranges
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


def initiate_model(
    config: EvalConfig, ckpt_path: str, device: str = "cuda"
) -> torch.nn.Module:
    """Initialize model from checkpoint with proper configuration."""
    print("Init Model from checkpoint:", ckpt_path)

    try:
        # Use ModelManager to reconstruct the model (same approach as train_utils.py)
        model = ModelManager.reconstruct_clam_model(config, ckpt_path)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model with ModelManager: {e}")
        print("Falling back to direct model loading...")

        # Fallback to direct model loading
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
        if config.task == TaskType.REGRESSION:
            if config.model_type in ["clam_sb", "clam_sbr"]:
                from models.model_clam import CLAM_SB_Regression

                model = CLAM_SB_Regression(**model_dict)
            elif config.model_type in ["clam_mb", "clam_mbr"]:
                from models.model_clam import CLAM_MB_Regression

                model = CLAM_MB_Regression(**model_dict)
            else:
                from models.model_mil import MIL_fc

                model = MIL_fc(**model_dict)
        else:
            if config.model_type == "clam_sb":
                from models.model_clam import CLAM_SB

                model = CLAM_SB(**model_dict)
            elif config.model_type == "clam_mb":
                from models.model_clam import CLAM_MB

                model = CLAM_MB(**model_dict)
            else:
                if config.n_classes > 2:
                    from models.model_mil import MIL_fc_mc

                    model = MIL_fc_mc(**model_dict)
                else:
                    from models.model_mil import MIL_fc

                    model = MIL_fc(**model_dict)

        # Load state dict
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_clean = {}

        for key in ckpt.keys():
            if "instance_loss_fn" in key:
                continue
            ckpt_clean.update({key.replace(".module", ""): ckpt[key]})

        model.load_state_dict(ckpt_clean, strict=True)
        model = model.to(device)
        model.eval()

        return model


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
            # Regression metrics
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

        else:
            # Classification metrics
            metrics["accuracy"] = float(np.mean(all_preds == all_labels))
            metrics["balanced_accuracy"] = float(
                balanced_accuracy_score(all_labels, all_preds)
            )

            # Precision, Recall, F1
            metrics["precision_macro"] = float(
                precision_score(all_labels, all_preds, average="macro", zero_division=0)
            )
            metrics["recall_macro"] = float(
                recall_score(all_labels, all_preds, average="macro", zero_division=0)
            )
            metrics["f1_macro"] = float(
                f1_score(all_labels, all_preds, average="macro", zero_division=0)
            )

            metrics["precision_weighted"] = float(
                precision_score(
                    all_labels, all_preds, average="weighted", zero_division=0
                )
            )
            metrics["recall_weighted"] = float(
                recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            )
            metrics["f1_weighted"] = float(
                f1_score(all_labels, all_preds, average="weighted", zero_division=0)
            )

            # Per-class metrics
            for class_idx in range(n_classes):
                class_mask = all_labels == class_idx
                if np.sum(class_mask) > 0:
                    class_accuracy = float(np.mean(all_preds[class_mask] == class_idx))
                    metrics[f"class_{class_idx}_accuracy"] = class_accuracy

                    # For binary classification, calculate class-specific precision/recall
                    if n_classes == 2:
                        binary_preds = (all_preds == class_idx).astype(int)
                        binary_labels = (all_labels == class_idx).astype(int)
                        if len(np.unique(binary_labels)) > 1:
                            metrics[f"class_{class_idx}_precision"] = float(
                                precision_score(
                                    binary_labels, binary_preds, zero_division=0
                                )
                            )
                            metrics[f"class_{class_idx}_recall"] = float(
                                recall_score(
                                    binary_labels, binary_preds, zero_division=0
                                )
                            )
                            metrics[f"class_{class_idx}_f1"] = float(
                                f1_score(binary_labels, binary_preds, zero_division=0)
                            )

            # Confidence metrics (only for classification)
            if all_probs is not None and all_probs.size > 0:
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

    except Exception as e:
        print(f"Warning: Error calculating detailed metrics: {e}")

    return metrics


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
        detailed_metrics = calculate_detailed_metrics(
            all_labels, all_preds, all_probs, config.n_classes, config.task
        )

        # Create confusion matrix plot for classification
        if config.task != TaskType.REGRESSION:
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

                cm_path = os.path.join(
                    config.save_dir, f"confusion_matrix_fold_{fold}.png"
                )
                plt.savefig(cm_path, bbox_inches="tight", dpi=300)
                plt.close()

                artifacts["confusion_matrix"] = cm_path
            except Exception as e:
                print(f"Could not create confusion matrix: {e}")

        # Create ROC curve for binary classification
        if config.task == TaskType.BINARY and all_probs is not None:
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

    except Exception as e:
        print(f"Warning: Could not create detailed artifacts: {e}")

    return detailed_metrics, artifacts


def eval(
    dataset: Any, config: EvalConfig, ckpt_path: str, fold: Optional[int] = None
) -> Tuple[
    torch.nn.Module,  # model
    Dict[str, Any],  # patient_results
    float,  # test_error
    float,  # auc_score or MAE
    pd.DataFrame,  # df
    Optional[Dict[str, Any]],  # detailed_results
]:
    """
    Evaluates the model and logs final results to MLflow with detailed metrics.
    """
    # MLflow nested run for evaluation
    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    eval_run_name = f"Evaluation{fold_suffix}: {os.path.basename(ckpt_path)}"

    with mlflow.start_run(run_name=eval_run_name, nested=True) as run:
        # Log essential parameters and the model checkpoint path
        mlflow.log_param("eval_ckpt_path", ckpt_path)
        mlflow.log_param("model_type", config.model_type)
        mlflow.log_param("n_classes", config.n_classes)
        mlflow.log_param("task_type", config.task.value)
        if fold is not None:
            mlflow.log_param("fold", fold)

        # Initiate model
        model = initiate_model(config, ckpt_path)

        print("Init Loaders")
        loader = DataLoaderFactory.get_simple_loader(dataset, config.task)

        # Use ModelEvaluator for consistent evaluation (same as train_utils.py)
        patient_results, test_error, auc_score, metrics_logger = ModelEvaluator.summary(
            model, loader, config.n_classes, config.task
        )

        # Calculate additional metrics based on task type
        if config.task == TaskType.REGRESSION:
            # For regression, use MAE as primary metric
            primary_metric = test_error  # test_error is MAE for regression
            mlflow.log_metric("test_mae", primary_metric)

            # Log additional regression metrics from metrics_logger
            regression_metrics = metrics_logger.get_all_metrics()
            for metric_name, metric_value in regression_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
        else:
            # For classification, use accuracy and AUC
            test_acc = 1.0 - test_error
            primary_metric = auc_score if auc_score is not None else test_acc

            mlflow.log_metric("test_error", test_error)
            mlflow.log_metric("test_accuracy", test_acc)
            if auc_score is not None:
                mlflow.log_metric("test_auc", auc_score)

            # Log class-wise accuracy
            if hasattr(metrics_logger, "get_summary"):
                for i in range(config.n_classes):
                    acc, correct, count = metrics_logger.get_summary(i)
                    if acc is not None:
                        mlflow.log_metric(f"test_class_{i}_acc", acc)
                        mlflow.log_metric(f"test_class_{i}_correct", correct)
                        mlflow.log_metric(f"test_class_{i}_total", count)

        # Generate detailed results if enabled
        detailed_results = None
        if config.detailed_metrics:
            try:
                # Extract predictions and labels for detailed metrics
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

                all_labels = np.array(all_labels)
                all_preds = np.array(all_preds)
                all_probs = np.array(all_probs)

                # Create detailed metrics and artifacts
                detailed_metrics, artifacts = create_detailed_metrics_artifacts(
                    all_labels,
                    all_preds,
                    all_probs,
                    config,
                    fold if fold is not None else 0,
                )

                # Log detailed metrics to MLflow
                for metric_name, metric_value in detailed_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"test_{metric_name}", metric_value)

                # Save comprehensive metrics as JSON
                metrics_json_path = os.path.join(
                    config.save_dir,
                    f"comprehensive_metrics_fold_{fold if fold is not None else 0}.json",
                )
                with open(metrics_json_path, "w") as f:
                    json.dump(detailed_metrics, f, indent=2)

                detailed_results = {
                    "detailed_metrics": detailed_metrics,
                    "artifacts": artifacts,
                    "metrics_json_path": metrics_json_path,
                }

                # Log artifacts
                mlflow.log_artifact(metrics_json_path)
                for artifact_name, artifact_path in artifacts.items():
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(
                            artifact_path, artifact_path="evaluation_plots"
                        )

            except Exception as e:
                print(f"Warning: Could not generate detailed metrics: {e}")
                detailed_results = None

        # Create results DataFrame
        slide_ids = list(patient_results.keys())
        all_labels = [patient_results[slide_id]["label"] for slide_id in slide_ids]

        if config.task == TaskType.REGRESSION:
            all_preds = [patient_results[slide_id]["prob"] for slide_id in slide_ids]
            # For regression with 2D output
            if isinstance(all_preds[0], (list, np.ndarray)) and len(all_preds[0]) == 2:
                df_data = {
                    "slide_id": slide_ids,
                    "true_primary": [
                        label[0] if isinstance(label, (list, np.ndarray)) else label
                        for label in all_labels
                    ],
                    "true_secondary": [
                        label[1] if isinstance(label, (list, np.ndarray)) else label
                        for label in all_labels
                    ],
                    "pred_primary": [pred[0] for pred in all_preds],
                    "pred_secondary": [pred[1] for pred in all_preds],
                }
            else:
                df_data = {
                    "slide_id": slide_ids,
                    "true_label": all_labels,
                    "pred_label": all_preds,
                }
        else:
            all_preds = [
                np.argmax(patient_results[slide_id]["prob"]) for slide_id in slide_ids
            ]
            all_probs = [patient_results[slide_id]["prob"] for slide_id in slide_ids]

            df_data = {
                "slide_id": slide_ids,
                "Y": all_labels,
                "Y_hat": all_preds,
            }
            for c in range(config.n_classes):
                df_data[f"p_{c}"] = [prob[c] for prob in all_probs]

        df = pd.DataFrame(df_data)

        # Log results DataFrame as artifact
        results_path = os.path.join(config.save_dir, f"{eval_run_name}_results.csv")
        df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        # Print summary
        if config.task == TaskType.REGRESSION:
            print(f"Test MAE: {test_error:.4f}")
            if "test_r2" in regression_metrics:
                print(f"Test R²: {regression_metrics['test_r2']:.4f}")
        else:
            print(f"Test Error: {test_error:.4f}")
            print(f"Test Accuracy: {1.0 - test_error:.4f}")
            if auc_score is not None:
                print(f"Test AUC: {auc_score:.4f}")

    return model, patient_results, test_error, primary_metric, df, detailed_results


def run_evaluation(config: EvalConfig, dataset: Any) -> Dict[str, Any]:
    """Run comprehensive evaluation across multiple folds."""
    all_results = {}

    for fold in config.folds:
        print(f"\n{'='*50}")
        print(f"Evaluating Fold {fold}")
        print(f"{'='*50}")

        # Load dataset split for this fold
        try:
            _, _, test_dataset = dataset.return_splits(
                from_id=False,
                csv_path=os.path.join(config.splits_dir, f"splits_{fold}.csv"),
            )
        except Exception as e:
            print(f"Error loading split for fold {fold}: {e}")
            continue

        # Find checkpoint for this fold
        ckpt_path = os.path.join(config.models_dir, f"s_{fold}_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for fold {fold}: {ckpt_path}")
            continue

        # Run evaluation
        try:
            model, patient_results, test_error, primary_metric, df, detailed_results = (
                eval(test_dataset, config, ckpt_path, fold)
            )

            # Store results
            all_results[fold] = {
                "model": model,
                "patient_results": patient_results,
                "test_error": test_error,
                "primary_metric": primary_metric,
                "dataframe": df,
                "detailed_results": detailed_results,
            }

        except Exception as e:
            print(f"Error evaluating fold {fold}: {e}")
            continue

    return all_results
