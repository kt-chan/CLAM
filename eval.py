from __future__ import print_function
from typing import Optional, Dict, List, Any, Tuple, Union
import json

import numpy as np
import argparse
import torch
import os
import pandas as pd
from train import reconstruct_clam_model
from utils.utils import *
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import (
    Generic_MIL_Dataset,
)
from utils.eval_utils import *
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class EvalConfig:
    """Configuration class for evaluation with strong type hints."""

    # Data parameters
    data_root_dir: Optional[str]
    data_set_name: Optional[str]

    # Results and paths
    results_dir: str
    save_exp_code: Optional[str]
    models_exp_code: Optional[str]
    splits_dir: Optional[str]

    # Model parameters
    model_size: str
    model_type: str
    drop_out: float
    embed_dim: int

    # Evaluation parameters
    k: int
    k_start: int
    k_end: int
    fold: int
    micro_average: bool
    split: str
    task: Optional[str]
    subtyping: bool

    # MLflow model registration parameters
    registered_model_name: Optional[str]
    register_best_model: bool
    detailed_metrics: bool

    # Derived attributes
    n_classes: Optional[int]
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
        # Set number of classes based on task
        if self.task == "task_1_tumor_vs_normal":
            self.n_classes = 2
        elif self.task == "task_2_tumor_subtyping":
            self.n_classes = 3
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


def setup_eval_dataset(config: EvalConfig) -> Generic_MIL_Dataset:
    """Setup dataset for evaluation based on configuration."""
    data_dir: str = os.path.join(config.data_root_dir, config.data_set_name)

    if config.task == "task_1_tumor_vs_normal":
        dataset = Generic_MIL_Dataset(
            csv_path="dataset_csv/tumor_vs_normal_dummy_clean.csv",
            data_dir=data_dir,
            shuffle=False,
            print_info=True,
            label_dict={"normal_tissue": 0, "tumor_tissue": 1},
            patient_strat=False,
            ignore=[],
        )
    elif config.task == "task_2_tumor_subtyping":
        dataset = Generic_MIL_Dataset(
            csv_path="dataset_csv/tumor_subtyping_dummy_clean.csv",
            data_dir=data_dir,
            shuffle=False,
            print_info=True,
            label_dict={"subtype_1": 0, "subtype_2": 1, "subtype_3": 2},
            patient_strat=False,
            ignore=[],
        )
    else:
        raise NotImplementedError(f"Task {config.task} not implemented")

    return dataset


def create_detailed_metrics_artifacts(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
    config: EvalConfig,
    fold: int,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Create detailed evaluation metrics and artifacts for MLflow logging."""
    metrics_dict: Dict[str, Any] = {}
    artifacts: Dict[str, str] = {}

    try:
        # Classification report
        class_report: Dict[str, Any] = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )
        metrics_dict["classification_report"] = class_report

        # Log per-class metrics
        for class_idx in range(config.n_classes):
            if str(class_idx) in class_report:
                class_metrics: Dict[str, float] = class_report[str(class_idx)]
                metrics_dict[f"class_{class_idx}_precision"] = class_metrics[
                    "precision"
                ]
                metrics_dict[f"class_{class_idx}_recall"] = class_metrics["recall"]
                metrics_dict[f"class_{class_idx}_f1_score"] = class_metrics["f1-score"]
                metrics_dict[f"class_{class_idx}_support"] = class_metrics["support"]

        # Overall metrics
        metrics_dict["overall_accuracy"] = class_report["accuracy"]
        metrics_dict["macro_avg_precision"] = class_report["macro avg"]["precision"]
        metrics_dict["macro_avg_recall"] = class_report["macro avg"]["recall"]
        metrics_dict["macro_avg_f1_score"] = class_report["macro avg"]["f1-score"]
        metrics_dict["weighted_avg_precision"] = class_report["weighted avg"][
            "precision"
        ]
        metrics_dict["weighted_avg_recall"] = class_report["weighted avg"]["recall"]
        metrics_dict["weighted_avg_f1_score"] = class_report["weighted avg"]["f1-score"]

        # Confusion matrix
        cm: np.ndarray = confusion_matrix(all_labels, all_preds)
        metrics_dict["confusion_matrix"] = cm.tolist()

        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        cm_path: str = os.path.join(
            config.save_dir, f"confusion_matrix_fold_{fold}.png"
        )
        plt.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close()
        artifacts["confusion_matrix"] = cm_path

        # ROC curve data for binary classification
        if config.n_classes == 2:
            from sklearn.metrics import roc_curve

            fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
            roc_data: Dict[str, List[float]] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
            metrics_dict["roc_curve"] = roc_data

            # Create ROC curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - Fold {fold}")
            plt.legend(loc="lower right")
            roc_path: str = os.path.join(config.save_dir, f"roc_curve_fold_{fold}.png")
            plt.savefig(roc_path, bbox_inches="tight", dpi=300)
            plt.close()
            artifacts["roc_curve"] = roc_path

        # Save detailed metrics as JSON
        metrics_json_path: str = os.path.join(
            config.save_dir, f"detailed_metrics_fold_{fold}.json"
        )
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        artifacts["detailed_metrics"] = metrics_json_path

    except Exception as e:
        print(f"Warning: Could not create detailed metrics for fold {fold}: {e}")

    return metrics_dict, artifacts


def log_best_model_to_mlflow(
    config: EvalConfig,
    all_auc: List[float],
    all_acc: List[float],
    all_detailed_metrics: List[Dict[str, Any]],
    folds: List[int],
    ckpt_paths: List[str],
) -> Optional[torch.nn.Module]:
    """Identify the best model from evaluation results and log it to MLflow."""
    if not config.register_best_model:
        print("Model registration is disabled. Skipping...")
        return None

    # Find the best fold based on evaluation AUC
    best_fold_idx: int = np.argmax(all_auc)
    best_fold: int = folds[best_fold_idx]
    best_auc: float = all_auc[best_fold_idx]
    best_acc: float = all_acc[best_fold_idx]

    print(f"Best model from fold {best_fold} with {config.split} AUC: {best_auc:.4f}")

    # Load the best model checkpoint
    best_model_checkpoint_path: str = ckpt_paths[best_fold_idx]

    if not os.path.exists(best_model_checkpoint_path):
        print(
            f"Warning: Best model checkpoint not found at {best_model_checkpoint_path}"
        )
        return None

    try:
        # Define explicit requirements
        pip_reqs: List[str] = [
            "timm==0.9.8",
            "torch==2.8.0+cu128",
            "torchvision==0.23.0+cu128",
            "torchaudio==2.8.0+cu128",
            f"mlflow=={mlflow.__version__}",
            "numpy",
            "pandas",
        ]

        # Create dummy input for signature inference
        L: int = 500  # number of instances (patches)
        D: int = config.embed_dim
        dummy_input_np: np.ndarray = np.random.randn(L, D).astype(np.float32)
        dummy_input_torch: torch.Tensor = torch.from_numpy(dummy_input_np)

        # Reconstruct the model architecture and load weights
        best_model: torch.nn.Module = reconstruct_clam_model(
            config, best_model_checkpoint_path
        )

        # Generate signature by passing dummy input through the model
        with torch.no_grad():
            output: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = best_model(
                dummy_input_torch
            )
            Y_prob: torch.Tensor = output[0] if isinstance(output, tuple) else output

        Y_prob_np: np.ndarray = Y_prob.cpu().numpy()
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
        print(f"   - Best {config.split} AUC: {best_auc:.4f}")
        print(f"   - Best {config.split} Accuracy: {best_acc:.4f}")
        print(f"   - From fold: {best_fold}")

        # Log additional metrics for the best model
        mlflow.set_tag("best_model_fold", str(best_fold))
        mlflow.set_tag("best_model_source", "evaluation")
        mlflow.log_metric(f"best_{config.split}_auc", best_auc)
        mlflow.log_metric(f"best_{config.split}_accuracy", best_acc)

        # Log detailed metrics for the best model if available
        if all_detailed_metrics and best_fold_idx < len(all_detailed_metrics):
            best_detailed_metrics: Optional[Dict[str, Any]] = all_detailed_metrics[
                best_fold_idx
            ]
            if best_detailed_metrics:
                mlflow.log_metric(
                    f"best_{config.split}_precision",
                    best_detailed_metrics.get("macro_avg_precision", 0.0),
                )
                mlflow.log_metric(
                    f"best_{config.split}_recall",
                    best_detailed_metrics.get("macro_avg_recall", 0.0),
                )
                mlflow.log_metric(
                    f"best_{config.split}_f1_score",
                    best_detailed_metrics.get("macro_avg_f1_score", 0.0),
                )

        return best_model

    except Exception as e:
        print(f"Error logging model to MLflow: {e}")
        return None


def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    """Main evaluation function. Starts the MLflow parent run for multi-fold evaluation."""
    # MLFLOW INTEGRATION: Start the main run for multi-fold evaluation
    experiment_name: str = f"Eval_{config.models_exp_code}"
    mlflow.set_experiment(experiment_name)

    run_name: str = f"Aggregate_{config.split}_from_{config.models_exp_code}"
    with mlflow.start_run(run_name=run_name) as run:
        # Setup device
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Setup dataset
        dataset: Generic_MIL_Dataset = setup_eval_dataset(config)

        # Log settings
        settings: Dict[str, Any] = config.get_settings()

        # MLFLOW INTEGRATION: Log all parameters for the aggregate run
        mlflow.log_params(settings)

        log_file: str = os.path.join(
            config.save_dir, f"eval_experiment_{config.save_exp_code}.txt"
        )
        with open(log_file, "w") as f:
            print(settings, file=f)

        print("Evaluation Settings:")
        for key, val in settings.items():
            print(f"{key}:  {val}")

        # Get checkpoint paths
        ckpt_paths: List[str] = [
            os.path.join(config.models_dir, f"s_{fold}_checkpoint.pt")
            for fold in config.folds
        ]

        # Dataset split mapping
        datasets_id: Dict[str, int] = {"train": 0, "val": 1, "test": 2, "all": -1}

        # Run evaluation across folds
        all_results: List[Any] = []
        all_auc: List[float] = []
        all_acc: List[float] = []
        all_detailed_metrics: List[Optional[Dict[str, Any]]] = []

        for ckpt_idx, fold in enumerate(config.folds):
            # Get the appropriate dataset split
            if datasets_id[config.split] < 0:
                split_dataset: Generic_MIL_Dataset = dataset
            else:
                csv_path: str = f"{config.splits_dir}/splits_{fold}.csv"
                datasets: List[Generic_MIL_Dataset] = dataset.return_splits(
                    from_id=False, csv_path=csv_path
                )
                split_dataset = datasets[datasets_id[config.split]]

            # Run evaluation
            model, patient_results, test_error, auc, df, detailed_results = eval(
                split_dataset, config, ckpt_paths[ckpt_idx], fold
            )

            all_results.append(patient_results)
            all_auc.append(auc)
            all_acc.append(1.0 - test_error)

            # Extract and store detailed metrics for aggregation
            if detailed_results and "detailed_metrics" in detailed_results:
                all_detailed_metrics.append(detailed_results["detailed_metrics"])
            else:
                all_detailed_metrics.append(None)

            # Save individual fold results
            fold_results_path: str = os.path.join(config.save_dir, f"fold_{fold}.csv")
            df.to_csv(fold_results_path, index=False)

            # Log the individual fold CSV as an artifact in the *parent* run too
            mlflow.log_artifact(fold_results_path, artifact_path="fold_results")

        # MLFLOW INTEGRATION: Log the best model based on evaluation metrics
        best_model: Optional[torch.nn.Module] = None
        if config.register_best_model:
            best_model = log_best_model_to_mlflow(
                config,
                all_auc,
                all_acc,
                all_detailed_metrics,
                list(config.folds),
                ckpt_paths,
            )

        # Save summary
        final_df: pd.DataFrame = pd.DataFrame(
            {"folds": config.folds, "test_auc": all_auc, "test_acc": all_acc}
        )

        if len(config.folds) != config.k:
            save_name: str = f"summary_partial_{config.folds[0]}_{config.folds[-1]}.csv"
        else:
            save_name: str = "summary.csv"

        final_summary_path: str = os.path.join(config.save_dir, save_name)
        final_df.to_csv(final_summary_path)

        # MLFLOW INTEGRATION: Log final aggregate statistics and summary file
        mlflow.log_artifact(final_summary_path)
        mlflow.set_tag(
            "Evaluating Info", f"CLAM model evaluating with {config.data_set_name} data"
        )

        # Log aggregate metrics
        mlflow.log_metric(f"Aggregate_{config.split}_AUC_Mean", float(np.mean(all_auc)))
        mlflow.log_metric(f"Aggregate_{config.split}_AUC_Std", float(np.std(all_auc)))
        mlflow.log_metric(
            f"Aggregate_{config.split}_Accuracy_Mean", float(np.mean(all_acc))
        )
        mlflow.log_metric(
            f"Aggregate_{config.split}_Accuracy_Std", float(np.std(all_acc))
        )

        # Log detailed aggregate metrics if available
        if all_detailed_metrics and config.detailed_metrics:
            try:
                # Calculate aggregate detailed metrics
                all_precisions: List[float] = [
                    m.get("macro_avg_precision", 0.0)
                    for m in all_detailed_metrics
                    if m is not None
                ]
                all_recalls: List[float] = [
                    m.get("macro_avg_recall", 0.0)
                    for m in all_detailed_metrics
                    if m is not None
                ]
                all_f1_scores: List[float] = [
                    m.get("macro_avg_f1_score", 0.0)
                    for m in all_detailed_metrics
                    if m is not None
                ]

                if all_precisions:
                    mlflow.log_metric(
                        f"Aggregate_{config.split}_Precision_Mean",
                        float(np.mean(all_precisions)),
                    )
                    mlflow.log_metric(
                        f"Aggregate_{config.split}_Precision_Std",
                        float(np.std(all_precisions)),
                    )
                if all_recalls:
                    mlflow.log_metric(
                        f"Aggregate_{config.split}_Recall_Mean",
                        float(np.mean(all_recalls)),
                    )
                    mlflow.log_metric(
                        f"Aggregate_{config.split}_Recall_Std",
                        float(np.std(all_recalls)),
                    )
                if all_f1_scores:
                    mlflow.log_metric(
                        f"Aggregate_{config.split}_F1_Mean",
                        float(np.mean(all_f1_scores)),
                    )
                    mlflow.log_metric(
                        f"Aggregate_{config.split}_F1_Std", float(np.std(all_f1_scores))
                    )

            except Exception as e:
                print(f"Warning: Could not log aggregate detailed metrics: {e}")

        return {
            "all_results": all_results,
            "all_auc": all_auc,
            "all_acc": all_acc,
            "all_detailed_metrics": all_detailed_metrics,
            "final_df": final_df,
        }


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation."""
    parser = argparse.ArgumentParser(description="CLAM Evaluation Script")

    # Data parameters
    parser.add_argument(
        "--data_root_dir", type=str, default=None, help="data directory"
    )
    parser.add_argument("--data_set_name", type=str, default=None, help="dataset name")

    # Results and paths
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="relative path to results folder, i.e. "
        + "the directory containing models_exp_code relative to project root (default: ./results)",
    )
    parser.add_argument(
        "--save_exp_code",
        type=str,
        default=None,
        help="experiment code to save eval results",
    )
    parser.add_argument(
        "--models_exp_code",
        type=str,
        default=None,
        help="experiment code to load trained models (directory under results_dir containing model checkpoints",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=None,
        help="splits directory, if using custom splits other than what matches the task (default: None)",
    )

    # Model parameters
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "big"],
        default="small",
        help="size of model (default: small)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["clam_sb", "clam_mb", "mil"],
        default="clam_sb",
        help="type of model (default: clam_sb)",
    )
    parser.add_argument("--drop_out", type=float, default=0.25, help="dropout")
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument(
        "--subtyping", action="store_true", default=False, help="subtyping problem"
    )

    # Evaluation parameters
    parser.add_argument(
        "--k", type=int, default=10, help="number of folds (default: 10)"
    )
    parser.add_argument(
        "--k_start", type=int, default=-1, help="start fold (default: -1, last fold)"
    )
    parser.add_argument(
        "--k_end", type=int, default=-1, help="end fold (default: -1, first fold)"
    )
    parser.add_argument("--fold", type=int, default=-1, help="single fold to evaluate")
    parser.add_argument(
        "--micro_average",
        action="store_true",
        default=False,
        help="use micro_average instead of macro_avearge for multiclass AUC",
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test", "all"], default="test"
    )
    parser.add_argument(
        "--task", type=str, choices=["task_1_tumor_vs_normal", "task_2_tumor_subtyping"]
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
    parser.add_argument(
        "--no_detailed_metrics",
        action="store_true",
        default=False,
        help="Disable detailed metrics calculation and logging",
    )

    return parser


def main() -> Dict[str, Any]:
    """Main function for command line execution."""
    parser = create_parser()
    args = parser.parse_args()

    # Convert args to dictionary and create config
    config_dict: Dict[str, Any] = vars(args)
    config_dict["register_best_model"] = not args.no_register_model
    config_dict["detailed_metrics"] = not args.no_detailed_metrics

    config: EvalConfig = EvalConfig(**config_dict)

    # Run evaluation
    results: Dict[str, Any] = run_evaluation(config)
    print("Evaluation completed!")

    return results


if __name__ == "__main__":
    main()
