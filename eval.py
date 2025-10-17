from __future__ import print_function
from typing import Optional, Dict, List, Any, Tuple, Union
import json

import numpy as np
import argparse
import torch
import os
import pandas as pd

from utils.utils import *
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import (
    Generic_MIL_Dataset,
    Generic_WSI_Classification_Dataset,
    Generic_WSI_Regression_Dataset,
)
from utils.eval_utils import *

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def setup_eval_dataset(config: EvalConfig) -> Generic_MIL_Dataset:
    """Setup dataset for evaluation based on configuration."""
    data_dir: str = os.path.join(config.data_root_dir, config.data_set_name)

    if config.task == TaskType.BINARY:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path="dataset_csv/tumor_vs_normal_dummy_clean.csv",
            data_dir=data_dir,
            shuffle=False,
            print_info=True,
            label_dict={"normal_tissue": 0, "tumor_tissue": 1},
            patient_strat=False,
            ignore=[],
        )
    elif config.task == TaskType.MULTICLASS:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path="dataset_csv/tumor_subtyping_dummy_clean.csv",
            data_dir=data_dir,
            shuffle=False,
            print_info=True,
            label_dict={"subtype_1": 0, "subtype_2": 1, "subtype_3": 2},
            patient_strat=False,
            ignore=[],
        )
    elif config.task == TaskType.REGRESSION:
        dataset = Generic_WSI_Regression_Dataset(
            csv_path="dataset_csv/tumor_regression_dummy_clean.csv",
            data_dir=data_dir,
            shuffle=False,
            print_info=True,
            label_dict={},
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
        "--task",
        type=str,
        choices=[
            "task_1_tumor_vs_normal",
            "task_2_tumor_subtyping",
            "task_3_tumor_count",
        ],
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
    # Setup dataset
    dataset: Generic_MIL_Dataset = setup_eval_dataset(config)

    # Run evaluation
    results: Dict[str, Any] = run_evaluation(config, dataset)
    print("Evaluation completed!")

    return results


if __name__ == "__main__":
    main()
