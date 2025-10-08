import numpy as np
import torch
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import os
import pandas as pd
from utils.utils import *
from utils.train_utils import AccuracyLogger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import mlflow
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
import json
from typing import Dict, Any, Tuple, Optional, List, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initiate_model(
    args: Any, ckpt_path: str, device: str = "cuda"
) -> Union[CLAM_SB, CLAM_MB, MIL_fc, MIL_fc_mc]:
    """Initialize model from checkpoint with proper configuration."""
    print("Init Model")

    model_dict: Dict[str, Any] = {
        "dropout": args.drop_out,
        "n_classes": args.n_classes,
        "embed_dim": args.embed_dim,
    }

    if args.model_size is not None and args.model_type in ["clam_sb", "clam_mb"]:
        model_dict.update({"size_arg": args.model_size})

    if args.model_type == "clam_sb":
        model = CLAM_SB(**model_dict)
    elif args.model_type == "clam_mb":
        model = CLAM_MB(**model_dict)
    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    # NOTE: print_network is assumed to be defined in utils.utils
    # print_network(model)

    ckpt: Dict[str, Any] = torch.load(ckpt_path)
    ckpt_clean: Dict[str, Any] = {}

    for key in ckpt.keys():
        if "instance_loss_fn" in key:
            continue
        ckpt_clean.update({key.replace(".module", ""): ckpt[key]})

    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model


def calculate_detailed_metrics(
    all_labels: np.ndarray, all_preds: np.ndarray, all_probs: np.ndarray, n_classes: int
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    metrics: Dict[str, Any] = {}

    try:
        # Basic metrics
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
            precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        )
        metrics["recall_weighted"] = float(
            recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        )

        # Per-class metrics
        for class_idx in range(n_classes):
            class_mask: np.ndarray = all_labels == class_idx
            if np.sum(class_mask) > 0:
                class_accuracy: float = float(
                    np.mean(all_preds[class_mask] == class_idx)
                )
                metrics[f"class_{class_idx}_accuracy"] = class_accuracy

                # For binary classification, calculate class-specific precision/recall
                if n_classes == 2:
                    binary_preds: np.ndarray = (all_preds == class_idx).astype(int)
                    binary_labels: np.ndarray = (all_labels == class_idx).astype(int)
                    if len(np.unique(binary_labels)) > 1:
                        metrics[f"class_{class_idx}_precision"] = float(
                            precision_score(
                                binary_labels, binary_preds, zero_division=0
                            )
                        )
                        metrics[f"class_{class_idx}_recall"] = float(
                            recall_score(binary_labels, binary_preds, zero_division=0)
                        )
                        metrics[f"class_{class_idx}_f1"] = float(
                            f1_score(binary_labels, binary_preds, zero_division=0)
                        )

        # Confidence metrics
        max_probs: np.ndarray = np.max(all_probs, axis=1)
        metrics["mean_confidence"] = float(np.mean(max_probs))
        metrics["confidence_std"] = float(np.std(max_probs))

        # Calibration metrics (simple version)
        correct_predictions: np.ndarray = all_preds == all_labels
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


def eval(dataset: Any, args: Any, ckpt_path: str, fold: Optional[int] = None) -> Tuple[
    Union[CLAM_SB, CLAM_MB, MIL_fc, MIL_fc_mc],  # model
    Dict[str, Any],  # patient_results
    float,  # test_error
    float,  # auc_score
    pd.DataFrame,  # df
    Optional[Dict[str, Any]],  # detailed_results
]:
    """
    Evaluates the model and logs final results to MLflow with detailed metrics.
    """
    # MLflow nested run for evaluation
    fold_suffix: str = f"_fold_{fold}" if fold is not None else ""
    eval_run_name: str = f"Evaluation{fold_suffix}: {os.path.basename(ckpt_path)}"

    with mlflow.start_run(run_name=eval_run_name, nested=True) as run:
        # Log essential parameters and the model checkpoint path
        mlflow.log_param("eval_ckpt_path", ckpt_path)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_classes", args.n_classes)
        if fold is not None:
            mlflow.log_param("fold", fold)

        # Initiate model and run summary
        model: Union[CLAM_SB, CLAM_MB, MIL_fc, MIL_fc_mc] = initiate_model(
            args, ckpt_path
        )

        print("Init Loaders")
        # NOTE: get_simple_loader is assumed to be defined in utils.utils
        loader: Any = get_simple_loader(dataset)
        patient_results, test_error, auc_score, df, acc_logger, detailed_results = (
            summary(model, loader, args)
        )

        test_acc: float = 1.0 - test_error  # Calculate test accuracy

        # Log basic metrics to MLflow
        mlflow.log_metric("test_error", test_error)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_auc", auc_score)

        # Log class-wise accuracy
        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            if acc is not None:
                mlflow.log_metric(f"test_class_{i}_acc", acc)
                mlflow.log_metric(f"test_class_{i}_correct", correct)
                mlflow.log_metric(f"test_class_{i}_total", count)

        # Log detailed metrics if available
        if detailed_results and args.detailed_metrics:
            try:
                # Log comprehensive metrics
                detailed_metrics: Dict[str, Any] = detailed_results.get(
                    "detailed_metrics", {}
                )
                for metric_name, metric_value in detailed_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"test_{metric_name}", metric_value)

                # Log per-class detailed metrics
                for class_idx in range(args.n_classes):
                    for metric_type in ["precision", "recall", "f1"]:
                        metric_key: str = f"class_{class_idx}_{metric_type}"
                        if metric_key in detailed_metrics:
                            mlflow.log_metric(
                                f"test_{metric_key}", detailed_metrics[metric_key]
                            )

                # Log confidence metrics
                confidence_metrics: List[str] = [
                    "mean_confidence",
                    "confidence_std",
                    "avg_confidence_correct",
                    "avg_confidence_incorrect",
                ]
                for metric in confidence_metrics:
                    if metric in detailed_metrics:
                        mlflow.log_metric(f"test_{metric}", detailed_metrics[metric])

            except Exception as e:
                print(f"Warning: Could not log detailed metrics: {e}")

        # Log artifacts
        # Log the results DataFrame (predictions, probabilities) as a CSV artifact
        results_path: str = os.path.join(
            os.path.dirname(ckpt_path), f"{eval_run_name}_results.csv"
        )
        df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        # Log detailed metrics JSON if available
        if detailed_results and "metrics_json_path" in detailed_results:
            mlflow.log_artifact(detailed_results["metrics_json_path"])

        # Log visualization artifacts if available
        if detailed_results and "artifacts" in detailed_results:
            for artifact_name, artifact_path in detailed_results["artifacts"].items():
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_path="evaluation_plots")

        print(f"test_error: {test_error}")
        print(f"auc: {auc_score}")

    return model, patient_results, test_error, auc_score, df, detailed_results


def summary(
    model: Union[CLAM_SB, CLAM_MB, MIL_fc, MIL_fc_mc], loader: Any, args: Any
) -> Tuple[
    Dict[str, Any],  # patient_results
    float,  # test_error
    float,  # auc_score
    pd.DataFrame,  # df
    AccuracyLogger,  # acc_logger
    Optional[Dict[str, Any]],  # detailed_results
]:
    """Run model inference on loader and compute comprehensive evaluation metrics."""
    acc_logger: AccuracyLogger = AccuracyLogger(n_classes=args.n_classes)
    model.eval()
    test_loss: float = 0.0
    test_error: float = 0.0

    all_probs: np.ndarray = np.zeros((len(loader), args.n_classes))
    all_labels: np.ndarray = np.zeros(len(loader))
    all_preds: np.ndarray = np.zeros(len(loader))

    slide_ids: pd.Series = loader.dataset.slide_data["slide_id"]
    patient_results: Dict[str, Any] = {}

    # NOTE: device is assumed to be defined globally as 'cuda' or 'cpu'
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id: str = slide_ids.iloc[batch_idx]

            # NOTE: model returns logits, Y_prob, Y_hat, _, results_dict
            logits, Y_prob, Y_hat, _, results_dict = model(data)

            acc_logger.log(Y_hat, label)

            probs: np.ndarray = Y_prob.cpu().numpy()

            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()

            patient_results.update(
                {
                    slide_id: {
                        "slide_id": np.array(slide_id),
                        "prob": probs,
                        "label": label.item(),
                    }
                }
            )

            # NOTE: calculate_error is assumed to be defined in utils.utils
            error: float = calculate_error(Y_hat, label)
            test_error += error

    del data
    test_error /= len(loader)

    aucs: List[float] = []
    auc_score: float

    if len(np.unique(all_labels)) == 1:
        auc_score = -1.0
    else:
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels: np.ndarray = label_binarize(
                all_labels, classes=[i for i in range(args.n_classes)]
            )
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(
                        binary_labels[:, class_idx], all_probs[:, class_idx]
                    )
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float("nan"))

            if args.micro_average:  # Assumes args.micro_average exists
                binary_labels = label_binarize(
                    all_labels, classes=[i for i in range(args.n_classes)]
                )
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    # Calculate detailed metrics if enabled
    detailed_results: Optional[Dict[str, Any]] = None
    if args.detailed_metrics:
        try:
            from eval import create_detailed_metrics_artifacts

            # Get fold from args if available
            fold: Optional[int] = getattr(args, "fold", None)
            if fold is None:
                fold = 0  # Default fold value

            detailed_metrics, artifacts = create_detailed_metrics_artifacts(
                all_labels, all_preds, all_probs, args, fold
            )

            # Calculate additional comprehensive metrics
            comprehensive_metrics: Dict[str, Any] = calculate_detailed_metrics(
                all_labels, all_preds, all_probs, args.n_classes
            )
            detailed_metrics.update(comprehensive_metrics)

            # Save comprehensive metrics as JSON
            metrics_json_path: str = os.path.join(
                args.save_dir, f"comprehensive_metrics_fold_{fold}.json"
            )
            with open(metrics_json_path, "w") as f:
                json.dump(detailed_metrics, f, indent=2)
            artifacts["comprehensive_metrics"] = metrics_json_path

            detailed_results = {
                "detailed_metrics": detailed_metrics,
                "artifacts": artifacts,
                "metrics_json_path": metrics_json_path,
            }

        except Exception as e:
            print(f"Warning: Could not generate detailed metrics: {e}")
            detailed_results = None

    # Create results DataFrame
    results_dict: Dict[str, Any] = {
        "slide_id": slide_ids,
        "Y": all_labels,
        "Y_hat": all_preds,
    }
    for c in range(args.n_classes):
        results_dict.update({f"p_{c}": all_probs[:, c]})

    df: pd.DataFrame = pd.DataFrame(results_dict)

    return patient_results, test_error, auc_score, df, acc_logger, detailed_results
