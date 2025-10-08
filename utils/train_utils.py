from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import mlflow
import numpy as np
import torch
import torch.nn as nn
from dataset_modules.dataset_generic import save_splits
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from utils.utils import calculate_error, get_optim, get_split_loader, seed_torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience: int = 20, stop_epoch: int = 50, verbose: bool = False) -> None:
        """
        Args:
            patience: How long to wait after last time validation loss improved
            stop_epoch: Earliest epoch possible for stopping
            verbose: If True, prints a message for each validation loss improvement
        """
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
        ckpt_name: str = "checkpoint.pt"
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

    def save_checkpoint(self, val_loss: float, model: nn.Module, ckpt_name: str) -> None:
        """Saves model when validation loss decreases"""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class TrainingResults(TypedDict):
    """Type definition for training results"""
    results_dict: Dict[str, Any]
    test_auc: float
    val_auc: float
    test_acc: float
    val_acc: float


class ModelDict(TypedDict, total=False):
    """Type definition for model configuration dictionary"""
    dropout: float
    n_classes: int
    embed_dim: int
    size_arg: str
    subtyping: bool
    k_sample: int


def train(
    datasets: Tuple[Any, Any, Any], 
    cur: int, 
    args: Any
) -> Tuple[Dict[str, Any], float, float, float, float]:
    """
    Train for a single fold with MLflow tracking.
    
    Args:
        datasets: Tuple of (train_split, val_split, test_split)
        cur: Current fold number
        args: Training configuration arguments
        
    Returns:
        Tuple of (results_dict, test_auc, val_auc, test_acc, val_acc)
    """
    with mlflow.start_run(run_name=f"Fold {cur}", nested=True) as run:
        # Log hyperparameters/settings
        mlflow.log_params({
            "fold": cur,
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "model_type": args.model_type,
            "bag_loss": args.bag_loss,
            "drop_out": args.drop_out,
            "n_classes": args.n_classes,
        })

        print(f"\nTraining Fold {cur}!")
        writer_dir = os.path.join(args.results_dir, str(cur))
        os.makedirs(writer_dir, exist_ok=True)

        if args.log_data:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(writer_dir, flush_secs=15)
        else:
            writer = None

        print("\nInit train/val/test splits...", end=" ")
        train_split, val_split, test_split = datasets
        split_csv_path = os.path.join(args.results_dir, f"splits_{cur}.csv")
        save_splits(datasets, ["train", "val", "test"], split_csv_path)
        mlflow.log_artifact(split_csv_path)
        print("Done!")

        print(f"Training on {len(train_split)} samples")
        print(f"Validating on {len(val_split)} samples") 
        print(f"Testing on {len(test_split)} samples")

        # Initialize loss function
        print("\nInit loss function...", end=" ")
        loss_fn = _setup_loss_function(args)
        print("Done!")

        # Initialize model
        print("\nInit Model...", end=" ")
        model = _setup_model(args, loss_fn)
        print("Done!")

        # Initialize optimizer
        print("\nInit optimizer ...", end=" ")
        optimizer = get_optim(model, args)
        print("Done!")

        # Initialize data loaders
        print("\nInit Loaders...", end=" ")
        train_loader, val_loader, test_loader = _setup_data_loaders(args, train_split, val_split, test_split)
        print("Done!")

        # Setup early stopping
        print("\nSetup EarlyStopping...", end=" ")
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True) if args.early_stopping else None
        print("Done!")

        ckpt_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")

        # Training loop
        stop_training = False
        for epoch in range(args.max_epochs):
            print("#" * 128)
            print(f"Running training for epoch: {epoch} / {args.max_epochs} ...")
            print("#" * 128)

            if args.model_type in ["clam_sb", "clam_mb"] and not args.no_inst_cluster:
                train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
                stop_training = validate_clam(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)
            else:
                train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
                stop_training = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)

            if stop_training:
                break

        # Load best/final model
        if args.early_stopping and early_stopping:
            model.load_state_dict(torch.load(ckpt_path))
        else:
            torch.save(model.state_dict(), ckpt_path)

        mlflow.log_artifact(ckpt_path)

        # Final evaluation
        _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
        print(f"Val error: {val_error:.4f}, ROC AUC: {val_auc:.4f}")

        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print(f"Test error: {test_error:.4f}, ROC AUC: {test_auc:.4f}")

        # Log final metrics
        _log_final_metrics(val_error, val_auc, test_error, test_auc, acc_logger, args.n_classes)

        if writer:
            writer.close()

        return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error


def _setup_loss_function(args: Any) -> nn.Module:
    """Setup loss function based on configuration"""
    if args.bag_loss == "svm":
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == "cuda":
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def _setup_model(args: Any, loss_fn: nn.Module) -> nn.Module:
    """Setup model based on configuration"""
    model_dict: ModelDict = {
        "dropout": args.drop_out,
        "n_classes": args.n_classes,
        "embed_dim": args.embed_dim,
    }

    if args.model_size is not None and args.model_type != "mil":
        model_dict["size_arg"] = args.model_size

    if args.model_type in ["clam_sb", "clam_mb"]:
        if args.subtyping:
            model_dict["subtyping"] = True

        if args.B > 0:
            model_dict["k_sample"] = args.B

        instance_loss_fn = _setup_instance_loss_fn(args)

        if args.model_type == "clam_sb":
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == "clam_mb":
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError(f"Model type {args.model_type} not implemented")
    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    model = model.to(device)
    return model


def _setup_instance_loss_fn(args: Any) -> nn.Module:
    """Setup instance loss function for CLAM models"""
    if args.inst_loss == "svm":
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes=2)
        if device.type == "cuda":
            instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()
    return instance_loss_fn


def _setup_data_loaders(
    args: Any, 
    train_split: Any, 
    val_split: Any, 
    test_split: Any
) -> Tuple[Any, Any, Any]:
    """Setup data loaders for training, validation, and testing"""
    train_loader = get_split_loader(
        train_split, training=True, testing=args.testing, weighted=args.weighted_sample
    )
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    return train_loader, val_loader, test_loader


def _log_final_metrics(
    val_error: float, 
    val_auc: float, 
    test_error: float, 
    test_auc: float, 
    acc_logger: AccuracyLogger, 
    n_classes: int
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


def train_loop_clam(
    epoch: int,
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    n_classes: int,
    bag_weight: float,
    writer: Optional[Any] = None,
    loss_fn: Optional[nn.Module] = None,
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
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        # Calculate losses
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label) if loss_fn else nn.CrossEntropyLoss()(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict["instance_loss"]
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        # Log instance predictions
        inst_preds = instance_dict["inst_preds"]
        inst_labels = instance_dict["inst_labels"]
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print(
                f"batch {batch_idx}, loss: {loss_value:.4f}, instance_loss: {instance_loss_value:.4f}, "
                f"weighted_loss: {total_loss.item():.4f}, label: {label.item()}, bag_size: {data.size(0)}"
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
                mlflow.log_metric(f"train_inst_cluster_class_{i}_acc", acc, step=epoch)

    print(
        f"Epoch: {epoch}, train_loss: {train_loss:.4f}, "
        f"train_clustering_loss: {train_inst_loss:.4f}, train_error: {train_error:.4f}"
    )

    # Log training metrics
    _log_training_metrics(epoch, train_loss, train_error, train_inst_loss, acc_logger, n_classes, writer)


def train_loop(
    epoch: int,
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    n_classes: int,
    writer: Optional[Any] = None,
    loss_fn: Optional[nn.Module] = None,
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
        loss = loss_fn(logits, label) if loss_fn else nn.CrossEntropyLoss()(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print(f"batch {batch_idx}, loss: {loss_value:.4f}, label: {label.item()}, bag_size: {data.size(0)}")

        error = calculate_error(Y_hat, label)
        train_error += error

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate epoch statistics
    train_loss /= len(loader)
    train_error /= len(loader)

    print(f"Epoch: {epoch}, train_loss: {train_loss:.4f}, train_error: {train_error:.4f}")

    # Log training metrics
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_error", train_error, step=epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f"class {i}: acc {acc}, correct {correct}/{count}")
        if writer:
            writer.add_scalar(f"train/class_{i}_acc", acc, epoch)
        if acc is not None:
            mlflow.log_metric(f"train_class_{i}_acc", acc, step=epoch)

    if writer:
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/error", train_error, epoch)


def validate(
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

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label) if loss_fn else nn.CrossEntropyLoss()(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    # Calculate AUC
    auc = _calculate_auc(n_classes, labels, prob)

    # Log validation metrics
    _log_validation_metrics(epoch, val_loss, val_error, auc, acc_logger, n_classes, writer)

    print(f"\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}")

    # Early stopping
    if early_stopping and results_dir:
        early_stopping(
            epoch, val_loss, model, 
            ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        )
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_clam(
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

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label) if loss_fn else nn.CrossEntropyLoss()(logits, label)
            val_loss += loss.item()

            instance_loss = instance_dict["instance_loss"]
            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict["inst_preds"]
            inst_labels = instance_dict["inst_labels"]
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    # Calculate AUC
    auc = _calculate_auc(n_classes, labels, prob)

    # Log validation metrics
    _log_validation_metrics(epoch, val_loss, val_error, auc, acc_logger, n_classes, writer)

    # Log instance metrics
    if inst_count > 0:
        val_inst_loss /= inst_count
        mlflow.log_metric("val_inst_loss", val_inst_loss, step=epoch)

        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print(f"class {i} clustering acc {acc}: correct {correct}/{count}")
            if acc is not None:
                mlflow.log_metric(f"val_inst_cluster_class_{i}_acc", acc, step=epoch)

    print(f"\nVal Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}")

    if writer:
        writer.add_scalar("val/inst_loss", val_inst_loss, epoch)

    # Early stopping
    if early_stopping and results_dir:
        early_stopping(
            epoch, val_loss, model,
            ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt")
        )
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def _calculate_auc(n_classes: int, labels: np.ndarray, prob: np.ndarray) -> float:
    """Calculate AUC score based on number of classes"""
    if n_classes == 2:
        return roc_auc_score(labels, prob[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=list(range(n_classes)))
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float("nan"))
        return np.nanmean(np.array(aucs))


def _log_training_metrics(
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


def _log_validation_metrics(
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

def summary(model, loader, n_classes):
    acc_logger = AccuracyLogger(n_classes=n_classes)
    model.eval()
    test_loss = 0.0  # This variable is unused in your original summary function
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
        # NOTE: calculate_error is assumed to be defined in utils.utils
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(
            all_labels, classes=[i for i in range(n_classes)]
        )
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(
                    binary_labels[:, class_idx], all_probs[:, class_idx]
                )
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float("nan"))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger
