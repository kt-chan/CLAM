import numpy as np
import torch
import torch.nn as nn  # Added import for nn
from utils.utils import *
import os
import mlflow  # <-- ADDED MLflow import
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += Y_hat == Y

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name="checkpoint.pt"):

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

    def save_checkpoint(self, val_loss, model, ckpt_name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """
    train for a single fold, with MLflow tracking.
    """
    # ====================================================================
    # MLFLOW INTEGRATION: Start nested run for the current fold
    # ====================================================================
    with mlflow.start_run(run_name=f"Fold {cur}", nested=True) as run:

        # Log Hyperparameters/Settings
        mlflow.log_params(
            {
                "fold": cur,
                "max_epochs": args.max_epochs,
                "lr": args.lr,
                "model_type": args.model_type,
                "bag_loss": args.bag_loss,
                "drop_out": args.drop_out,
                "n_classes": args.n_classes,
            }
        )

        print("\nTraining Fold {}!".format(cur))
        writer_dir = os.path.join(args.results_dir, str(cur))
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)

        if args.log_data:
            from tensorboardX import SummaryWriter

            writer = SummaryWriter(writer_dir, flush_secs=15)
        else:
            writer = None

        print("\nInit train/val/test splits...", end=" ")
        train_split, val_split, test_split = datasets
        split_csv_path = os.path.join(args.results_dir, "splits_{}.csv".format(cur))
        save_splits(
            datasets,
            ["train", "val", "test"],
            split_csv_path,
        )
        # Log splits CSV as an artifact
        mlflow.log_artifact(split_csv_path)

        print("Done!")
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

        print("\nInit loss function...", end=" ")
        if args.bag_loss == "svm":
            from topk.svm import SmoothTop1SVM

            loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
            if device.type == "cuda":
                loss_fn = loss_fn.cuda()
        else:
            loss_fn = nn.CrossEntropyLoss()
        print("Done!")

        print("\nInit Model...", end=" ")
        model_dict = {
            "dropout": args.drop_out,
            "n_classes": args.n_classes,
            "embed_dim": args.embed_dim,
        }

        if args.model_size is not None and args.model_type != "mil":
            model_dict.update({"size_arg": args.model_size})

        if args.model_type in ["clam_sb", "clam_mb"]:
            if args.subtyping:
                model_dict.update({"subtyping": True})

            if args.B > 0:
                model_dict.update({"k_sample": args.B})

            if args.inst_loss == "svm":
                from topk.svm import SmoothTop1SVM

                instance_loss_fn = SmoothTop1SVM(n_classes=2)
                if device.type == "cuda":
                    instance_loss_fn = instance_loss_fn.cuda()
            else:
                instance_loss_fn = nn.CrossEntropyLoss()

            if args.model_type == "clam_sb":
                model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            elif args.model_type == "clam_mb":
                model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
            else:
                raise NotImplementedError

        else:  # args.model_type == 'mil'
            if args.n_classes > 2:
                model = MIL_fc_mc(**model_dict)
            else:
                model = MIL_fc(**model_dict)

        _ = model.to(device)
        print("Done!")

        # NOTE: print_network is assumed to be defined in utils.utils
        # print_network(model)

        print("\nInit optimizer ...", end=" ")
        # NOTE: get_optim is assumed to be defined in utils.utils
        optimizer = get_optim(model, args)
        print("Done!")

        print("\nInit Loaders...", end=" ")
        # NOTE: get_split_loader is assumed to be defined in utils.utils
        train_loader = get_split_loader(
            train_split,
            training=True,
            testing=args.testing,
            weighted=args.weighted_sample,
        )
        val_loader = get_split_loader(val_split, testing=args.testing)
        test_loader = get_split_loader(test_split, testing=args.testing)
        print("Done!")

        print("\nSetup EarlyStopping...", end=" ")
        if args.early_stopping:
            early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)

        else:
            early_stopping = None
        print("Done!")

        ckpt_path = os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))

        for epoch in range(args.max_epochs):
            print("#" * 128)
            print(f"running training for epoch: {epoch} / {args.max_epochs} ... ")
            print("#" * 128)

            # NOTE: train_loop_clam/train_loop and validate_clam/validate are called
            # and now contain MLflow logging inside their definitions (updated below)
            if args.model_type in ["clam_sb", "clam_mb"] and not args.no_inst_cluster:
                # The train_loop_clam now logs training metrics to MLflow
                train_loop_clam(
                    epoch,
                    model,
                    train_loader,
                    optimizer,
                    args.n_classes,
                    args.bag_weight,
                    writer,
                    loss_fn,
                )
                # The validate_clam now logs validation metrics to MLflow
                stop = validate_clam(
                    cur,
                    epoch,
                    model,
                    val_loader,
                    args.n_classes,
                    early_stopping,
                    writer,
                    loss_fn,
                    args.results_dir,
                )

            else:
                # The train_loop now logs training metrics to MLflow
                train_loop(
                    epoch,
                    model,
                    train_loader,
                    optimizer,
                    args.n_classes,
                    writer,
                    loss_fn,
                )
                # The validate now logs validation metrics to MLflow
                stop = validate(
                    cur,
                    epoch,
                    model,
                    val_loader,
                    args.n_classes,
                    early_stopping,
                    writer,
                    loss_fn,
                    args.results_dir,
                )

            if stop:
                break

        # Load best/final model
        if args.early_stopping:
            model.load_state_dict(torch.load(ckpt_path))
        else:
            torch.save(
                model.state_dict(),
                ckpt_path,
            )

        # Log the final checkpoint artifact
        mlflow.log_artifact(ckpt_path)

        # Final Evaluation (Note: summary is assumed to be defined in utils.utils)
        # Summary returns patient_results, error, auc, acc_logger
        _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
        print("Val error: {:.4f}, ROC AUC: {:.4f}".format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger = summary(
            model, test_loader, args.n_classes
        )
        print("Test error: {:.4f}, ROC AUC: {:.4f}".format(test_error, test_auc))

        # ====================================================================
        # MLFLOW INTEGRATION: Log Final Metrics
        # ====================================================================
        mlflow.log_metric(
            "final_val_loss", val_error
        )  # Original code uses error as loss
        mlflow.log_metric("final_val_auc", val_auc)
        mlflow.log_metric("final_test_loss", test_error)
        mlflow.log_metric("final_test_auc", test_auc)
        mlflow.log_metric("final_val_acc", 1 - val_error)
        mlflow.log_metric("final_test_acc", 1 - test_error)

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print("class {}: acc {}, correct {}/{}".format(i, acc, correct, count))
            if acc is not None:
                mlflow.log_metric(f"final_test_class_{i}_acc", acc)

        if writer:
            writer.close()

        # Return values match the original function signature
        return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error


def train_loop_clam(
    epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None
):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.0
    train_error = 0.0
    train_inst_loss = 0.0
    inst_count = 0

    print("\n")
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # NOTE: model is assumed to be CLAM_MB/CLAM_SB
        logits, Y_prob, Y_hat, _, instance_dict = model(
            data, label=label, instance_eval=True
        )

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict["instance_loss"]
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = instance_dict["inst_preds"]
        inst_labels = instance_dict["inst_labels"]
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print(
                "batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, ".format(
                    batch_idx, loss_value, instance_loss_value, total_loss.item()
                )
                + "label: {}, bag_size: {}".format(label.item(), data.size(0))
            )

        # NOTE: calculate_error is assumed to be defined in utils.utils
        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print("\n")
        # MLflow Log Train Instance Metrics (Clustering Acc)
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print(
                "class {} clustering acc {}: correct {}/{}".format(
                    i, acc, correct, count
                )
            )
            if acc is not None:
                mlflow.log_metric(f"train_inst_cluster_class_{i}_acc", acc, step=epoch)

    print(
        "Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}".format(
            epoch, train_loss, train_inst_loss, train_error
        )
    )

    # MLflow Log Train Metrics
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_error", train_error, step=epoch)
    mlflow.log_metric("train_clustering_loss", train_inst_loss, step=epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print("class {}: acc {}, correct {}/{}".format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar("train/class_{}_acc".format(i), acc, epoch)
        if acc is not None:
            mlflow.log_metric(f"train_class_{i}_acc", acc, step=epoch)

    if writer:
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/error", train_error, epoch)
        writer.add_scalar("train/clustering_loss", train_inst_loss, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.0
    train_error = 0.0

    print("\n")
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print(
                "batch {}, loss: {:.4f}, label: {}, bag_size: {}".format(
                    batch_idx, loss_value, label.item(), data.size(0)
                )
            )

        # NOTE: calculate_error is assumed to be defined in utils.utils
        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print(
        "Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}".format(
            epoch, train_loss, train_error
        )
    )

    # MLflow Log Train Metrics
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_error", train_error, step=epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print("class {}: acc {}, correct {}/{}".format(i, acc, correct, count))
        if writer:
            writer.add_scalar("train/class_{}_acc".format(i), acc, epoch)
        if acc is not None:
            mlflow.log_metric(f"train_class_{i}_acc", acc, step=epoch)

    if writer:
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/error", train_error, epoch)


def validate(
    cur,
    epoch,
    model,
    loader,
    n_classes,
    early_stopping=None,
    writer=None,
    loss_fn=None,
    results_dir=None,
):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.0
    val_error = 0.0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(
                device, non_blocking=True
            )

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            # NOTE: calculate_error is assumed to be defined in utils.utils
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class="ovr")

    # MLflow Log Validation Metrics
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_error", val_error, step=epoch)
    mlflow.log_metric("val_auc", auc, step=epoch)

    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/auc", auc, epoch)
        writer.add_scalar("val/error", val_error, epoch)

    print(
        "\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}".format(
            val_loss, val_error, auc
        )
    )
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print("class {}: acc {}, correct {}/{}".format(i, acc, correct, count))
        if acc is not None:
            mlflow.log_metric(f"val_class_{i}_acc", acc, step=epoch)

    if early_stopping:
        assert results_dir
        early_stopping(
            epoch,
            val_loss,
            model,
            ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
        )

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_clam(
    cur,
    epoch,
    model,
    loader,
    n_classes,
    early_stopping=None,
    writer=None,
    loss_fn=None,
    results_dir=None,
):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.0
    val_error = 0.0

    val_inst_loss = 0.0
    val_inst_acc = 0.0
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    # sample_size = model.k_sample # This is not used but kept in original code
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            # NOTE: model is assumed to be CLAM_MB/CLAM_SB
            logits, Y_prob, Y_hat, _, instance_dict = model(
                data, label=label, instance_eval=True
            )
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

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

            # NOTE: calculate_error is assumed to be defined in utils.utils
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float("nan"))

        auc = np.nanmean(np.array(aucs))

    # MLflow Log Validation Metrics
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_error", val_error, step=epoch)
    mlflow.log_metric("val_auc", auc, step=epoch)

    print(
        "\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}".format(
            val_loss, val_error, auc
        )
    )
    if inst_count > 0:
        val_inst_loss /= inst_count
        mlflow.log_metric("val_inst_loss", val_inst_loss, step=epoch)

        # MLflow Log Val Instance Metrics (Clustering Acc)
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print(
                "class {} clustering acc {}: correct {}/{}".format(
                    i, acc, correct, count
                )
            )
            if acc is not None:
                mlflow.log_metric(f"val_inst_cluster_class_{i}_acc", acc, step=epoch)

    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/auc", auc, epoch)
        writer.add_scalar("val/error", val_error, epoch)
        writer.add_scalar("val/inst_loss", val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print("class {}: acc {}, correct {}/{}".format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar("val/class_{}_acc".format(i), acc, epoch)
        if acc is not None:
            mlflow.log_metric(f"val_class_{i}_acc", acc, step=epoch)

    if early_stopping:
        assert results_dir
        early_stopping(
            epoch,
            val_loss,
            model,
            ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
        )

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
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
