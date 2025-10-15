import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import (
    DataLoader,
    Sampler,
    WeightedRandomSampler,
    RandomSampler,
    SequentialSampler,
    sampler,
)
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL_classification(batch):
    """Collate function for classification tasks"""
    img = torch.cat([item[0] for item in batch], dim=0)
    # For classification, use LongTensor
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_MIL_regression(batch):
    """Collate function for regression tasks"""
    img = torch.cat([item[0] for item in batch], dim=0)
    # For regression, use FloatTensor and handle tuple labels
    labels = []
    for item in batch:
        if isinstance(item[1], tuple):
            # Convert tuple to tensor [primary, secondary]
            labels.append(torch.FloatTensor([item[1][0], item[1][1]]))
        else:
            # Single value regression
            labels.append(torch.FloatTensor([float(item[1])]))
    label = torch.stack(labels)
    return [img, label]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_collate_fn(task=TaskType.BINARY):
    """Get appropriate collate function based on task type"""
    if task == TaskType.REGRESSION:
        return collate_MIL_regression
    elif task == TaskType.BINARY or task == TaskType.MULTICLASS:
        return collate_MIL_classification
    else:
        raise Exception("Not supported task exception!")


def get_split_loader(
    split_dataset, training=False, testing=False, weighted=False, task=TaskType.BINARY
):
    """
    Return either the validation loader or training loader
    """
    kwargs = {"num_workers": 4} if device.type == "cuda" else {}
    collate_fn = get_collate_fn(task)

    if not testing:
        if training:
            if weighted and task != TaskType.REGRESSION:
                # Weighted sampling only makes sense for classification tasks
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(
                    split_dataset,
                    batch_size=1,
                    sampler=WeightedRandomSampler(weights, len(weights)),
                    collate_fn=collate_fn,
                    **kwargs,
                )
            else:
                loader = DataLoader(
                    split_dataset,
                    batch_size=1,
                    sampler=RandomSampler(split_dataset),
                    collate_fn=collate_fn,
                    **kwargs,
                )
        else:
            loader = DataLoader(
                split_dataset,
                batch_size=1,
                sampler=SequentialSampler(split_dataset),
                collate_fn=collate_fn,
                **kwargs,
            )
    else:
        ids = np.random.choice(
            np.arange(len(split_dataset)), int(len(split_dataset) * 0.1), replace=False
        )
        loader = DataLoader(
            split_dataset,
            batch_size=1,
            sampler=SubsetSequentialSampler(ids),
            collate_fn=collate_fn,
            **kwargs,
        )

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.reg,
        )
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.reg,
        )
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print("Total number of parameters: %d" % num_params)
    print("Total number of trainable parameters: %d" % num_params_train)


def generate_split(
    cls_ids,
    val_num,
    test_num,
    samples,
    n_splits=5,
    seed=7,
    label_frac=1.0,
    custom_test_ids=None,
):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(
                cls_ids[c], indices
            )  # all indices of this class
            remaining_ids = possible_indices

            if len(possible_indices) > 0:
                if len(possible_indices) >= val_num[c]:
                    val_ids = np.random.choice(
                        possible_indices, val_num[c], replace=False
                    )
                    remaining_ids = np.setdiff1d(possible_indices, val_ids)
                else:
                    # If not enough samples, use all available
                    val_ids = possible_indices
                    remaining_ids = np.array([])

                all_val_ids.extend(val_ids)

                if custom_test_ids is None:  # sample test split
                    if len(remaining_ids) >= test_num[c]:
                        test_ids = np.random.choice(
                            remaining_ids, test_num[c], replace=False
                        )
                        remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                    else:
                        # If not enough samples, use all available
                        test_ids = remaining_ids
                        remaining_ids = np.array([])
                    all_test_ids.extend(test_ids)

                if label_frac == 1:
                    sampled_train_ids.extend(remaining_ids)
                else:
                    sample_num = math.ceil(len(remaining_ids) * label_frac)
                    if len(remaining_ids) > 0:
                        slice_ids = np.arange(sample_num)
                        sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y, task=TaskType.BINARY):
    """Calculate error based on task type"""
    if task == TaskType.REGRESSION:
        # For regression, use Mean Absolute Error (MAE)
        error = F.l1_loss(Y_hat, Y.float()).item()
    elif task == TaskType.BINARY or task == TaskType.MULTICLASS:
        # For classification, use classification error
        if Y_hat.dim() > 1:
            Y_hat = torch.argmax(Y_hat, dim=1)
        error = 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()
    else:
        raise Exception("Not supported task exception！")
    return error


def calculate_metrics(Y_hat, Y, task=TaskType.BINARY):
    """Calculate comprehensive metrics based on task type"""
    metrics = {}

    if task == TaskType.REGRESSION:
        # Regression metrics
        metrics["mae"] = F.l1_loss(Y_hat, Y.float()).item()
        metrics["mse"] = F.mse_loss(Y_hat, Y.float()).item()
        metrics["rmse"] = math.sqrt(metrics["mse"])

        # Handle tuple labels (primary:secondary format)
        if Y_hat.shape[1] == 2:  # Primary and secondary patterns
            metrics["primary_mae"] = F.l1_loss(Y_hat[:, 0], Y[:, 0].float()).item()
            metrics["secondary_mae"] = F.l1_loss(Y_hat[:, 1], Y[:, 1].float()).item()

    elif task == TaskType.BINARY or task == TaskType.MULTICLASS:
        # Classification metrics
        if Y_hat.dim() > 1:
            Y_pred = torch.argmax(Y_hat, dim=1)
        else:
            Y_pred = (Y_hat > 0.5).long()

        metrics["accuracy"] = Y_pred.float().eq(Y.float()).float().mean().item()
        metrics["error"] = 1 - metrics["accuracy"]

        # Additional classification metrics can be added here
        if task == TaskType.BINARY:
            # Binary classification specific metrics
            from sklearn.metrics import roc_auc_score, average_precision_score

            try:
                if Y_hat.dim() > 1:
                    y_score = torch.softmax(Y_hat, dim=1)[:, 1].cpu().numpy()
                else:
                    y_score = torch.sigmoid(Y_hat).cpu().numpy()
                y_true = Y.cpu().numpy()
                metrics["auc"] = roc_auc_score(y_true, y_score)
                metrics["ap"] = average_precision_score(y_true, y_score)
            except:
                metrics["auc"] = 0.0
                metrics["ap"] = 0.0
    else:
        raise Exception("Not supported task exception！")

    return metrics


def make_weights_for_balanced_classes_split(dataset):
    """Create weights for balanced sampling (classification only)"""
    # Check if dataset has task attribute and skip if regression
    if hasattr(dataset, "task") and dataset.task == TaskType.REGRESSION:
        # Return uniform weights for regression
        N = float(len(dataset))
        return torch.DoubleTensor([1.0] * int(N))

    N = float(len(dataset))
    weight_per_class = [
        N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))
    ]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def seed_torch(seed: int = 7) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Configure CuDNN for deterministic results
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DataLoaderFactory:
    """Factory class for creating data loaders based on task type"""

    @staticmethod
    def create_data_loaders(config, train_split, val_split, test_split):
        """Create data loaders for train, validation, and test splits"""

        # Get task type from train_split (assuming all splits have same task)
        task = getattr(train_split, "task", TaskType.BINARY)

        print(f"Creating data loaders for task: {task.value}")

        # Training loader
        if train_split is not None:
            train_loader = get_split_loader(
                train_split, training=True, weighted=config.weighted_sample, task=task
            )
        else:
            train_loader = None

        # Validation loader
        if val_split is not None:
            val_loader = get_split_loader(val_split, training=False, task=task)
        else:
            val_loader = None

        # Test loader
        if test_split is not None:
            test_loader = get_split_loader(test_split, testing=True, task=task)
        else:
            test_loader = None

        return train_loader, val_loader, test_loader
