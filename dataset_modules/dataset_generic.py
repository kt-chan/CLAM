import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
from enum import Enum
from typing import Optional, Union

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth, TaskType


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [
        split_datasets[i].slide_data["slide_id"] for i in range(len(split_datasets))
    ]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=["train", "val", "test"])

    df.to_csv(filename)
    print()


class Generic_WSI_Dataset(Dataset):
    """Base class for WSI datasets supporting classification and regression tasks"""

    def __init__(
        self,
        csv_path: str = "dataset_csv/ccrcc_clean.csv",
        data_dir: Optional[str] = None,
        shuffle: bool = False,
        seed: int = 7,
        print_info: bool = True,
        label_dict: dict = {},
        filter_dict: dict = {},
        ignore: list = [],
        patient_strat: bool = False,
        label_col: Optional[str] = None,
        patient_voting: str = "max",
        task: TaskType = TaskType.BINARY,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
            patient_strat (boolean): Whether to use patient-level stratification
            label_col (string): Column name for labels
            patient_voting (string): Voting method for patient-level labels ('max', 'maj')
            task (TaskType): Type of task - binary, multiclass, or regression
        """
        self.label_dict = label_dict
        self.task = task
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = data_dir

        if not label_col:
            label_col = "label"
        self.label_col = label_col

        # Set num_classes based on task type
        if self.task == TaskType.REGRESSION:
            self.num_classes = 1
        elif self.task == TaskType.BINARY:
            self.num_classes = 2
        elif self.task == TaskType.MULTICLASS:
            self.num_classes = len(set(self.label_dict.values()))
        else:  # others
            raise Exception("Not supported Task Type Exception!")

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

        # shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        if self.task == TaskType.REGRESSION:
            # For regression, we don't have discrete classes, so we treat all as one class
            self.patient_cls_ids = [np.arange(len(self.patient_data["case_id"]))]
            self.slide_cls_ids = [np.arange(len(self.slide_data))]
        elif self.task == TaskType.BINARY or self.task == TaskType.MULTICLASS:
            # store ids corresponding each class at the patient or case level
            self.patient_cls_ids = [[] for i in range(self.num_classes)]
            for i in range(self.num_classes):
                self.patient_cls_ids[i] = np.where(self.patient_data["label"] == i)[0]

            # store ids corresponding each class at the slide level
            self.slide_cls_ids = [[] for i in range(self.num_classes)]
            for i in range(self.num_classes):
                self.slide_cls_ids[i] = np.where(self.slide_data["label"] == i)[0]
        else:
            raise Exception("Not supported task type exception!")

    def patient_data_prep(self, patient_voting: str = "max"):
        patients = np.unique(
            np.array(self.slide_data["case_id"])
        )  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data["case_id"] == p].index.tolist()
            assert len(locations) > 0

            if self.task == TaskType.REGRESSION:
                # For regression, handle both single values and primary:secondary format
                labels = self.slide_data["label"][locations].values
                if patient_voting == "max":
                    # For patterns, take the maximum pattern from all slides
                    if isinstance(labels[0], tuple):
                        # Handle primary:secondary format - take max of primary patterns
                        primary_patterns = [label[0] for label in labels]
                        label = max(primary_patterns)
                    else:
                        label = max([float(label) for label in labels])
                elif patient_voting == "maj":
                    # For regression, use mean as majority equivalent
                    if isinstance(labels[0], tuple):
                        primary_patterns = [label[0] for label in labels]
                        secondary_patterns = [label[1] for label in labels]
                        label = (np.mean(primary_patterns), np.mean(secondary_patterns))
                    else:
                        label = np.mean([float(label) for label in labels])
                else:
                    raise NotImplementedError(
                        f"Patient voting {patient_voting} not implemented for regression"
                    )
            elif self.task == TaskType.BINARY or self.task == TaskType.MULTICLASS:
                # Classification tasks
                label = self.slide_data["label"][locations].values
                if patient_voting == "max":
                    label = label.max()  # get patient label (MIL convention)
                elif patient_voting == "maj":
                    label = stats.mode(label)[0]
                else:
                    raise NotImplementedError(
                        f"Patient voting {patient_voting} not implemented for classification"
                    )
            else:
                raise Exception("Not supported task exception!")

            patient_labels.append(label)

        self.patient_data = {"case_id": patients, "label": np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != "label":
            data["label"] = data[label_col].copy()

        mask = data["label"].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)

        # Check if this is regression data (contains ':' separated values)
        sample_label = data.iloc[0]["label"] if len(data) > 0 else None
        is_regression = isinstance(sample_label, str) and ":" in str(sample_label)

        if not is_regression and label_dict:
            # Classification: use label_dict to convert labels
            for i in data.index:
                key = data.loc[i, "label"]
                data.at[i, "label"] = label_dict[key]
        elif is_regression:
            # Regression: parse primary:secondary format and store as tuple
            parsed_labels = []
            for i in data.index:
                label_str = str(data.loc[i, "label"])
                if ":" in label_str:
                    primary, secondary = map(float, label_str.split(":"))
                    parsed_labels.append((primary, secondary))
                else:
                    # Single value regression
                    parsed_labels.append(float(label_str))
            data["label"] = parsed_labels

        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data["case_id"])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("task type: {}".format(self.task.value))

        if self.task == TaskType.BINARY or self.task == TaskType.MULTICLASS:
            print("label dictionary: {}".format(self.label_dict))
            print("number of classes: {}".format(self.num_classes))
            print(
                "slide-level counts: ",
                "\n",
                self.slide_data["label"].value_counts(sort=False),
            )
            for i in range(self.num_classes):
                print(
                    "Patient-LVL; Number of samples registered in class %d: %d"
                    % (i, self.patient_cls_ids[i].shape[0])
                )
                print(
                    "Slide-LVL; Number of samples registered in class %d: %d"
                    % (i, self.slide_cls_ids[i].shape[0])
                )
        elif self.task == TaskType.REGRESSION:
            # Regression task
            labels = self.slide_data["label"].values
            if isinstance(labels[0], tuple):
                primary_patterns = [label[0] for label in labels]
                secondary_patterns = [label[1] for label in labels]
                print(
                    "Primary pattern stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(primary_patterns),
                        max(primary_patterns),
                        np.mean(primary_patterns),
                        np.std(primary_patterns),
                    )
                )
                print(
                    "Secondary pattern stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(secondary_patterns),
                        max(secondary_patterns),
                        np.mean(secondary_patterns),
                        np.std(secondary_patterns),
                    )
                )
            else:
                print(
                    "Regression label stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(labels), max(labels), np.mean(labels), np.std(labels)
                    )
                )
        else:
            raise Exception("Not supported task exception!")

    def create_splits(
        self,
        k: int = 3,
        val_num: tuple = (25, 25),
        test_num: tuple = (40, 40),
        label_frac: float = 1.0,
        custom_test_ids: Optional[list] = None,
    ):
        settings = {
            "n_splits": k,
            "val_num": val_num,
            "test_num": test_num,
            "label_frac": label_frac,
            "seed": self.seed,
            "custom_test_ids": custom_test_ids,
        }

        if self.patient_strat:
            settings.update(
                {
                    "cls_ids": self.patient_cls_ids,
                    "samples": len(self.patient_data["case_id"]),
                }
            )
        else:
            settings.update(
                {"cls_ids": self.slide_cls_ids, "samples": len(self.slide_data)}
            )

        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from: Optional[int] = None):
        if start_from:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]

            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data["case_id"][idx]
                    slide_indices = self.slide_data[
                        self.slide_data["case_id"] == case_id
                    ].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = (
                slide_ids[0],
                slide_ids[1],
                slide_ids[2],
            )
        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key: str = "train"):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data["slide_id"].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(
                df_slice,
                data_dir=self.data_dir,
                num_classes=self.num_classes,
                task=self.task,
            )
        else:
            split = None

        return split

    def get_merged_split_from_df(self, all_splits, split_keys: list = ["train"]):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(merged_split) > 0:
            mask = self.slide_data["slide_id"].isin(merged_split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(
                df_slice,
                data_dir=self.data_dir,
                num_classes=self.num_classes,
                task=self.task,
            )
        else:
            split = None

        return split

    def return_splits(self, from_id: bool = True, csv_path: Optional[str] = None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(
                    train_data,
                    data_dir=self.data_dir,
                    num_classes=self.num_classes,
                    task=self.task,
                )
            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(
                    val_data,
                    data_dir=self.data_dir,
                    num_classes=self.num_classes,
                    task=self.task,
                )
            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(
                    test_data,
                    data_dir=self.data_dir,
                    num_classes=self.num_classes,
                    task=self.task,
                )
            else:
                test_split = None

        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data["slide_id"].dtype)
            train_split = self.get_split_from_df(all_splits, "train")
            val_split = self.get_split_from_df(all_splits, "val")
            test_split = self.get_split_from_df(all_splits, "test")

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data["slide_id"][ids]

    def getlabel(self, ids):
        return self.slide_data["label"][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor: bool = False):
        if return_descriptor:
            if self.task == TaskType.REGRESSION:
                index = ["regression"]
                columns = ["train", "val", "test"]
                df = pd.DataFrame(
                    np.full((len(index), len(columns)), 0, dtype=np.int32),
                    index=index,
                    columns=columns,
                )
            elif self.task == TaskType.BINARY or self.task == TaskType.MULTICLASS:
                index = [
                    list(self.label_dict.keys())[
                        list(self.label_dict.values()).index(i)
                    ]
                    for i in range(self.num_classes)
                ]
                columns = ["train", "val", "test"]
                df = pd.DataFrame(
                    np.full((len(index), len(columns)), 0, dtype=np.int32),
                    index=index,
                    columns=columns,
                )
            else:
                raise Exception("Not supported task exception!")

        count = len(self.train_ids)
        print("\nnumber of training samples: {}".format(count))
        if self.task == TaskType.REGRESSION:
            labels = self.getlabel(self.train_ids)
            if isinstance(labels.iloc[0], tuple):
                primary = [label[0] for label in labels]
                secondary = [label[1] for label in labels]
                print(
                    "Primary pattern stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(primary), max(primary), np.mean(primary), np.std(primary)
                    )
                )
                print(
                    "Secondary pattern stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(secondary),
                        max(secondary),
                        np.mean(secondary),
                        np.std(secondary),
                    )
                )
            else:
                print(
                    "Regression label stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(labels), max(labels), np.mean(labels), np.std(labels)
                    )
                )
        elif self.task == TaskType.BINARY or self.task == TaskType.MULTICLASS:
            labels = self.getlabel(self.train_ids)
            unique, counts = np.unique(labels, return_counts=True)
            for u in range(len(unique)):
                print("number of samples in cls {}: {}".format(unique[u], counts[u]))
                if return_descriptor:
                    df.loc[index[u], "train"] = counts[u]
        else:
            raise Exception("Not supported task exception!")

        count = len(self.val_ids)
        print("\nnumber of val samples: {}".format(count))
        if self.task == TaskType.REGRESSION:
            labels = self.getlabel(self.val_ids)
            if isinstance(labels.iloc[0], tuple):
                primary = [label[0] for label in labels]
                secondary = [label[1] for label in labels]
                print(
                    "Primary pattern stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(primary), max(primary), np.mean(primary), np.std(primary)
                    )
                )
                print(
                    "Secondary pattern stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(secondary),
                        max(secondary),
                        np.mean(secondary),
                        np.std(secondary),
                    )
                )
            else:
                print(
                    "Regression label stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(labels), max(labels), np.mean(labels), np.std(labels)
                    )
                )
        elif self.task == TaskType.BINARY or self.task == TaskType.MULTICLASS:
            labels = self.getlabel(self.val_ids)
            unique, counts = np.unique(labels, return_counts=True)
            for u in range(len(unique)):
                print("number of samples in cls {}: {}".format(unique[u], counts[u]))
                if return_descriptor:
                    df.loc[index[u], "val"] = counts[u]
        else:
            raise Exception("Not supported task exception!")

        count = len(self.test_ids)
        print("\nnumber of test samples: {}".format(count))
        if self.task == TaskType.REGRESSION:
            labels = self.getlabel(self.test_ids)
            if isinstance(labels.iloc[0], tuple):
                primary = [label[0] for label in labels]
                secondary = [label[1] for label in labels]
                print(
                    "Primary pattern stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(primary), max(primary), np.mean(primary), np.std(primary)
                    )
                )
                print(
                    "Secondary pattern stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(secondary),
                        max(secondary),
                        np.mean(secondary),
                        np.std(secondary),
                    )
                )
            else:
                print(
                    "Regression label stats - min: {:.1f}, max: {:.1f}, mean: {:.2f}, std: {:.2f}".format(
                        min(labels), max(labels), np.mean(labels), np.std(labels)
                    )
                )
        elif self.task == TaskType.BINARY or self.task == TaskType.MULTICLASS:
            labels = self.getlabel(self.test_ids)
            unique, counts = np.unique(labels, return_counts=True)
            for u in range(len(unique)):
                print("number of samples in cls {}: {}".format(unique[u], counts[u]))
                if return_descriptor:
                    df.loc[index[u], "test"] = counts[u]
        else:
            raise Exception("Not supported task exception!")

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({"train": train_split})
        df_v = pd.DataFrame({"val": val_split})
        df_t = pd.DataFrame({"test": test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)


class Generic_WSI_Classification_Dataset(Generic_WSI_Dataset):
    """Classification-specific dataset that inherits from the base class"""

    def __init__(self, **kwargs):
        # Set appropriate task type based on label_dict
        if "label_dict" in kwargs and len(kwargs["label_dict"]) > 2:
            kwargs["task"] = TaskType.MULTICLASS
        else:
            kwargs["task"] = TaskType.BINARY
        super(Generic_WSI_Classification_Dataset, self).__init__(**kwargs)


class Generic_WSI_Regression_Dataset(Generic_WSI_Dataset):
    """Regression-specific dataset that inherits from the base class"""

    def __init__(self, **kwargs):
        # Force task to regression
        kwargs["task"] = TaskType.REGRESSION
        super(Generic_WSI_Regression_Dataset, self).__init__(**kwargs)


class Generic_MIL_Dataset(Generic_WSI_Dataset):
    """Generic MIL Dataset that supports binary, multiclass, and regression tasks"""

    def __init__(self, **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = self.slide_data["slide_id"][idx]
        label = self.slide_data["label"][idx]

        if type(self.data_dir) == dict:
            source = self.slide_data["source"][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                full_path = os.path.join(data_dir, "pt_files", "{}.pt".format(slide_id))
                features = torch.load(full_path)
                return features, label
            else:
                return slide_id, label
        else:
            full_path = os.path.join(data_dir, "h5_files", "{}.h5".format(slide_id))
            with h5py.File(full_path, "r") as hdf5_file:
                features = hdf5_file["features"][:]
                coords = hdf5_file["coords"][:]

            features = torch.from_numpy(features)
            return features, label, coords


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2, task=TaskType.BINARY):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.task = task

        if self.task == TaskType.BINARY or self.task == TaskType.MULTICLASS:
            self.slide_cls_ids = [[] for i in range(self.num_classes)]
            for i in range(self.num_classes):
                self.slide_cls_ids[i] = np.where(self.slide_data["label"] == i)[0]
        elif self.task == TaskType.REGRESSION:
            # For regression, all samples belong to one "class"
            self.slide_cls_ids = [np.arange(len(self.slide_data))]
            self.num_classes = -1
        else:
            raise Exception("Not supported task exception!")

    def __len__(self):
        return len(self.slide_data)
