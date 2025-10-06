from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import (
    Generic_WSI_Classification_Dataset,
    Generic_MIL_Dataset,
    save_splits,
)
import h5py
from utils.eval_utils import *


class EvalConfig:
    """Configuration class for evaluation"""
    
    def __init__(self, **kwargs):
        # Data parameters
        self.data_root_dir = kwargs.get('data_root_dir', None)
        self.data_set_name = kwargs.get('data_set_name', None)
        
        # Results and paths
        self.results_dir = kwargs.get('results_dir', './results')
        self.save_exp_code = kwargs.get('save_exp_code', None)
        self.models_exp_code = kwargs.get('models_exp_code', None)
        self.splits_dir = kwargs.get('splits_dir', None)
        
        # Model parameters
        self.model_size = kwargs.get('model_size', 'small')
        self.model_type = kwargs.get('model_type', 'clam_sb')
        self.drop_out = kwargs.get('drop_out', 0.25)
        self.embed_dim = kwargs.get('embed_dim', 1024)
        
        # Evaluation parameters
        self.k = kwargs.get('k', 10)
        self.k_start = kwargs.get('k_start', -1)
        self.k_end = kwargs.get('k_end', -1)
        self.fold = kwargs.get('fold', -1)
        self.micro_average = kwargs.get('micro_average', False)
        self.split = kwargs.get('split', 'test')
        self.task = kwargs.get('task', None)
        
        # Derived attributes
        self._setup_derived_attributes()
    
    def _setup_derived_attributes(self):
        """Setup derived attributes and paths"""
        # Set number of classes based on task
        if self.task == "task_1_tumor_vs_normal":
            self.n_classes = 2
        elif self.task == "task_2_tumor_subtyping":
            self.n_classes = 3
        else:
            self.n_classes = None
        
        # Setup directories
        self.save_dir = os.path.join("./eval_results", "EVAL_" + str(self.save_exp_code))
        self.models_dir = os.path.join(self.results_dir, str(self.models_exp_code))
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup splits directory
        if self.splits_dir is None:
            self.splits_dir = self.models_dir
        
        # Validate directories
        assert os.path.isdir(self.models_dir), f"Models directory {self.models_dir} does not exist"
        assert os.path.isdir(self.splits_dir), f"Splits directory {self.splits_dir} does not exist"
        
        # Setup fold ranges
        if self.k_start == -1:
            self.start_fold = 0
        else:
            self.start_fold = self.k_start
            
        if self.k_end == -1:
            self.end_fold = self.k
        else:
            self.end_fold = self.k_end
        
        # Setup folds to evaluate
        if self.fold == -1:
            self.folds = range(self.start_fold, self.end_fold)
        else:
            self.folds = range(self.fold, self.fold + 1)
    
    def get_settings(self):
        """Return settings dictionary for logging"""
        return {
            "task": self.task,
            "split": self.split,
            "save_dir": self.save_dir,
            "models_dir": self.models_dir,
            "model_type": self.model_type,
            "drop_out": self.drop_out,
            "model_size": self.model_size,
        }


def setup_eval_dataset(config):
    """Setup dataset for evaluation based on configuration"""
    if config.task == "task_1_tumor_vs_normal":
        dataset = Generic_MIL_Dataset(
            csv_path="dataset_csv/tumor_vs_normal_dummy_clean.csv",
            data_dir=os.path.join(config.data_root_dir, config.data_set_name),
            shuffle=False,
            print_info=True,
            label_dict={"normal_tissue": 0, "tumor_tissue": 1},
            patient_strat=False,
            ignore=[],
        )
    elif config.task == "task_2_tumor_subtyping":
        dataset = Generic_MIL_Dataset(
            csv_path="dataset_csv/tumor_subtyping_dummy_clean.csv",
            data_dir=os.path.join(config.data_root_dir, config.data_set_name),
            shuffle=False,
            print_info=True,
            label_dict={"subtype_1": 0, "subtype_2": 1, "subtype_3": 2},
            patient_strat=False,
            ignore=[],
        )
    else:
        raise NotImplementedError(f"Task {config.task} not implemented")
    
    return dataset


def run_evaluation(config):
    """Main evaluation function"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup dataset
    dataset = setup_eval_dataset(config)
    
    # Log settings
    settings = config.get_settings()
    log_file = os.path.join(config.save_dir, "eval_experiment_{}.txt".format(config.save_exp_code))
    with open(log_file, "w") as f:
        print(settings, file=f)
    
    print("Evaluation Settings:")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    
    # Get checkpoint paths
    ckpt_paths = [
        os.path.join(config.models_dir, "s_{}_checkpoint.pt".format(fold)) 
        for fold in config.folds
    ]
    
    # Dataset split mapping
    datasets_id = {"train": 0, "val": 1, "test": 2, "all": -1}
    
    # Run evaluation across folds
    all_results = []
    all_auc = []
    all_acc = []
    
    for ckpt_idx in range(len(ckpt_paths)):
        # Get the appropriate dataset split
        if datasets_id[config.split] < 0:
            split_dataset = dataset
        else:
            csv_path = "{}/splits_{}.csv".format(config.splits_dir, config.folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[config.split]]
        
        # Run evaluation
        model, patient_results, test_error, auc, df = eval(
            split_dataset, config, ckpt_paths[ckpt_idx]
        )
        
        all_results.append(patient_results)
        all_auc.append(auc)
        all_acc.append(1 - test_error)
        
        # Save individual fold results
        df.to_csv(
            os.path.join(config.save_dir, "fold_{}.csv".format(config.folds[ckpt_idx])),
            index=False,
        )
    
    # Save summary
    final_df = pd.DataFrame({
        "folds": config.folds, 
        "test_auc": all_auc, 
        "test_acc": all_acc
    })
    
    if len(config.folds) != config.k:
        save_name = "summary_partial_{}_{}.csv".format(config.folds[0], config.folds[-1])
    else:
        save_name = "summary.csv"
        
    final_df.to_csv(os.path.join(config.save_dir, save_name))
    
    return {
        'all_results': all_results,
        'all_auc': all_auc,
        'all_acc': all_acc,
        'final_df': final_df
    }


def create_parser():
    """Create argument parser for evaluation"""
    parser = argparse.ArgumentParser(description="CLAM Evaluation Script")
    
    # Data parameters
    parser.add_argument("--data_root_dir", type=str, default=None, help="data directory")
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
    
    # Evaluation parameters
    parser.add_argument("--k", type=int, default=10, help="number of folds (default: 10)")
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
    
    return parser


def main():
    """Main function for command line execution"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Convert args to dictionary and create config
    config_dict = vars(args)
    config = EvalConfig(**config_dict)
    
    # Run evaluation
    results = run_evaluation(config)
    print("Evaluation completed!")
    
    return results


if __name__ == "__main__":
    main()