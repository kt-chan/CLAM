import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

def create_splits(task, label_frac=1.0, seed=1, k=10, val_frac=0.1, test_frac=0.1):
    """
    Create dataset splits for whole slide classification
    
    Args:
        task (str): Task identifier - 'task_1_tumor_vs_normal' or 'task_2_tumor_subtyping'
        label_frac (float): Fraction of labels to use (default: 1.0)
        seed (int): Random seed (default: 1)
        k (int): Number of splits (default: 10)
        val_frac (float): Fraction of labels for validation (default: 0.1)
        test_frac (float): Fraction of labels for test (default: 0.1)
    """
    
    if task == 'task_1_tumor_vs_normal':
        n_classes = 2
        dataset = Generic_WSI_Classification_Dataset(
            csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
            shuffle=False, 
            seed=seed, 
            print_info=True,
            label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
            patient_strat=True,
            ignore=[]
        )

    elif task == 'task_2_tumor_subtyping':
        n_classes = 3
        dataset = Generic_WSI_Classification_Dataset(
            csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
            shuffle=False, 
            seed=seed, 
            print_info=True,
            label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
            patient_strat=True,
            patient_voting='maj',
            ignore=[]
        )

    else:
        raise NotImplementedError(f"Task {task} not implemented")

    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.round(num_slides_cls * val_frac).astype(int)
    test_num = np.round(num_slides_cls * test_frac).astype(int)

    if label_frac > 0:
        label_fracs = [label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = f'splits/{task}_{int(lf * 100)}'
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k=k, val_num=val_num, test_num=test_num, label_frac=lf)
        for i in range(k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, f'splits_{i}.csv'))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, f'splits_{i}_bool.csv'), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, f'splits_{i}_descriptor.csv'))

# Keep command-line functionality for backward compatibility
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
    parser.add_argument('--label_frac', type=float, default=1.0,
                        help='fraction of labels (default: 1)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--k', type=int, default=10,
                        help='number of splits (default: 10)')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'],
                        required=True, help='Task to perform')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='fraction of labels for validation (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='fraction of labels for test (default: 0.1)')

    args = parser.parse_args()
    
    # Call the function with command-line arguments
    create_splits(
        task=args.task,
        label_frac=args.label_frac,
        seed=args.seed,
        k=args.k,
        val_frac=args.val_frac,
        test_frac=args.test_frac
    )