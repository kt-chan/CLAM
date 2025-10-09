import time
import os
import argparse
import pdb
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder
from pathlib import Path

# Set device and parallelization parameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_workers_per_process = 4
num_cpus = int((mp.cpu_count()) / num_workers_per_process) - 1 if mp.cpu_count() > 2 else 1
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Enable CUDA optimizations
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def compute_w_loader(output_path, loader, model, verbose=0):
    """
    Optimized feature computation with better GPU utilization
    """
    if verbose > 0:
        print(f"processing a total of {len(loader)} batches")

    mode = "w"

    # Pre-allocate tensors on GPU to reduce overhead
    for count, data in enumerate(tqdm(loader, desc="Processing batches")):
        with torch.inference_mode():
            batch = data["img"].to(device, non_blocking=True)
            coords = data["coord"].numpy().astype(np.int32)

            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = "a"

    return output_path


def process_single_slide(slide_info, args, model_state_dict=None):
    """
    Process a single slide in parallel
    """
    slide_id, bag_candidate_idx, total, gpu_id = slide_info

    # Set the GPU device for this process
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        current_device = torch.device(f"cuda:{gpu_id}")
    else:
        current_device = device

    # Initialize model for this process
    model, img_transforms = get_encoder(
        args.model_name, target_img_size=args.target_patch_size
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    model = model.to(current_device)
    model.eval()

    bag_name = slide_id + ".h5"
    h5_file_path = os.path.join(args.data_h5_dir, "patches", bag_name)
    slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

    print(
        f"Processing {slide_id} ({bag_candidate_idx + 1}/{total}) on device {current_device}"
    )

    output_path = os.path.join(args.feat_dir, "h5_files", bag_name)

    # Skip if already processed
    pt_file_path = os.path.join(args.feat_dir, "pt_files", slide_id + ".pt")
    if not args.no_auto_skip and os.path.exists(pt_file_path):
        print(f"Skipped {slide_id} - already processed")
        return slide_id, None

    time_start = time.time()

    # Configure DataLoader for optimal performance
    loader_kwargs = (
        {
            "num_workers": num_workers_per_process,
            "pin_memory": True,
            "persistent_workers": False,  # Disable for multiprocessing compatibility
            "prefetch_factor": 4,
        }
        if current_device.type == "cuda"
        else {}
    )

    dataset = Whole_Slide_Bag_FP(
        file_path=h5_file_path,
        wsi_path=slide_file_path,
        img_transforms=img_transforms,
    )

    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

    output_file_path = compute_w_loader(
        output_path, loader=loader, model=model, verbose=0
    )

    time_elapsed = time.time() - time_start

    # Convert to PT file
    with h5py.File(output_file_path, "r") as file:
        features = file["features"][:]

    features = torch.from_numpy(features)
    bag_base, _ = os.path.splitext(bag_name)
    torch.save(features, pt_file_path)

    # Clean up GPU memory
    if current_device.type == "cuda":
        torch.cuda.empty_cache()

    return slide_id, time_elapsed


def get_available_gpus():
    """Get list of available GPU IDs"""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def process_slides_parallel(bags_dataset, args, model):
    """
    Process multiple slides in parallel with GPU distribution
    """
    # Prepare slide information
    slide_infos = []
    available_gpus = get_available_gpus()

    for i in range(len(bags_dataset)):
        slide_id = bags_dataset[i].split(args.slide_ext)[0]
        # Distribute slides across available GPUs
        gpu_id = available_gpus[i % len(available_gpus)] if available_gpus else None
        slide_infos.append((slide_id, i, len(bags_dataset), gpu_id))

    # Use model state dict to avoid pickle issues
    model_state_dict = model.state_dict()

    # Determine optimal number of parallel processes
    if available_gpus:
        # For multi-GPU, limit to number of GPUs to avoid memory contention
        max_workers = min(len(slide_infos), num_cpus)
    else:
        # For CPU, use more conservative parallelization
        max_workers = min(4, len(slide_infos), num_cpus)

    print(f"Using {max_workers} parallel processes across {len(available_gpus)} GPUs")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_slide = {
            executor.submit(
                process_single_slide, slide_info, args, model_state_dict
            ): slide_info[0]
            for slide_info in slide_infos
        }

        # Collect results as they complete
        for future in tqdm(future_to_slide, desc="Processing slides in parallel"):
            try:
                slide_id, time_elapsed = future.result()
                if time_elapsed:
                    results.append((slide_id, time_elapsed))
                    print(f"✓ Completed {slide_id} in {time_elapsed:.2f}s")
            except Exception as e:
                slide_id = future_to_slide[future]
                print(f"✗ Error processing {slide_id}: {e}")
                # Print full traceback for debugging
                import traceback

                traceback.print_exc()

    return results


def process_slides_sequential(bags_dataset, args, model):
    """
    Fallback sequential processing with optimizations
    """
    model = model.to(device)
    model.eval()

    total = len(bags_dataset)
    dest_dir = os.path.join(args.feat_dir, "pt_files")

    # Optimized DataLoader settings
    loader_kwargs = (
        {
            "num_workers": num_workers_per_process,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        if device.type == "cuda"
        else {}
    )

    results = []
    for bag_candidate_idx in tqdm(range(total), desc="Processing slides sequentially"):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + ".h5"

        pt_file_path = os.path.join(dest_dir, slide_id + ".pt")
        if not args.no_auto_skip and os.path.exists(pt_file_path):
            print(f"Skipped {slide_id}")
            continue

        h5_file_path = os.path.join(args.data_h5_dir, "patches", bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        output_path = os.path.join(args.feat_dir, "h5_files", bag_name)

        time_start = time.time()

        dataset = Whole_Slide_Bag_FP(
            file_path=h5_file_path,
            wsi_path=slide_file_path,
            img_transforms=args.img_transforms,
        )

        loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, **loader_kwargs
        )

        output_file_path = compute_w_loader(
            output_path, loader=loader, model=model, verbose=0
        )

        time_elapsed = time.time() - time_start

        # Convert to PT file
        with h5py.File(output_file_path, "r") as file:
            features = file["features"][:]

        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, pt_file_path)

        results.append((slide_id, time_elapsed))
        print(f"Completed {slide_id} in {time_elapsed:.2f}s")

    return results


parser = argparse.ArgumentParser(description="Feature Extraction")
parser.add_argument("--data_h5_dir", type=str, default=None)
parser.add_argument("--data_slide_dir", type=str, default=None)
parser.add_argument("--slide_ext", type=str, default=".svs")
parser.add_argument("--csv_path", type=str, default=None)
parser.add_argument("--feat_dir", type=str, default=None)
parser.add_argument(
    "--model_name",
    type=str,
    default="resnet50_trunc",
    choices=["resnet50_trunc", "uni_v1", "conch_v1"],
)
parser.add_argument(
    "--batch_size", type=int, default=512
)  # Increased default batch size
parser.add_argument("--no_auto_skip", default=False, action="store_true")
parser.add_argument("--target_patch_size", type=int, default=224)
parser.add_argument(
    "--no_parallel",
    default=False,
    action="store_true",
    help="Disable parallel processing",
)
parser.add_argument(
    "--max_workers", type=int, default=None, help="Max parallel workers"
)

if __name__ == "__main__":
    print("*" * 128)
    print(f"Running feature extraction on {device}")
    print(f"Available CPUs: {num_cpus}, Available GPUs: {num_gpus}")
    print("*" * 128)

    args = parser.parse_args()

    if args.csv_path is None:
        raise NotImplementedError("CSV path must be provided")

    print("Initializing dataset and model")
    bags_dataset = Dataset_All_Bags(args.csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "pt_files"), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "h5_files"), exist_ok=True)

    model, img_transforms = get_encoder(
        args.model_name, target_img_size=args.target_patch_size
    )
    args.img_transforms = img_transforms  # Store for reuse

    total_start_time = time.time()

    # Use parallel processing by default unless explicitly disabled
    if not args.no_parallel and num_cpus > 1:
        print("Using PARALLEL processing mode (default)")
        try:
            results = process_slides_parallel(bags_dataset, args, model)
        except Exception as e:
            print(f"Parallel processing failed, falling back to sequential: {e}")
            print("Using SEQUENTIAL processing mode (fallback)")
            results = process_slides_sequential(bags_dataset, args, model)
    else:
        print("Using SEQUENTIAL processing mode (requested)")
        results = process_slides_sequential(bags_dataset, args, model)

    total_time = time.time() - total_start_time

    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    successful_results = [r for r in results if r[1] is not None]
    for slide_id, slide_time in successful_results:
        print(f"✓ {slide_id}: {slide_time:.2f}s")

    print(f"\nTotal slides processed: {len(successful_results)}")
    print(f"Total processing time: {total_time:.2f}s")
    if successful_results:
        print(f"Average time per slide: {total_time/len(successful_results):.2f}s")
        print(
            f"Estimated sequential time: {sum(r[1] for r in successful_results):.2f}s"
        )
        if len(successful_results) > 1:
            speedup = sum(r[1] for r in successful_results) / total_time
            print(f"Parallel speedup: {speedup:.2f}x")
