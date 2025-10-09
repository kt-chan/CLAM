# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df

# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm
import os
import shutil
from pathlib import Path
import h5py


# --- HELPER FUNCTION FOR DUMMY H5 ---
def create_dummy_h5(slide_id, patch_save_dir, patch_size, patch_level=0):
    """
    Creates a dummy H5 file containing a single patch coordinate [0, 0].
    Attributes 'patch_level' and 'patch_size' are added to the 'coords' dataset
    to satisfy the Whole_Slide_Bag_FP reader class.
    """
    h5_output_path = os.path.join(patch_save_dir, slide_id + ".h5")

    # The coordinate is [0, 0] to start at the top-left of the image.
    coords = np.array([[0, 0]], dtype=np.int32)

    with h5py.File(h5_output_path, "w") as f:
        # Create the 'coords' dataset
        dset = f.create_dataset("coords", data=coords)

        # Attach required attributes to the 'coords' dataset (dset)
        dset.attrs["patch_level"] = patch_level
        dset.attrs["patch_size"] = patch_size

        # The 'downsample' attribute is not strictly needed by your provided
        # reader class, but retaining it here if it's expected elsewhere:
        dset.attrs["downsample"] = 1

    print(f"  --> Dummy H5 created for {slide_id} at {h5_output_path}")
    return h5_output_path, 0.0  # Return file_path and 0 elapsed time


# ------------------------------------


def move_svs_files_to_root(root_dir, file_extension):
    """
    Moves all files with the specified extension to the root directory.
    """
    if not os.path.exists(root_dir):
        print(f"Root directory {root_dir} does not exist.")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith("." + file_extension):
                file_path = str(Path(os.path.join(dirpath, filename)))
                dest_path = str(Path(os.path.join(root_dir, filename)))

                if file_path == dest_path:
                    break

                if os.path.exists(dest_path):
                    print(
                        f"File {filename} already exists in the root directory. Skipping."
                    )
                else:
                    shutil.move(file_path, dest_path)
                    print(f"Moved {file_path} to {dest_path}")


def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(
        file_path,
        wsi_object,
        downscale=downscale,
        bg_color=(0, 0, 0),
        alpha=-1,
        draw_grid=False,
    )
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    ### Start Seg Timer
    start_time = time.time()

    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(
    source,
    save_dir,
    patch_save_dir,
    mask_save_dir,
    stitch_save_dir,
    patch_size=256,
    step_size=256,
    seg_params={
        "seg_level": -1,
        "sthresh": 8,
        "mthresh": 7,
        "close": 4,
        "use_otsu": False,
        "keep_ids": "none",
        "exclude_ids": "none",
    },
    filter_params={"a_t": 100, "a_h": 16, "max_n_holes": 8},
    vis_params={"vis_level": -1, "line_thickness": 500},
    patch_params={"use_padding": True, "contour_fn": "four_pt"},
    patch_level=0,
    use_default_params=False,
    seg=False,
    save_mask=True,
    stitch=False,
    patch=False,
    auto_skip=True,
    process_list=None,
):

    slides = sorted(os.listdir(source))
    supported_extensions = [".svs", ".tif", ".tiff", ".ndpi", ".png", ".jpg", ".jpeg"]
    slides = [
        slide
        for slide in slides
        if os.path.splitext(slide)[1].lower() in supported_extensions
    ]

    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df["process"] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = "a" in df.keys()
    if legacy_support:
        print("detected legacy segmentation csv file, legacy support enabled")
        df = df.assign(
            **{
                "a_t": np.full((len(df)), int(filter_params["a_t"]), dtype=np.uint32),
                "a_h": np.full((len(df)), int(filter_params["a_h"]), dtype=np.uint32),
                "max_n_holes": np.full(
                    (len(df)), int(filter_params["max_n_holes"]), dtype=np.uint32
                ),
                "line_thickness": np.full(
                    (len(df)), int(vis_params["line_thickness"]), dtype=np.uint32
                ),
                "contour_fn": np.full((len(df)), patch_params["contour_fn"]),
            }
        )

    seg_times = 0.0
    patch_times = 0.0
    stitch_times = 0.0

    for i in tqdm(range(total)):
        df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, "slide_id"]
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print("processing {}".format(slide))

        df.loc[idx, "process"] = 0
        slide_id, file_ext = os.path.splitext(slide)
        file_ext = file_ext.lower()

        # Check if output H5 already exists
        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + ".h5")):
            print("{} already exist in destination location, skipped".format(slide_id))
            df.loc[idx, "status"] = "already_exist"
            continue

        # Initialize WSI object to read dimensions
        full_path = os.path.join(source, slide)

        WSI_object = None  # Initialize to None for the small image case

        try:
            # Attempt to open the file using WholeSlideImage
            WSI_object = WholeSlideImage(full_path)
            w, h = WSI_object.level_dim[
                0
            ]  # Get dimensions at level 0 (full resolution)
        except Exception as e:
            # This block handles all errors: unsupported format, WSI file corruption, etc.
            print(f"Error initializing WSI object for {slide}: {e}. Skipping.")
            df.loc[idx, "status"] = "failed_init"
            continue

        # --- CORE LOGIC: DIMENSION CHECK FOR PRE-PATCHED IMAGES (e.g., MHIST) ---
        if w <= 256 and h <= 256:
            print(
                f"Detected small image ({w}x{h}). Bypassing WSI pipeline with dummy H5."
            )

            file_path, patch_time_elapsed = create_dummy_h5(
                slide_id, patch_save_dir, patch_size=patch_size, patch_level=0
            )
            seg_time_elapsed = 0.0

            df.loc[idx, "status"] = "bypassed"

        # --- STANDARD WSI/SVS LOGIC ---
        else:
            print(f"Detected large image ({w}x{h}). Proceeding with WSI pipeline.")

            # --- Parameter Initialization (Remaining unchanged) ---
            if use_default_params:
                current_vis_params = vis_params.copy()
                current_filter_params = filter_params.copy()
                current_seg_params = seg_params.copy()
                current_patch_params = patch_params.copy()
            else:
                current_vis_params = {}
                current_filter_params = {}
                current_seg_params = {}
                current_patch_params = {}

                for key in vis_params.keys():
                    if legacy_support and key == "vis_level":
                        df.loc[idx, key] = -1
                    current_vis_params.update({key: df.loc[idx, key]})

                for key in filter_params.keys():
                    if legacy_support and key == "a_t":
                        old_area = df.loc[idx, "a"]
                        seg_level = df.loc[idx, "seg_level"]
                        scale = WSI_object.level_downsamples[seg_level]
                        adjusted_area = int(
                            old_area * (scale[0] * scale[1]) / (512 * 512)
                        )
                        current_filter_params.update({key: adjusted_area})
                        df.loc[idx, key] = adjusted_area
                    current_filter_params.update({key: df.loc[idx, key]})

                for key in seg_params.keys():
                    if legacy_support and key == "seg_level":
                        df.loc[idx, key] = -1
                    current_seg_params.update({key: df.loc[idx, key]})

                for key in patch_params.keys():
                    current_patch_params.update({key: df.loc[idx, key]})
            # --- End Parameter Initialization ---

            # --- Level Selection (Remaining unchanged) ---
            if current_vis_params["vis_level"] < 0:
                if len(WSI_object.level_dim) == 1:
                    current_vis_params["vis_level"] = 0
                else:
                    wsi = WSI_object.getOpenSlide()
                    best_level = wsi.get_best_level_for_downsample(64)
                    current_vis_params["vis_level"] = best_level

            if current_seg_params["seg_level"] < 0:
                if len(WSI_object.level_dim) == 1:
                    current_seg_params["seg_level"] = 0
                else:
                    wsi = WSI_object.getOpenSlide()
                    best_level = wsi.get_best_level_for_downsample(64)
                    current_seg_params["seg_level"] = best_level
            # --- End Level Selection ---

            # --- ID/Exclusion Handling (Remaining unchanged) ---
            keep_ids = str(current_seg_params["keep_ids"])
            if keep_ids != "none" and len(keep_ids) > 0:
                str_ids = current_seg_params["keep_ids"]
                current_seg_params["keep_ids"] = np.array(str_ids.split(",")).astype(
                    int
                )
            else:
                current_seg_params["keep_ids"] = []

            exclude_ids = str(current_seg_params["exclude_ids"])
            if exclude_ids != "none" and len(exclude_ids) > 0:
                str_ids = current_seg_params["exclude_ids"]
                current_seg_params["exclude_ids"] = np.array(str_ids.split(",")).astype(
                    int
                )
            else:
                current_seg_params["exclude_ids"] = []
            # --- End ID/Exclusion Handling ---

            # --- Size Check & Status Update ---
            w_seg, h_seg = WSI_object.level_dim[current_seg_params["seg_level"]]
            if w_seg * h_seg > 1e8:
                print(
                    "level_dim {} x {} is likely too large for successful segmentation, aborting".format(
                        w_seg, h_seg
                    )
                )
                df.loc[idx, "status"] = "failed_seg"
                continue

            df.loc[idx, "vis_level"] = current_vis_params["vis_level"]
            df.loc[idx, "seg_level"] = current_seg_params["seg_level"]

            # --- Segmentation & Mask Saving ---
            seg_time_elapsed = -1
            if seg:
                WSI_object, seg_time_elapsed = segment(
                    WSI_object, current_seg_params, current_filter_params
                )

            if save_mask:
                mask = WSI_object.visWSI(**current_vis_params)
                mask_path = os.path.join(mask_save_dir, slide_id + ".jpg")
                mask.save(mask_path)

            # --- Patching ---
            patch_time_elapsed = -1
            if patch:
                current_patch_params.update(
                    {
                        "patch_level": patch_level,
                        "patch_size": patch_size,
                        "step_size": step_size,
                        "save_path": patch_save_dir,
                    }
                )
                file_path, patch_time_elapsed = patching(
                    WSI_object=WSI_object, **current_patch_params
                )

            df.loc[idx, "status"] = "processed"  # Set status for WSI

        # --- COMMON POST-PROCESSING LOGIC (Stitching) ---
        stitch_time_elapsed = -1
        h5_file_path = os.path.join(patch_save_dir, slide_id + ".h5")

        # Only attempt stitching if an H5 file was generated AND the image was large (WSI)
        if stitch and os.path.isfile(h5_file_path):
            # Check the status set earlier to determine if it was a small image skip
            is_small_img = df.loc[idx, "status"] == "bypassed"

            if is_small_img:
                print(
                    "Skipping stitching for small image (no WSI object for visualization)."
                )
                stitch_time_elapsed = 0.0
            else:
                heatmap, stitch_time_elapsed = stitching(
                    h5_file_path, WSI_object, downscale=64
                )
                stitch_path = os.path.join(stitch_save_dir, slide_id + ".jpg")
                heatmap.save(stitch_path)

        print(
            "segmentation took {} seconds".format(
                seg_time_elapsed if seg_time_elapsed > 0 else 0.0
            )
        )
        print(
            "patching took {} seconds".format(
                patch_time_elapsed if patch_time_elapsed > 0 else 0.0
            )
        )
        print(
            "stitching took {} seconds".format(
                stitch_time_elapsed if stitch_time_elapsed > 0 else 0.0
            )
        )

        # Accumulate times
        if seg_time_elapsed > 0:
            seg_times += seg_time_elapsed
        if patch_time_elapsed > 0:
            patch_times += patch_time_elapsed
        if stitch_time_elapsed > 0:
            stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times


parser = argparse.ArgumentParser(description="seg and patch")
parser.add_argument(
    "--source", type=str, help="path to folder containing raw wsi image files"
)
parser.add_argument("--step_size", type=int, default=256, help="step_size")
parser.add_argument("--patch_size", type=int, default=256, help="patch_size")
parser.add_argument("--patch", default=False, action="store_true")
parser.add_argument("--seg", default=False, action="store_true")
parser.add_argument("--stitch", default=False, action="store_true")
parser.add_argument("--no_auto_skip", default=True, action="store_false")
parser.add_argument("--save_dir", type=str, help="directory to save processed data")
parser.add_argument(
    "--preset",
    default=None,
    type=str,
    help="predefined profile of default segmentation and filter parameters (.csv)",
)
parser.add_argument(
    "--patch_level", type=int, default=0, help="downsample level at which to patch"
)
parser.add_argument(
    "--process_list",
    type=str,
    default=None,
    help="name of list of images to process with parameters (.csv)",
)

if __name__ == "__main__":
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, "patches")
    mask_save_dir = os.path.join(args.save_dir, "masks")
    stitch_save_dir = os.path.join(args.save_dir, "stitches")

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None

    print("source: ", args.source)
    print("patch_save_dir: ", patch_save_dir)
    print("mask_save_dir: ", mask_save_dir)
    print("stitch_save_dir: ", stitch_save_dir)

    directories = {
        "source": args.source,
        "save_dir": args.save_dir,
        "patch_save_dir": patch_save_dir,
        "mask_save_dir": mask_save_dir,
        "stitch_save_dir": stitch_save_dir,
    }

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ["source"]:
            os.makedirs(val, exist_ok=True)

    seg_params = {
        "seg_level": -1,
        "sthresh": 8,
        "mthresh": 7,
        "close": 4,
        "use_otsu": False,
        "keep_ids": "none",
        "exclude_ids": "none",
    }
    filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
    vis_params = {"vis_level": -1, "line_thickness": 250}
    patch_params = {"use_padding": True, "contour_fn": "four_pt"}

    if args.preset:
        preset_df = pd.read_csv(os.path.join("presets", args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {
        "seg_params": seg_params,
        "filter_params": filter_params,
        "patch_params": patch_params,
        "vis_params": vis_params,
    }

    print(parameters)

    move_svs_files_to_root(args.source, "svs")

    seg_times, patch_times = seg_and_patch(
        **directories,
        **parameters,
        patch_size=args.patch_size,
        step_size=args.step_size,
        seg=args.seg,
        use_default_params=False,
        save_mask=True,
        stitch=args.stitch,
        patch_level=args.patch_level,
        patch=args.patch,
        process_list=process_list,
        auto_skip=args.no_auto_skip,
    )
