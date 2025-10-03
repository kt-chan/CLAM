import os
import subprocess
import sys
import argparse
import mlflow
import re

# IMPORTANT: Set this to the directory where you cloned the CLAM repository
CLAM_ROOT = "/path/to/CLAM" 

def parse_metrics_from_output(output):
    """
    Parses key metrics (e.g., AUC, Accuracy) from the console output 
    of the CLAM training script. You will need to inspect the CLAM 
    main.py output to match the exact patterns.
    """
    metrics = {}
    
    # Example pattern matching (adjust as necessary)
    # The CLAM main.py typically outputs final validation metrics.
    auc_match = re.search(r"Validation AUC:\s*([\d\.]+)", output)
    if auc_match:
        metrics["val_auc"] = float(auc_match.group(1))

    # Look for the last checkpoint path for artifact logging
    checkpoint_match = re.search(r"Saved final model checkpoint at:\s*(.+?\.pt)", output)
    checkpoint_path = checkpoint_match.group(1).strip() if checkpoint_match else None
    
    return metrics, checkpoint_path

def run_clam_with_mlflow(args):
    # Construct the original CLAM training command
    clam_cmd = [
        "python", os.path.join(CLAM_ROOT, "main.py"),
        "--drop_out", "0.25",
        "--early_stopping",
        "--lr", str(args.lr),
        "--k", "10",
        "--exp_code", args.exp_code,
        "--weighted_sample",
        "--bag_loss", "ce",
        "--inst_loss", "svm",
        "--task", args.task,
        "--model_type", args.model_type,
        "--log_data",  # Keep TensorBoard logging on for detailed view
        "--subtyping", # Use for MHIST multi-class classification
        "--data_root_dir", args.data_root_dir,
        "--embed_dim", str(args.embed_dim),
        # You may need to specify the fold if running a single cross-validation fold:
        "--opt_fold", str(args.fold) 
    ]

    # Start MLflow run
    with mlflow.start_run(run_name=f"CLAM_Fold_{args.fold}"):
        
        # Log all parameters from the MLproject file
        mlflow.log_params(vars(args))

        print(f"Executing CLAM command: {' '.join(clam_cmd)}")
        
        # Execute the CLAM script
        # We capture stdout to parse metrics later
        process = subprocess.run(clam_cmd, cwd=CLAM_ROOT, capture_output=True, text=True, check=False)
        
        print("CLAM Script Output:\n" + process.stdout)
        print("CLAM Script Error (if any):\n" + process.stderr)

        if process.returncode != 0:
            print("CLAM Training failed. Exiting MLflow run.")
            mlflow.set_tag("status", "failed")
            return

        # 1. Parse and log final metrics
        metrics, checkpoint_path = parse_metrics_from_output(process.stdout)
        mlflow.log_metrics(metrics)
        print(f"Logged Metrics to MLflow: {metrics}")

        # 2. Log the model checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            mlflow.log_artifact(checkpoint_path, "model_checkpoint")
            print(f"Logged model checkpoint as artifact: {checkpoint_path}")
        else:
            # Fallback: CLAM saves results in results/{exp_code}
            results_dir = os.path.join(CLAM_ROOT, "results", args.exp_code)
            if os.path.exists(results_dir):
                mlflow.log_artifact(results_dir, "clam_results")
                print(f"Logged entire results directory: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--exp_code', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--model_type', type=str, default='clam_sb')
    parser.add_argument('--embed_dim', type=int, default=1024)
    args = parser.parse_args()
    
    # IMPORTANT: Update the CLAM_ROOT variable in this script to point to 
    # the actual location of the CLAM repository.
    
    run_clam_with_mlflow(args)