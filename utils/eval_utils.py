import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import mlflow  # <-- ADDED MLflow import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initiate_model(args, ckpt_path, device="cuda"):
    print("Init Model")
    model_dict = {
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

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if "instance_loss_fn" in key:
            continue
        ckpt_clean.update({key.replace(".module", ""): ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model


def eval(dataset, args, ckpt_path):
    """
    Evaluates the model and logs final results to MLflow.
    """
    # ====================================================================
    # MLFLOW INTEGRATION: Start a new MLflow run for the evaluation/testing phase
    # This run is separate from the training runs, or could be a nested run
    # if called from a primary experiment run.
    # We use a unique run name based on the checkpoint path.
    # ====================================================================
    eval_run_name = f"Evaluation: {os.path.basename(ckpt_path)}"
    with mlflow.start_run(run_name=eval_run_name, nested=True) as run:

        # 1. Log essential parameters and the model checkpoint path
        mlflow.log_param("eval_ckpt_path", ckpt_path)
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_classes", args.n_classes)

        # 2. Initiate model and run summary
        model = initiate_model(args, ckpt_path)

        print("Init Loaders")
        # NOTE: get_simple_loader is assumed to be defined in utils.utils
        loader = get_simple_loader(dataset)
        patient_results, test_error, auc_score, df, acc_logger = summary(
            model, loader, args
        )

        test_acc = 1.0 - test_error  # Calculate test accuracy

        # 3. Log final metrics to MLflow
        mlflow.log_metric("test_error", test_error)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_auc", auc_score)

        # Log class-wise accuracy
        for i in range(args.n_classes):
            acc, _, _ = acc_logger.get_summary(i)
            if acc is not None:
                mlflow.log_metric(f"test_class_{i}_acc", acc)

        # 4. Log the results DataFrame (predictions, probabilities) as a CSV artifact
        results_path = os.path.join(
            os.path.dirname(ckpt_path), f"{eval_run_name}_results.csv"
        )
        df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        print("test_error: ", test_error)
        print("auc: ", auc_score)

    return model, patient_results, test_error, auc_score, df


def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.0
    test_error = 0.0

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data["slide_id"]
    patient_results = {}

    # NOTE: device is assumed to be defined globally as 'cuda' or 'cpu'
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            # NOTE: model returns logits, Y_prob, Y_hat, _, results_dict
            logits, Y_prob, Y_hat, _, results_dict = model(data)

            acc_logger.log(Y_hat, label)

            probs = Y_prob.cpu().numpy()

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
            error = calculate_error(Y_hat, label)
            test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else:
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(
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

    results_dict = {"slide_id": slide_ids, "Y": all_labels, "Y_hat": all_preds}
    for c in range(args.n_classes):
        results_dict.update({"p_{}".format(c): all_probs[:, c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
