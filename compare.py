import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc,
)

# Modify input here ──────────────────────────────────────────────────
data_dir = [
    "experiments/custom_cnn/exp07_lr005_cos_50",
    "experiments/resnet18/exp02_cos",
]
data_name = [
    "CustomCNN",
    "ResNet18",
]
output_dir = "outputs/comparison"
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = ["#4C72B0", "#E8A076", "#55A868", "#C4686B", "#8172B2",
           "#8F7055", "#E9AAD7", "#8C8C8C", "#EBCF68", "#64B5CD"]

plot_choices = ["all", "metrics", "training_curve", "roc", "reliability_diagram"]


def get_color(experiments, name):
    return PALETTE[list(experiments.keys()).index(name) % len(PALETTE)]


def load_experiment(exp_dir, name):
    data = {"name": name}

    history_path = os.path.join(exp_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            data["history"] = json.load(f)

    predictions_path = os.path.join(exp_dir, "predictions.json")
    if os.path.exists(predictions_path):
        with open(predictions_path) as f:
            p = json.load(f)
            data["labels"] = np.array(p["labels"])
            data["preds"]  = np.array(p["preds"])
            data["probs"]  = np.array(p["probs"])

    cal_path = os.path.join(exp_dir, "calibration_data.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            data["calibration"] = json.load(f)

    return data


def plot_training_curve(experiments, output_dir):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, exp in experiments.items():
        if "history" not in exp:
            print(f"  [skip] no history.json for {name}")
            continue
        c = get_color(experiments, name)
        h = exp["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        ax1.plot(epochs, h["val_loss"],   label=name, color=c, lw=2)
        ax1.plot(epochs, h["train_loss"], color=c, lw=1, linestyle="--", alpha=0.4)
        ax2.plot(epochs, h["val_acc"],    label=name, color=c, lw=2)
        ax2.plot(epochs, h["train_acc"],  color=c, lw=1, linestyle="--", alpha=0.4)

    style_handles = [
        Line2D([0], [0], color="gray", lw=2,                label="Val"),
        Line2D([0], [0], color="gray", lw=1, linestyle="--", alpha=0.6, label="Train"),
    ]

    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.add_artist(ax1.legend(loc="upper right"))
    ax1.legend(handles=style_handles, loc="upper left")

    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.add_artist(ax2.legend(loc="lower right"))
    ax2.legend(handles=style_handles, loc="upper left")

    plt.tight_layout()
    out = os.path.join(output_dir, "compare_training_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training curves saved to: {out}")


def plot_metrics(experiments):
    print(f"\n{'='*70}")
    print("Model Comparison Summary")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'AUC':>10}")
    print("-" * 70)

    rows = []
    for name, exp in experiments.items():
        if "labels" not in exp:
            print(f"{name:<25} {'(no predictions.json)':>42}")
            continue
        acc       = accuracy_score(exp["labels"], exp["preds"])
        prec      = precision_score(exp["labels"], exp["preds"], zero_division=0)
        rec       = recall_score(exp["labels"], exp["preds"], zero_division=0)
        auc_score = roc_auc_score(exp["labels"], exp["probs"])
        print(f"{name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {auc_score:>10.4f}")
        rows.append((name, acc, prec, rec, auc_score))

    print(f"{'='*70}\n")



def plot_roc(experiments, output_dir):
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

    for name, exp in experiments.items():
        if "labels" not in exp:
            print(f"  [skip] no predictions.json for {name}")
            continue
        c = get_color(experiments, name)
        fpr, tpr, _ = roc_curve(exp["labels"], exp["probs"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={roc_auc:.4f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(output_dir, "compare_roc.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"ROC curve saved to: {out}")


def plot_reliability_diagram(experiments, output_dir):
    _, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Perfect calibration")

    for name, exp in experiments.items():
        if "calibration" not in exp:
            print(f"  [skip] no calibration_data.json for {name}")
            continue
        c = get_color(experiments, name)
        cal = exp["calibration"]
        bin_centers = np.array(cal["bin_centers"])
        bin_accs    = np.array([v if v is not None else np.nan for v in cal["bin_accs"]])
        valid = ~np.isnan(bin_accs)
        ax.plot(bin_centers[valid], bin_accs[valid], marker="o", color=c, lw=2, label=name)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram Comparison")
    ax.legend(loc="upper left")
    plt.tight_layout()
    out = os.path.join(output_dir, "compare_reliability_diagram.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Reliability diagram saved to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiment results")
    parser.add_argument(
        "--plot", nargs="+", default=plot_choices,
        choices=plot_choices,
        metavar="PLOT",
        help=f"Which plots to generate. Choices: {plot_choices}. Default: all."
    )
    args = parser.parse_args()

    os.makedirs(output_dir, exist_ok=True)
    plots = set(plot_choices) if "all" in args.plot else set(args.plot)

    experiments = {}
    for name, d in zip(data_name, data_dir):
        if not os.path.isdir(d):
            print(f"Warning: directory not found: {d}")
            continue
        experiments[name] = load_experiment(d, name)
        print(f"Loaded: {name}  ({d})")

    if not experiments:
        print("No valid experiment directories found.")
        return

    if "training_curve" in plots: plot_training_curve(experiments, output_dir)
    if "metrics" in plots: plot_metrics(experiments)
    if "roc" in plots: plot_roc(experiments, output_dir)
    if "reliability_diagram" in plots: plot_reliability_diagram(experiments, output_dir)
    print(f"\nDone. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
