import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    classification_report,
    confusion_matrix,
)


def predict(model, test_loader, device):
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Predicting"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, class_names=None, output_path=None):
    # Calculate metrics: accuracy, precision, recall

    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall":    recall_score(labels, preds),
    }

    lines = []
    lines.append(f'Accuracy: {metrics["accuracy"]:.4f}')
    lines.append(f'Precision: {metrics["precision"]:.4f}')
    lines.append(f'Recall: {metrics["recall"]:.4f}')

    if class_names:
        report = classification_report(labels, preds, target_names=class_names, output_dict=True)
        lines.append(f"\nClassification Report:")
        lines.append(f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10}")
        for cls in class_names:
            r = report[cls]
            lines.append(f"{cls:>20} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f}")

    for line in lines:
        print(line)

    if output_path:
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Metrics saved to: {output_path}")


    return metrics


def plot_history(history, output_path):
    # Plot training and validation loss and accuracy curves
    
    epochs = range(1, len(history["train_loss"]) + 1)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="coral")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(range(0, len(history["train_loss"]) + 1, 10))
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="steelblue")
    ax2.plot(epochs, history["val_acc"], label="Validation Accuracy", color="coral")

    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc = max(history["val_acc"])
    ax2.annotate(f"Best: {best_acc:.4f}",
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + 1, best_acc - 0.02),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=9)

    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(range(0, len(history["train_loss"]) + 1, 10))
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to: {output_path}")
    

def plot_confusion_matrix(labels, preds, class_names, output_path):

    cm = confusion_matrix(labels, preds)

    _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_roc(labels, probs, model_name, output_path):

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
 
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to: {output_path}")


def plot_calibration(labels, probs, model_name, output_dir, n_bins=10):

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            bin_accs.append(np.nan)
            bin_counts.append(0)
        else:
            bin_accs.append(labels[mask].mean())
            bin_counts.append(int(mask.sum()))

    bin_accs = np.array(bin_accs, dtype=float)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram as curve
    valid = ~np.isnan(bin_accs)
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect calibration")
    ax1.plot(bin_centers[valid], bin_accs[valid], marker="o", color="steelblue",
             lw=2, label=model_name)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives (Class: dog)")
    ax1.set_title("Reliability Diagram")
    ax1.legend(loc="upper left")

    # Probability distribution
    ax2.hist(probs, bins=20, color="steelblue", edgecolor="white")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Probability Distribution")

    plt.suptitle(f"Calibration Analysis — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration.png"), dpi=150)
    plt.close()
    print(f"Calibration plot saved to: {os.path.join(output_dir, 'calibration.png')}")

    cal_data = {
        "model": model_name,
        "bin_centers": bin_centers.tolist(),
        "bin_accs": [v if not np.isnan(v) else None for v in bin_accs],
        "bin_counts": bin_counts,
        "confs": probs.tolist(),
    }
    cal_path = os.path.join(output_dir, "calibration_data.json")
    with open(cal_path, "w") as f:
        json.dump(cal_data, f)
    print(f"Calibration data saved to: {cal_path}")