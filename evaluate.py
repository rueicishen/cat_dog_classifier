import os
import json
import argparse
import yaml
import torch
import numpy as np
from torchvision import datasets
from src.dataset import dataloaders
from src.model import BaseResNet18, CustomCNN
from src.trainer import get_device
from src.utils import predict, compute_metrics, plot_confusion_matrix, plot_roc, plot_calibration


def load_model(model_name, cfg, device):
    
    output_dir = cfg["models"][model_name]["output_dir"]
    checkpoint_path = os.path.join(output_dir, "best_model.pth") # get the best model
 
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
        )
 
    freeze = cfg["models"][model_name].get("freeze_backbone", True)
 
    if model_name == "base_resnet18":
        model = BaseResNet18(freeze_backbone=freeze)
    elif model_name == "custom_cnn":
        model = CustomCNN(num_class=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")
 
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
 
    print(f"Loaded checkpoint: {checkpoint_path}")
 
    return model


def evaluate_model(model_name, cfg, device, val_loader, class_names):
    print(f"\n{'*'*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'*'*60}")
 
    output_dir = cfg["models"][model_name]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
 
 
    model = load_model(model_name, cfg, device)
    labels, preds, probs = predict(model, val_loader, device)

    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w") as f:
        json.dump({
            "labels": labels.tolist(),
            "preds":  preds.tolist(),
            "probs":  probs.tolist(),
        }, f)

    metrics = compute_metrics(labels, preds, class_names=class_names,
                              output_path=os.path.join(output_dir, "metrics.txt")
    )
    plot_confusion_matrix(labels, preds, class_names=class_names,
                          output_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    plot_roc(labels, probs, model_name=model_name,
             output_path=os.path.join(output_dir, "roc_curve.png")
    )
    plot_calibration(labels, probs, model_name=model_name, output_dir=output_dir)

    return {
        "labels":  labels,
        "preds":   preds,
        "probs":   probs,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cat vs dog classifier"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["base_resnet18", "custom_cnn", "all"],
        help="Model to evaluate (default: custom_cnn)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    args = parser.parse_args()
 

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
 
    device = get_device()
    print(f"Using device: {device}")
 

    _, val_loader = dataloaders(
        train_data_dir=cfg["data"]["train_dir"],
        split=cfg["data"]["val_split"],
        seed=cfg["data"]["seed"],
        batch_size=cfg["data"]["batch_size"]
    )
    class_names = datasets.ImageFolder(cfg["data"]["train_dir"]).classes
 
    if args.model == "all":
        results = {}
        for model_name in ["base_resnet18", "custom_cnn"]:
            results[model_name] = evaluate_model(
                model_name, cfg, device, val_loader, class_names
            )
        print(f"\n{'*'*60}")
        print("Model Comparison Summary")
        print(f"{'*'*60}")
        print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 52)
        for model_name, result in results.items():
            m = result["metrics"]
            print(
                f"{model_name:<20} "
                f"{m['accuracy']:>10.4f} "
                f"{m['precision']:>10.4f} "
                f"{m['recall']:>10.4f}"
            )
    else:
        evaluate_model(args.model, cfg, device, val_loader, class_names)
 
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()