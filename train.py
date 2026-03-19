import os
import argparse
import yaml
import random
import numpy as np
import torch
from src.dataset import dataloaders
from src.model import BaseResNet18, CustomCNN
from src.trainer import train
from src.utils import plot_history


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(model_name, cfg):
    
    model_cfg = cfg["models"].get(model_name)
    if model_cfg is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(cfg['models'].keys())}"
        )
 
    freeze = model_cfg.get("freeze_backbone", True)
    dropout = cfg["training"].get("dropout", 0.5)

    if model_name == "base_resnet18":
        return BaseResNet18(freeze_backbone=freeze)
    elif model_name == "custom_cnn":
        return CustomCNN(num_class=2, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model_name, cfg):

    print(f"\n{'*'*60}")
    print(f"Training: {model_name}")
    print(f"{'*'*60}")
 

    train_loader, val_loader = dataloaders(
        train_data_dir=cfg["data"]["train_dir"],
        split=cfg["data"]["val_split"],
        seed=cfg["data"]["seed"],
        batch_size=cfg["data"]["batch_size"]
    )

 
    model = get_model(model_name, cfg)
    # Load late_model.pth for resuming training
    # print("Load model check!")
    # model = get_model(model_name, cfg)
    # model.load_state_dict(torch.load("./outputs/custom_cnn/last_model.pth", map_location="cpu"))

    output_dir = cfg["models"][model_name]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
 

    history = train(model, train_loader, val_loader, cfg, output_dir)
    plot_history(history, output_path=os.path.join(output_dir, "history.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Train cat vs dog classifier"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["base_resnet18", "custom_cnn", "all"],
        help="Model to train (default: custom_cnn)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    args = parser.parse_args()
    if args.list_models:
        print("Available models:")
        print("  custom_cnn      — Custom CNN with 3 conv layers and 2 linear layers")
        print("  base_resnet18   — ResNet-18 with single linear classifier head")
        print("  all             — Train all models sequentially")
        return
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
 

    set_seed(cfg["data"]["seed"])
 

    if args.model == "all":
        for model_name in ["base_resnet18", "custom_cnn"]:
            train_model(model_name, cfg)
    else:
        train_model(args.model, cfg)
 
    print("\nFinish Training.")


if __name__ == "__main__":
    main()