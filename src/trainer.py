import os
import json
import time
from tqdm import tqdm, trange
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="  Training")

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()


        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / total
    acc = correct / total
    return epoch_loss, acc


def evaluate(model, val_loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Validating"):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)

            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = total_loss / total
    acc = correct / total
    return epoch_loss, acc


def train(model, train_loader, val_loader, cfg, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"\nUsing device: {device}")
    model = model.to(device)

    lr = cfg["training"].get("learning_rate", 0.001)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
        )

    epochs = cfg["training"]["epochs"]
    scheduler_name = cfg["training"].get("scheduler", "none")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    else:
        scheduler = None

    print(f"Scheduler: {scheduler_name}")

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0


    for epoch in trange(1, epochs + 1):
        start = time.time()
        tqdm.write(f"\nEpoch {epoch}/{epochs}")
        # Train and validate
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        runtime = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tqdm.write(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        tqdm.write(f"Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}")
        tqdm.write(f"LR: {optimizer.param_groups[0]['lr']:.6f}  Time: {runtime:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))


    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pth"))

    # Save history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Models saved to: {output_dir}")

    return history
