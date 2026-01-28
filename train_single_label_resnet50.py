"""
ANA Pattern Classification using ResNet50
Author: Dr. Rayan Abri
Description: Training script for automated classification of 32 ANA patterns
             using deep learning with stratified train/validation/test split.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np


def build_transforms(img_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def stratified_split(dataset: ImageFolder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Stratified train/validation/test split with fixed seed - using sklearn for reproducibility"""
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Get all indices and labels
    indices = np.arange(len(dataset))
    labels = np.array([label for _, label in dataset.samples])
    
    # First split: train vs (val+test)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, 
        test_size=(val_ratio + test_ratio), 
        random_state=seed, 
        stratify=labels
    )
    
    # Second split: val vs test (split temp 50/50)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # 50% of temp goes to test
        random_state=seed,
        stratify=temp_labels
    )
    
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def compute_class_weights(dataset: ImageFolder):
    # Count per class
    counts = np.zeros(len(dataset.classes), dtype=np.int64)
    for _, label in dataset.samples:
        counts[label] += 1
    # Inverse frequency
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def build_model(model_name: str, num_classes: int | None):
    if model_name.lower() == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        img_size = 224
        if num_classes is not None:
            in_feats = model.fc.in_features
            model.fc = nn.Linear(in_feats, num_classes)
    elif model_name.lower() in ("efficientnet_b0", "efficientnet", "efficientnetb0"):
        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        img_size = 224
        if num_classes is not None:
            in_feats = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use 'resnet50' or 'efficientnet_b0'.")
    return model, img_size


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += images.size(0)
    return running_loss / total, running_corrects / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / total
    avg_acc = running_corrects / total
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="ANA Patterns Training Pipeline")
    parser.add_argument("--dataset", type=str, default="Dataset_Augmented", help="Path to dataset root (class folders)")
    parser.add_argument("--model", type=str, default="resnet50", help="Model backbone: resnet50 or efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Build loaders and model, then exit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model and transforms
    model, img_size = build_model(args.model, num_classes=None)
    train_tfms, eval_tfms = build_transforms(img_size)

    # Dataset
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    full_dataset = ImageFolder(str(dataset_root), transform=eval_tfms)
    num_classes = len(full_dataset.classes)

    # Rebuild model with correct classifier output size
    model, img_size = build_model(args.model, num_classes=num_classes)
    train_tfms, eval_tfms = build_transforms(img_size)

    # Use different transforms for train/eval by wrapping dataset
    # We will create two ImageFolder instances pointing to same root with different transforms
    train_full = ImageFolder(str(dataset_root), transform=train_tfms)
    eval_full = ImageFolder(str(dataset_root), transform=eval_tfms)

    # Stratified splits
    train_idx, val_idx, test_idx = stratified_split(full_dataset, 0.7, 0.15, 0.15, seed=42)
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(eval_full, val_idx)
    test_ds = Subset(eval_full, test_idx)

    print(f"Split sizes → train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    print(f"Classes ({num_classes}): {full_dataset.classes}")

    if args.dry_run:
        print("Dry run complete: datasets and model initialized.")
        return

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Class weights
    class_weights = compute_class_weights(full_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with smaller LR for backbone layers, larger for classifier
    # Find classifier parameters
    if args.model.lower() == "resnet50":
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    else:
        head_params = list(model.classifier[1].parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.1")] 

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr}
    ], weight_decay=args.weight_decay)

    model.to(device)

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {"model": model.state_dict(), "epoch": epoch}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    # Save best checkpoint
    out_dir = Path("runs") / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.ckpt"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint to {ckpt_path}")

    # Evaluate on test set
    model.load_state_dict(best_state["model"]) if best_state else None
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")

    # Reports (NumPy-based to avoid sklearn/scipy dependency)
    num_classes = len(full_dataset.classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for y_true, y_pred in zip(labels, preds):
        cm[y_true, y_pred] += 1

    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    per_class_acc = np.divide(per_class_correct, per_class_total, out=np.zeros_like(per_class_correct, dtype=float), where=per_class_total!=0)

    with open(out_dir / "report.txt", "w") as f:
        f.write(f"Overall test accuracy: {test_acc:.4f}\n\n")
        f.write("Per-class accuracy:\n")
        for cls_name, acc, total in zip(full_dataset.classes, per_class_acc, per_class_total):
            f.write(f"  {cls_name}: acc={acc:.4f} (n={total})\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"Saved report to {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
