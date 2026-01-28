"""
ANA Pattern Single-Label Ensemble Evaluation
Author: Dr. Rayan Abri

Evaluates ensemble of ResNet50 + EfficientNet-B0 on single-label test set
Uses softmax activation and averages predictions from both models
"""
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision


def build_transforms(img_size: int = 224):
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return eval_tfms


def stratified_split(dataset: ImageFolder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    indices = np.arange(len(dataset))
    labels = np.array([label for _, label in dataset.samples])
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def build_resnet50(num_classes: int):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def build_efficientnet_b0(num_classes: int):
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = Path("Dataset")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    eval_tfms = build_transforms(224)
    full_dataset = ImageFolder(str(dataset_root), transform=eval_tfms)
    classes = full_dataset.classes
    num_classes = len(classes)

    # Consistent split
    _, _, test_idx = stratified_split(full_dataset, 0.7, 0.15, 0.15, seed=42)
    test_ds = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Build models and load checkpoints
    resnet = build_resnet50(num_classes)
    effnet = build_efficientnet_b0(num_classes)
    resnet_ckpt = Path("runs/resnet50_20260117_000621/best.ckpt")
    effnet_ckpt = Path("runs/efficientnet_b0_20260117_184327/best.ckpt")

    if not resnet_ckpt.exists():
        raise FileNotFoundError(f"ResNet50 checkpoint not found: {resnet_ckpt}")
    if not effnet_ckpt.exists():
        raise FileNotFoundError(f"EfficientNet-B0 checkpoint not found: {effnet_ckpt}")

    resnet_state = torch.load(resnet_ckpt, map_location="cpu")
    effnet_state = torch.load(effnet_ckpt, map_location="cpu")
    resnet.load_state_dict(resnet_state["model"])  # saved as {"model": state_dict, "epoch": N}
    effnet.load_state_dict(effnet_state["model"])  # same format
    resnet.to(device)
    effnet.to(device)
    resnet.eval()
    effnet.eval()

    # Evaluate ensemble: average softmax probabilities
    criterion = nn.CrossEntropyLoss()  # not used for loss here, but consistent
    total, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits_r = resnet(images)
            logits_e = effnet(images)
            probs_r = torch.softmax(logits_r, dim=1)
            probs_e = torch.softmax(logits_e, dim=1)
            probs_avg = (probs_r + probs_e) / 2.0
            preds = torch.argmax(probs_avg, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total if total > 0 else 0.0
    print(f"Ensemble (ResNet50 + EfficientNet-B0) Test Accuracy: {acc*100:.2f}% ({correct}/{total})")

    # Per-class metrics and confusion matrix
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for y_true, y_pred in zip(all_labels, all_preds):
        cm[y_true, y_pred] += 1
    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    per_class_acc = np.divide(per_class_correct, per_class_total, out=np.zeros_like(per_class_correct, dtype=float), where=per_class_total!=0)

    out_dir = Path("runs") / f"ensemble_resnet50_efficientnet_b0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.txt", "w") as f:
        f.write(f"Ensemble Test Accuracy: {acc*100:.2f}% ({correct}/{total})\n\n")
        f.write("Per-class accuracy:\n")
        for cls_name, a, total_n in zip(classes, per_class_acc, per_class_total):
            f.write(f"  {cls_name}: acc={a:.4f} (n={total_n})\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"Saved ensemble report to {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
