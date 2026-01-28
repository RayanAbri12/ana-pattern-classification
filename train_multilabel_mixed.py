"""
ANA Pattern Multi-Label Classification (Mixed-Data Training)
Author: Dr. Rayan Abri

Train multi-label (sigmoid) models using both datasets:
- Trusted single-label dataset (converted to single-hot multi-label)
- Hospital multi-pattern dataset (folder names may contain multiple AC ids)

Outputs checkpoints to: model for both dataset/
"""
import argparse
import os
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

SEED = 42
NUM_CLASSES = 32
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 5
MIX_RATIO = 0.5  # share of single-label samples per epoch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

TRAIN_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

EVAL_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_mapping(single_ds: ImageFolder):
    mapping = {}
    for name, idx in single_ds.class_to_idx.items():
        m = re.findall(r"\d+", name)
        if not m:
            continue
        number = int(m[0])
        mapping[number] = idx
    if len(mapping) != len(single_ds.classes):
        print(f"Warning: mapping has {len(mapping)} entries, classes={len(single_ds.classes)}")
    return mapping


def _get_classes(ds):
    if hasattr(ds, 'classes'):
        return ds.classes
    if hasattr(ds, 'dataset') and hasattr(ds.dataset, 'classes'):
        return ds.dataset.classes
    raise AttributeError("Dataset has no 'classes' attribute")


class SingleToMultiLabelWrapper(Dataset):
    def __init__(self, base_ds: Dataset):
        self.base = base_ds
        self.num_classes = len(_get_classes(base_ds))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[label] = 1.0
        return img, target


class HospitalMultiLabelDataset(Dataset):
    def __init__(self, items, transform, num_classes, class_mapping):
        self.items = items  # list of tuples (path, label_indices)
        self.transform = transform
        self.num_classes = num_classes
        self.class_mapping = class_mapping

    @classmethod
    def from_root(cls, root, transform, num_classes, class_mapping):
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        root = Path(root)
        paths = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in img_exts]
        items = []
        skipped = 0
        for p in paths:
            labels = set()
            for num_str in re.findall(r"\d+", p.parent.name):
                num = int(num_str)
                if num in class_mapping:
                    labels.add(class_mapping[num])
            if not labels:
                skipped += 1
                continue
            items.append((p, sorted(labels)))
        print(f"Hospital dataset: kept {len(items)} images, skipped {skipped} (no valid labels)")
        return cls(items, transform, num_classes, class_mapping)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, labels = self.items[idx]
        try:
            img = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            # Return blank image and zeros to avoid crashing; caller may skip
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for l in labels:
            target[l] = 1.0
        return img, target


class SubsetFromList(Dataset):
    def __init__(self, items, transform, num_classes):
        self.items = items
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, labels = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for l in labels:
            target[l] = 1.0
        return img, target


def split_indices(n, val_ratio, seed):
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=seed, shuffle=True)
    return train_idx, val_idx


def build_model(kind: str):
    if kind == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif kind == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, log_interval=20):
    model.train()
    total_loss = 0.0
    total = 0
    for step, (imgs, targets) in enumerate(loader, 1):
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

        if log_interval and step % log_interval == 0:
            print(f"    step {step}/{len(loader)} | loss={loss.item():.4f}")
    return total_loss / max(total, 1)


def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    total_loss, total = 0.0, 0
    exact, tp, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            exact += (preds == targets).all(dim=1).sum().item()
            tp += ((preds == 1) & (targets == 1)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()
    exact_acc = exact / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return total_loss / max(total, 1), exact_acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-single', default='Dataset', help='Single-label dataset root')
    parser.add_argument('--data-hospital', default='ANA Hospital Dataset', help='Hospital multi-pattern dataset root')
    parser.add_argument('--out', default='model for both dataset', help='Output directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--mix-ratio', type=float, default=MIX_RATIO, help='Proportion of single-label samples per epoch (0-1)')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio for each source')
    parser.add_argument('--threshold', type=float, default=0.5, help='Sigmoid threshold for eval')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    set_seed(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-label dataset (trusted)
    single_full = ImageFolder(args.data_single, transform=TRAIN_TFMS)
    class_mapping = build_class_mapping(single_full)
    train_idx_s, val_idx_s = split_indices(len(single_full), args.val_ratio, SEED)
    train_single = SingleToMultiLabelWrapper(torch.utils.data.Subset(single_full, train_idx_s))
    single_val_ds = ImageFolder(args.data_single, transform=EVAL_TFMS)
    val_single = SingleToMultiLabelWrapper(torch.utils.data.Subset(single_val_ds, val_idx_s))

    # Hospital dataset (multi-pattern)
    hospital_items = HospitalMultiLabelDataset.from_root(args.data_hospital, None, NUM_CLASSES, class_mapping).items
    if len(hospital_items) == 0:
        raise RuntimeError('No labeled hospital images found')
    train_idx_h, val_idx_h = split_indices(len(hospital_items), args.val_ratio, SEED)
    train_hosp = SubsetFromList([hospital_items[i] for i in train_idx_h], TRAIN_TFMS, NUM_CLASSES)
    val_hosp = SubsetFromList([hospital_items[i] for i in val_idx_h], EVAL_TFMS, NUM_CLASSES)

    # Combined train loader with weighted sampling to balance sources
    concat_train = ConcatDataset([train_single, train_hosp])
    n_s, n_h = len(train_single), len(train_hosp)
    w_single = args.mix_ratio / max(n_s, 1)
    w_hosp = (1 - args.mix_ratio) / max(n_h, 1)
    weights = [w_single] * n_s + [w_hosp] * n_h
    sampler = WeightedRandomSampler(weights, num_samples=n_s + n_h, replacement=True)
    train_loader = DataLoader(concat_train, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    # Validation loader combines both sources
    concat_val = ConcatDataset([val_single, val_hosp])
    val_loader = DataLoader(concat_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for kind in ['resnet50', 'efficientnet_b0']:
        print(f"\n=== Training {kind} on mixed data ===")
        model = build_model(kind).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

        best_loss = float('inf')
        patience_left = args.patience
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_exact, val_f1 = evaluate(model, val_loader, criterion, threshold=args.threshold)
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_exact={val_exact*100:.2f}% val_f1={val_f1:.4f}")

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                patience_left = args.patience
                ckpt_path = out_dir / f"{kind}_multilabel_mixed.ckpt"
                torch.save({'model': model.state_dict()}, ckpt_path)
                print(f"  ✓ Saved {ckpt_path}")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("  Early stopping")
                    break

    print("\nDone. Checkpoints saved in:", out_dir.resolve())


if __name__ == '__main__':
    main()
