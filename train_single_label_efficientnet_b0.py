"""
Train EfficientNet-B0 on ANA Pattern Dataset
Compare to ResNet50 baseline (85.00% ± 6.98%)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
NUM_CLASSES = 32
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 5
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DATASET_PATH = 'Dataset'

def build_transforms_efficientnet(img_size=224):
    """EfficientNet typically uses 224 or 240 for B0"""
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tfms, eval_tfms

def compute_class_weights(dataset):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    # Handle both ImageFolder and Subset
    if hasattr(dataset, 'samples'):
        for _, label in dataset.samples:
            counts[label] += 1
    else:
        # It's a Subset
        for idx in dataset.indices:
            _, label = dataset.dataset.samples[idx]
            counts[label] += 1
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    return running_loss / total, correct / total

def build_efficientnet_model(num_classes):
    """Build EfficientNet-B0 with custom classifier"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def main():
    print("=" * 80)
    print("EFFICIENTNET-B0 BASELINE (Single Split, 70/15/15)")
    print("=" * 80)
    print()
    
    # Load dataset
    full_dataset = ImageFolder(DATASET_PATH)
    print(f"Dataset: {len(full_dataset)} images, {len(full_dataset.classes)} classes")
    
    # Create split
    train_tfms, eval_tfms = build_transforms_efficientnet()
    
    all_indices = np.arange(len(full_dataset))
    all_labels = np.array([label for _, label in full_dataset.samples])
    
    # 70/15/15 split
    train_idx, temp_idx, _, temp_labels = train_test_split(
        all_indices, all_labels,
        test_size=0.3,
        random_state=SEED,
        stratify=all_labels
    )
    
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_labels
    )
    
    # Create datasets
    train_full = ImageFolder(DATASET_PATH, transform=train_tfms)
    eval_full = ImageFolder(DATASET_PATH, transform=eval_tfms)
    
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(eval_full, val_idx)
    test_ds = Subset(eval_full, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print()
    
    # Build model
    model = build_efficientnet_model(NUM_CLASSES)
    model.to(DEVICE)
    print(f"Model: EfficientNet-B0 (pretrained ImageNet)")
    print(f"Device: {DEVICE}")
    print()
    
    # Class weights
    class_weights = compute_class_weights(train_ds)
    class_weights = class_weights.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with different LRs
    # For EfficientNet, features are in model.features, classifier is model.classifier
    backbone_params = list(model.features.parameters())
    head_params = list(model.classifier.parameters())
    
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": LEARNING_RATE * 0.1},
        {"params": head_params, "lr": LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0
    
    print("Training...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate on test
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"EfficientNet-B0 Test Accuracy: {test_acc*100:.2f}% ({int(test_acc*len(test_idx))}/{len(test_idx)} correct)")
    print(f"Best Val Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print()
    print(f"COMPARISON TO RESNET50:")
    print(f"  ResNet50 (single split):  79.55% (35/44)")
    print(f"  ResNet50 (5-fold mean):   85.00% ± 6.98%")
    print(f"  EfficientNet-B0:          {test_acc*100:.2f}%")
    
    if test_acc > 0.79:
        diff = (test_acc - 0.7955) * 100
        print(f"\n  Improvement over ResNet50 single: +{diff:.2f}%")
    else:
        diff = (0.7955 - test_acc) * 100
        print(f"\n  Below ResNet50 single: -{diff:.2f}%")
    
    print()
    print("=" * 80)
    
    # Save model
    run_id = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = Path('runs') / f'efficientnet_b0_{run_id}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({'model': model.state_dict(), 'epoch': best_epoch}, ckpt_dir / 'best.ckpt')
    print(f"Checkpoint saved to: {ckpt_dir / 'best.ckpt'}")

if __name__ == "__main__":
    main()
