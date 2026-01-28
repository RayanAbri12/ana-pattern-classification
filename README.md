# ANA Pattern Classification using Deep Learning

**Author:** Dr. Rayan Abri

Automated classification of Antinuclear Antibody (ANA) patterns from immunofluorescence microscopy images using deep learning ensemble models.

## Overview

This repository contains the training and evaluation code for automated ANA pattern classification using:

- **Single-label models**: ResNet50 and EfficientNet-B0 trained on trusted single-label dataset
- **Multi-label models**: Mixed-data ensemble trained on both single-label and hospital multi-pattern data
- **32 ANA pattern classes** (AC-0 to AC-31)

### Dataset

- **Single-label Dataset**: 289 images, 32 classes
- **Hospital Multi-pattern Dataset**: 1122 images with folder-based multi-label annotations
- **Training Split**: 70% train / 15% validation / 15% test

### Prerequisites

```bash
Python 3.8+
PyTorch 2.6.0
torchvision 0.21.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ana-pattern-classification.git
cd ana-pattern-classification

# Install dependencies
pip install -r requirements.txt
```

### Training Single-Label Models

**ResNet50:**

```bash
python train_single_label_resnet50.py \
    --data Dataset \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4
```

**EfficientNet-B0:**

```bash
python train_single_label_efficientnet_b0.py \
    --data Dataset \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4
```

### Training Multi-Label Models (Mixed-Data)

```bash
python train_multilabel_mixed.py \
    --data-single Dataset \
    --data-hospital "ANA Hospital Dataset" \
    --mix-ratio 0.6 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --val-ratio 0.15
```

**Parameters:**

- `--mix-ratio`: Proportion of single-label samples per epoch (0.6 = 60%)
- `--val-ratio`: Validation split ratio (0.15 = 15%)
- `--threshold`: Sigmoid threshold for evaluation (default: 0.5)

### Evaluation

```bash
python evaluate_multilabel.py
```

## Model Architecture

### Single-Label Models

- **Architecture**: ResNet50 / EfficientNet-B0
- **Pretrained**: ImageNet weights
- **Loss**: CrossEntropyLoss with class weights
- **Activation**: Softmax
- **Output**: 32 classes

### Multi-Label Models (Mixed-Data)

- **Architecture**: ResNet50 + EfficientNet-B0 Ensemble
- **Training Data**:
  - Single-label: 202 images (60% per batch)
  - Hospital multi-pattern: 953 images (40% per batch)
- **Loss**: BCEWithLogitsLoss
- **Activation**: Sigmoid (threshold = 0.3)
- **Output**: 32 classes (multi-label)
- **Ensemble**: Average logits from both models

## Evaluation Metrics

The multi-label model is evaluated using:

1. **Exact Match Accuracy**: All predicted labels must match ground truth
2. **Hamming Loss**: Fraction of labels incorrectly predicted
3. **Micro F1-Score**: Treats each label prediction independently
4. **Weighted F1-Score**: Weighted by class frequency
5. **Error Distribution**: Analysis of off-by-N label errors
6. **Per-Class Metrics**: Precision, Recall, F1 for each AC pattern

## Clinical Application

This model is designed for automated ANA pattern classification in clinical laboratories:

- **Use Case**: Diagnostic aid for immunofluorescence microscopy
- **Multi-pattern Detection**: Identifies multiple ANA patterns simultaneously
- **Confidence Scoring**: Provides probability scores for all patterns
- **Clinical Validation**: 82.25% exact match on validation set

**Disclaimer**: This tool is for research and diagnostic aid purposes. Always confirm results with clinical expertise.

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{abri2026ana,
  title={Automated ANA Pattern Classification using Deep Learning Ensemble Models},
  author={Abri, Rayan},
  year={2026},
  note={GitHub repository: https://github.com/RayanAbri12/ana-pattern-classification}
}
```

Or cite as:

> Abri, R. (2026). Automated ANA Pattern Classification using Deep Learning Ensemble Models.
> GitHub repository: https://github.com/RayanAbri12/ana-pattern-classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Dr. Rayan Abri**

For questions or collaborations, please open an issue on GitHub.

### Key Features

- Multi-label classification with sigmoid activation
- Mixed-data training strategy (single-label + multi-pattern)
- Weighted sampling for balanced training
- Ensemble approach for robust predictions
- Validation on held-out test set (no data leakage)

### Training Strategy

1. Load single-label dataset (289 images, 32 classes)
2. Load hospital multi-pattern dataset (1122 images, multi-label)
3. Convert single-label to single-hot multi-label format
4. Parse folder names for multi-pattern annotations
5. Use weighted sampling (60% single-label, 40% hospital)
6. Train with early stopping (patience=5)
7. Ensemble predictions from ResNet50 + EfficientNet-B0

## Links

- **Deployment**: [HuggingFace Spaces](https://huggingface.co/spaces/Rayan1201/ana-pattern-classifier)
- **Paper**: (Coming soon)
- **Dataset**: Contact author for access
