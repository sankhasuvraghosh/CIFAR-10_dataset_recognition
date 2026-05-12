# CIFAR-10 Image Classification
### Convolutional Neural Network · PyTorch · 10-Class RGB Image Recognition

A deep learning model that classifies real-world colour images across 10 categories — airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks — trained on the CIFAR-10 benchmark dataset. Built with PyTorch, the project demonstrates a production-style CNN pipeline with data augmentation, batch normalization, dropout regularization, and learning rate scheduling.

> A meaningful step up from grayscale digit recognition — CIFAR-10 uses 32×32 **RGB** images of real-world objects, making it a standard benchmark for evaluating CNN architectures.

---

## Demo

```
Enter the index of the CIFAR-10 test image (0–9999): 512

Actual label:    dog
Model prediction: dog  ✓
```

![Sample prediction](assets/sample_prediction.png)
> <img width="366" height="113" alt="image" src="https://github.com/user-attachments/assets/0f809283-a76d-4905-991d-61c495c263b3" />


---

## Model Architecture

```
Input (3×32×32)
    │
    ▼
Conv Block 1 → Conv2d(3→32)   + BatchNorm + ReLU + MaxPool  →  32×16×16
Conv Block 2 → Conv2d(32→64)  + BatchNorm + ReLU + MaxPool  →  64×8×8
Conv Block 3 → Conv2d(64→128) + BatchNorm + ReLU + MaxPool  → 128×4×4
Conv Block 4 → Conv2d(128→256)+ BatchNorm + ReLU            → 256×4×4
    │
    ▼
Flatten → Linear(4096→512) + ReLU + Dropout(0.5)
        → Linear(512→256)  + ReLU + Dropout(0.3)
        → Linear(256→10)
    │
    ▼
Output (10 classes)
```

| Component         | Detail                              |
|-------------------|-------------------------------------|
| Conv blocks       | 4 (channels: 3→32→64→128→256)      |
| Regularization    | BatchNorm + Dropout (0.5, 0.3)      |
| Optimizer         | Adam (lr=0.001, weight_decay=1e-4)  |
| Loss function     | CrossEntropyLoss                    |
| LR Scheduler      | StepLR (step=10, γ=0.5)            |
| Batch size        | 64                                  |
| Epochs            | 30                                  |

---

## Results

| Metric          | Value         |
|-----------------|---------------|
| Test Accuracy   | **~85–88%**   |
| Dataset         | CIFAR-10      |
| Training images | 50,000        |
| Test images     | 10,000        |
| Image size      | 32×32 RGB     |

> Replace with your actual test accuracy after training.

---

## Data Augmentation

A key feature of this project is applying augmentation **only to the training set**, keeping the test set clean — the correct approach for unbiased evaluation.

```python
# Training: augmented for better generalisation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test: no augmentation — evaluate on clean data only
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

---

## Project Structure

```
cifar-10_dataset_recognition/
│
├── cifar-10_dataset_recognition.py    # Main script — training, evaluation, prediction
├── cifar_10_model.pth       # Saved model weights (generated after training)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- GPU recommended (CUDA-compatible) — the script auto-detects and uses it if available

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sankhasuvraghosh/cifar10-image-classification.git
cd cifar10-image-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and run
python cifar10_classifier.py
```

The CIFAR-10 dataset (~170 MB) will be downloaded automatically on first run.

---

## How It Works

1. **Data loading** — CIFAR-10 is loaded via `torchvision` with separate train/test splits (50k/10k images).
2. **Augmentation** — Training images are randomly flipped, cropped, and colour-jittered to improve generalisation. Test images are untouched.
3. **Architecture** — Four convolutional blocks progressively extract spatial features (3→256 channels), followed by a 3-layer fully connected classifier with dropout.
4. **Training** — 30 epochs with Adam optimizer, weight decay for L2 regularisation, and a StepLR scheduler that halves the learning rate every 10 epochs.
5. **Evaluation** — Accuracy measured on the held-out test set — data the model never saw during training.
6. **Inference** — Enter any test image index to visualise the image and compare the true vs predicted class label.

---

Classes

| Label | Class      | Label | Class  |
|-------|------------|-------|--------|
| 0     | Airplane   | 5     | Dog    |
| 1     | Automobile | 6     | Frog   |
| 2     | Bird       | 7     | Horse  |
| 3     | Cat        | 8     | Ship   |
| 4     | Deer       | 9     | Truck  |

---

## Technologies

| Tool      |                     Purpose        |
|-------------------|----------------------------|
| [PyTorch](https://pytorch.org/) | Model building, training, inference |
| [Torchvision](https://pytorch.org/vision/) | CIFAR-10 dataset, transforms, augmentation |
| [Matplotlib](https://matplotlib.org/) | Image visualisation |
| CUDA (optional) | GPU acceleration |

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
```

---





## Author

**Sankha Suvra Ghosh**
[GitHub](https://github.com/sankhasuvraghosh) · 
---
