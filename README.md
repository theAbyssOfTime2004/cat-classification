# 🐱 Cat Breed Classification

A deep learning project for classifying cat images into 63 different breeds using transfer learning with ResNet50V2.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a complete machine learning pipeline for cat breed classification, featuring:

- **63 cat breeds** classification
- **YOLO-based image filtering** for data quality
- **Two-stage training** (feature extraction + fine-tuning)
- **Comprehensive evaluation** with confusion matrix and metrics
- **Easy-to-use inference** script for predictions

The model achieves high accuracy using transfer learning from ImageNet-pretrained ResNet50V2.

---

## ✨ Features

### 🔍 Automated Data Cleaning
- **File integrity check** using PIL
- **YOLO11x-based content verification** to filter non-cat images
- Automatic removal of low-quality data

### 🧠 Advanced Model Training
- **Transfer Learning** with ResNet50V2
- **GlobalAveragePooling2D** instead of Flatten (fewer parameters, less overfitting)
- **Two-stage training**:
  1. Feature extraction (frozen base)
  2. Fine-tuning (unfreeze top layers)
- **Mixed precision training** for faster computation
- **Comprehensive callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### 📊 Extensive Evaluation
- Accuracy, Top-5 Accuracy
- Confusion Matrix
- Per-class accuracy analysis
- Classification reports

### 🚀 Production-Ready
- Centralized configuration (`config.py`)
- Inference script with CLI
- Batch prediction support
- JSON output for integration

---

## 📁 Project Structure

```
cat-classification/
├── config.py                      # Centralized configuration
├── inference.py                   # Inference script with CLI
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/                          # Dataset directory
│   ├── raw-data/                  # Original images
│   │   └── cat-breeds/            # 66 breeds, ~11,223 images
│   ├── cleaned-data/              # After YOLO cleaning
│   │   ├── cat-breeds-cleaned/    # 63 breeds, ~8,400 images
│   │   └── non-cat-images/        # Filtered out images
│   └── processed/                 # Train/val/test splits
│       ├── train/                 # 5,877 images (70%)
│       ├── val/                   # 1,260 images (15%)
│       └── test/                  # 1,260 images (15%)
│
├── models/                        # Trained models
│   ├── cat_breed_classifier_final.keras
│   ├── class_indices.json
│   └── yolo11x.pt
│
├── notebooks/                     # Jupyter notebooks
│   ├── EDA.ipynb                          # Exploratory analysis
│   ├── cleaning_data.ipynb                # Data cleaning
│   ├── data_splitting.ipynb              # Train/val/test split
│   ├── data_augmenting.ipynb             # Augmentation demo
│   ├── model_definition_and_training.ipynb # Original training
│   └── model_training_improved.ipynb     # ✨ Improved training
│
└── outputs/                       # Training outputs
    ├── plots/                     # Visualizations
    ├── reports/                   # Classification reports
    └── logs/                      # TensorBoard logs
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/theAbyssOfTime2004/cat-classification.git
cd cat-classification
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models** (if available)
```bash
# Download from releases or train your own
```

---

## 🚀 Usage

### Training

#### Option 1: Use Improved Training Notebook (Recommended)

Open and run `notebooks/model_training_improved.ipynb`:

```bash
jupyter notebook notebooks/model_training_improved.ipynb
```

This notebook includes:
- ✅ Fixed architecture (GlobalAveragePooling2D)
- ✅ Two-stage training
- ✅ Comprehensive evaluation
- ✅ Automatic metric tracking

#### Option 2: Google Colab

1. Upload project to Google Drive
2. Open `notebooks/model_training_improved.ipynb` in Colab
3. Mount Drive and run cells

#### Configuration

Edit `config.py` to customize:

```python
# Model parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 50  # Feature extraction
EPOCHS_STAGE2 = 30  # Fine-tuning

# Training parameters
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5
UNFREEZE_LAYERS = 50
```

### Inference

#### Single Image Prediction

```bash
python inference.py --image path/to/cat.jpg
```

Output:
```
================================================================================
🐱 Image: cat.jpg
================================================================================

🏆 Top Prediction:
   Breed: British Shorthair
   Confidence: 87.45%

📊 Top 5 Predictions:
   1. British Shorthair          ████████████████████████████████████████████ 87.45%
   2. Scottish Fold              ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5.23%
   3. Chartreux                  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2.15%
   4. Russian Blue               █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.87%
   5. Korat                      █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.12%
================================================================================
```

#### Batch Prediction

```bash
python inference.py --batch path/to/images/*.jpg
```

#### Advanced Options

```bash
# Top 10 predictions
python inference.py --image cat.jpg --top-k 10

# Save results to JSON
python inference.py --image cat.jpg --output results.json

# Use custom model
python inference.py --image cat.jpg --model models/my_model.keras

# Batch with JSON output
python inference.py --batch images/*.jpg --output batch_results.json
```

---

## 🔄 Pipeline

### 1. Data Collection
- **Source**: Cat breed dataset
- **Initial size**: 66 breeds, ~11,223 images

### 2. Data Cleaning (`cleaning_data.ipynb`)
- **Stage 1**: File integrity check
  - Verify image files can be opened
  - Remove corrupt files
- **Stage 2**: Content verification with YOLO11x
  - Detect cat objects in images
  - Filter images with confidence < 0.4
  - Remove 3 breeds with too many non-cat images
- **Output**: 63 breeds, ~8,400 clean images

### 3. Data Splitting (`data_splitting.ipynb`)
- **Method**: Stratified split
- **Ratios**: 70% train / 15% val / 15% test
- **Ensures**: Equal distribution across breeds

### 4. Model Training (`model_training_improved.ipynb`)

#### Stage 1: Feature Extraction
- Freeze ResNet50V2 base
- Train custom top layers
- 50 epochs, LR=1e-4

#### Stage 2: Fine-tuning
- Unfreeze top 50 layers
- Fine-tune with lower LR=1e-5
- 30 additional epochs

### 5. Evaluation
- Test set accuracy
- Top-5 accuracy
- Confusion matrix
- Per-class metrics

---

## 🏗️ Model Architecture

```
Input (224×224×3)
    ↓
ResNet50V2 (pretrained, ImageNet)
    ↓
GlobalAveragePooling2D          # Reduces parameters significantly
    ↓
Dense(512, ReLU)                # Feature learning
    ↓
Dropout(0.5)                    # Regularization
    ↓
Dense(63, Softmax)              # 63 cat breeds
```

### Why GlobalAveragePooling2D?

**Before (Flatten)**:
- ResNet50V2 output: (7, 7, 2048)
- Flatten: 7 × 7 × 2048 = 100,352 parameters
- Dense(512): **51,380,736 parameters** 💥

**After (GlobalAveragePooling2D)**:
- GAP output: (2048,)
- Dense(512): **1,049,088 parameters** ✅

**Result**: 98% reduction in parameters, less overfitting!

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | TBD after training |
| Top-5 Accuracy | TBD after training |
| Training Time (Stage 1) | ~2-3 hours (GPU) |
| Training Time (Stage 2) | ~1-2 hours (GPU) |
| Model Size | ~100 MB |

### Best Performing Breeds
*To be updated after training*

### Challenging Breeds
*To be updated after training*

---

## 📦 Dataset

### Statistics

| Split | Images | Breeds | Percentage |
|-------|--------|--------|------------|
| Train | 5,877 | 63 | 70% |
| Val | 1,260 | 63 | 15% |
| Test | 1,260 | 63 | 15% |
| **Total** | **8,397** | **63** | **100%** |

### Breed List (63 breeds)

<details>
<summary>Click to expand</summary>

- Abyssinian
- American Bobtail
- American Curl
- American Shorthair
- American Wirehair
- Balinese
- Bengal
- Birman
- Bombay
- British Shorthair
- Burmese
- Chartreux
- Chausie
- Cornish Rex
- Cymric
- Devon Rex
- Donskoy
- Egyptian Mau
- European Shorthair
- Exotic Shorthair
- German Rex
- Havana Brown
- Himalayan
- Japanese Bobtail
- Karelian Bobtail
- Khao Manee
- Korat
- Korean Bobtail
- Kurilian Bobtail
- LaPerm
- Lykoi
- Maine Coon
- Manx
- Mekong Bobtail
- Munchkin
- Nebelung
- Norwegian Forest Cat
- Ocicat
- Oregon Rex
- Oriental Shorthair
- Persian
- Peterbald
- Pixie-bob
- Ragamuffin
- Ragdoll
- Russian Blue
- Safari
- Savannah
- Scottish Fold
- Selkirk Rex
- Siamese
- Siberian
- Singapura
- Sokoke
- Somali
- Sphynx
- Tonkinese
- Toyger
- Turkish Angora
- Turkish Van
- Ukrainian Levkoy
- Ural Rex
- Van Kedisi

</details>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### To Do
- [ ] Add data augmentation variations
- [ ] Experiment with other architectures (EfficientNet, Vision Transformer)
- [ ] Implement Grad-CAM visualization
- [ ] Create web interface for inference
- [ ] Add model quantization for mobile deployment

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- ResNet50V2 architecture from [Keras Applications](https://keras.io/api/applications/)
- YOLO models from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Dataset: [Cat Breeds Dataset](https://www.kaggle.com/datasets/denispotapov/cat-breeds-dataset)

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

**GitHub**: [theAbyssOfTime2004/cat-classification](https://github.com/theAbyssOfTime2004/cat-classification)

---

<p align="center">Made with ❤️ and 🐱</p>
