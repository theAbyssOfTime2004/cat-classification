"""
Configuration file for Cat Classification Project
Centralized configuration for paths, hyperparameters, and settings
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw-data" / "cat-breeds"
CLEANED_DATA_DIR = DATA_DIR / "cleaned-data" / "cat-breeds-cleaned"
NON_CAT_DATA_DIR = DATA_DIR / "cleaned-data" / "non-cat-images"
PROBLEMATIC_DATA_DIR = DATA_DIR / "cleaned-data" / "problematic-files"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Processed data splits
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Model directories
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "cat_breed_classifier_best.keras"
YOLO_MODEL_PATH = MODELS_DIR / "yolo11x.pt"

# Notebook directory
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Dataset statistics
DATASET_STATS_CSV = RAW_DATA_DIR / "dataset_stats.csv"
SPLIT_INDICES_PATH = PROCESSED_DATA_DIR / "split_indices.json"

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Data cleaning
MIN_IMAGES_PER_BREED = 2
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

# YOLO-based content filtering
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Increased from 0.3 for better quality
CAT_CLASS_ID_COCO = 15  # Cat class ID in COCO dataset

# Data splitting ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Image specifications
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Model architecture
BASE_MODEL_NAME = "ResNet50V2"  # Options: ResNet50V2, EfficientNetB0, etc.
USE_GLOBAL_AVERAGE_POOLING = True  # True for GAP, False for Flatten
DENSE_UNITS = 512
DROPOUT_RATE = 0.5

# Training parameters
BATCH_SIZE = 32
EPOCHS_STAGE1 = 50  # Feature extraction stage
EPOCHS_STAGE2 = 30  # Fine-tuning stage
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5  # Lower for fine-tuning

# Fine-tuning configuration
UNFREEZE_LAYERS = 50  # Number of layers to unfreeze from the end

# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================

# Training augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# ============================================================================
# CALLBACKS CONFIGURATION
# ============================================================================

# ModelCheckpoint
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'
CHECKPOINT_SAVE_BEST_ONLY = True

# EarlyStopping
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_RESTORE_BEST = True

# ReduceLROnPlateau
REDUCE_LR_MONITOR = 'val_loss'
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_PATIENCE = 5
REDUCE_LR_MIN_LR = 1e-7

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Plot settings
PLOT_FIGSIZE = (12, 8)
PLOT_DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Sample visualization
NUM_SAMPLE_IMAGES = 9
NUM_AUGMENTATIONS_TO_SHOW = 5

# ============================================================================
# HARDWARE & PERFORMANCE
# ============================================================================

# GPU settings
USE_GPU = True
MIXED_PRECISION = True  # Enable mixed precision training for faster training
GPU_MEMORY_GROWTH = True

# Multiprocessing
USE_MULTIPROCESSING = True
WORKERS = 4

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================

VERBOSE = 1
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Create all necessary directories if they don't exist"""
    dirs_to_create = [
        DATA_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR, NON_CAT_DATA_DIR,
        PROBLEMATIC_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DIR, VAL_DIR,
        TEST_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR
    ]

    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)

    print("✓ All directories created successfully")


def get_num_classes():
    """Get the number of classes (cat breeds) from training directory"""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    num_classes = len([d for d in TRAIN_DIR.iterdir() if d.is_dir()])

    if num_classes == 0:
        raise ValueError(f"No class subdirectories found in {TRAIN_DIR}")

    return num_classes


def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("CAT CLASSIFICATION PROJECT CONFIGURATION")
    print("=" * 80)
    print(f"\nPATHS:")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Models Directory: {MODELS_DIR}")

    print(f"\nDATA PARAMETERS:")
    print(f"  Train/Val/Test Split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  YOLO Confidence Threshold: {YOLO_CONFIDENCE_THRESHOLD}")
    print(f"  Min Images per Breed: {MIN_IMAGES_PER_BREED}")

    print(f"\nMODEL PARAMETERS:")
    print(f"  Base Model: {BASE_MODEL_NAME}")
    print(f"  Image Size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Pooling: {'GlobalAveragePooling2D' if USE_GLOBAL_AVERAGE_POOLING else 'Flatten'}")
    print(f"  Dense Units: {DENSE_UNITS}")
    print(f"  Dropout Rate: {DROPOUT_RATE}")

    print(f"\nTRAINING PARAMETERS:")
    print(f"  Stage 1 Epochs: {EPOCHS_STAGE1} (LR: {LEARNING_RATE_STAGE1})")
    print(f"  Stage 2 Epochs: {EPOCHS_STAGE2} (LR: {LEARNING_RATE_STAGE2})")
    print(f"  Unfreeze Layers: {UNFREEZE_LAYERS}")

    print(f"\nHARDWARE:")
    print(f"  Use GPU: {USE_GPU}")
    print(f"  Mixed Precision: {MIXED_PRECISION}")
    print(f"  Workers: {WORKERS}")
    print("=" * 80)


if __name__ == "__main__":
    # Test configuration
    create_directories()
    print_config()

    try:
        num_classes = get_num_classes()
        print(f"\n✓ Found {num_classes} cat breeds in training data")
    except Exception as e:
        print(f"\n⚠ Could not determine number of classes: {e}")
