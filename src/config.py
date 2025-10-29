# src/config.py

from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "flower_photos"
SUBSETS_DIR = BASE_DIR / "subsets"

CURRENT_SUBSET = SUBSETS_DIR / "subset_200"

# Dataset Parameters
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.2

# Model Training Parameters (for later parts)
EPOCHS = 25
LEARNING_RATE = 1e-3


# Utility
def print_config():
    print("Configuration:")
    print(f"Base Dir: {BASE_DIR}")
    print(f"Current Subset: {CURRENT_SUBSET}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Seed: {SEED}")
    print(f"Validation Split: {VALIDATION_SPLIT}")
