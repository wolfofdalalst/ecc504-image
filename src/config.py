class Config:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 5

    # Data paths
    SUBSET_PATH = "subsets/subset_200"

    # Training config
    EPOCHS = 25
    VALIDATION_SPLIT = 0.2
    SEED = 42

    # Outputs
    MODEL_DIR = "outputs/models"
    PLOTS_DIR = "outputs/plots"
