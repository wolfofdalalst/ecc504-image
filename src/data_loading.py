import tensorflow as tf
from typing import Tuple, Optional

# Suppress TensorFlow logging for clarity (uncomment to use)
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_datasets(
    data_dir: str,
    img_size: Optional[Tuple[int, int]] = None,
    batch_size: Optional[int] = None,
    validation_split: Optional[float] = None,
    seed: Optional[int] = None,
):
    """Load dataset from directory with training and validation split.

    Returns: (train_ds, val_ds)
    """
    # Lazy import to avoid circulars and allow overrides via args
    try:
        from config import Config  # type: ignore
    except Exception:
        class Config:  # fallback
            IMG_SIZE = (224, 224)
            BATCH_SIZE = 32
            VALIDATION_SPLIT = 0.2
            SEED = 42

    img_size = img_size or Config.IMG_SIZE
    batch_size = batch_size or Config.BATCH_SIZE
    validation_split = validation_split or Config.VALIDATION_SPLIT
    seed = seed or Config.SEED

    # Use modern keras.utils API
    image_ds_from_dir = tf.keras.utils.image_dataset_from_directory

    train_ds = image_ds_from_dir(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = image_ds_from_dir(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    # Cache and shuffle here; leave prefetch to caller (after augmentation mapping)
    train_ds = train_ds.cache().shuffle(1000, seed=seed, reshuffle_each_iteration=True)
    val_ds = val_ds.cache()

    print(
        f"Dataset loaded. Train batches: {len(train_ds)}, Validation batches: {len(val_ds)}"
    )

    return train_ds, val_ds


def get_augmentation():
    """Return a tf.keras.Sequential data augmentation pipeline."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


# Backward-compatible alias
get_augmentation_pipeline = get_augmentation


def visualize_augmentation(train_ds, data_augmentation, num_examples: int = 5):
    """Visualize N augmented versions of a single image from dataset."""
    import matplotlib.pyplot as plt  # local import to keep training light
    import tensorflow as tf

    plt.figure(figsize=(12, 6))
    for images, _ in train_ds.take(1):
        sample_img = images[0]
        for i in range(num_examples):
            augmented_img = data_augmentation(tf.expand_dims(sample_img, 0))
            ax = plt.subplot(1, num_examples, i + 1)
            plt.imshow(augmented_img[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.suptitle("Data Augmentation Examples", fontsize=16)
    plt.show()


if __name__ == "__main__":
    from config import Config

    train_dataset, val_dataset = load_datasets(
        data_dir=Config.SUBSET_PATH,
        img_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        validation_split=Config.VALIDATION_SPLIT,
        seed=Config.SEED,
    )

    augmentation_pipeline = get_augmentation()
    visualize_augmentation(train_dataset, augmentation_pipeline, num_examples=5)
