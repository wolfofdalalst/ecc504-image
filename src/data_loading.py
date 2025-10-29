import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress TensorFlow logging for clarity
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_dataset(
    data_dir, img_size=(180, 180), batch_size=32, validation_split=0.2, seed=42
):
    """Load dataset from directory with training and validation split."""
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print(
        f"Dataset loaded. Train batches: {len(train_ds)}, Validation batches: {len(val_ds)}"
    )

    return train_ds, val_ds


def get_augmentation_pipeline():
    """Return a tf.keras.Sequential data augmentation pipeline."""
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    return data_augmentation


def visualize_augmentation(train_ds, data_augmentation, num_examples=5):
    """Visualize N augmented versions of a single image from dataset."""
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
    from config import CURRENT_SUBSET, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, SEED

    train_dataset, val_dataset = load_dataset(
        data_dir=CURRENT_SUBSET,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
    )

    augmentation_pipeline = get_augmentation_pipeline()
    visualize_augmentation(train_dataset, augmentation_pipeline, num_examples=5)
