import tensorflow as tf
from pathlib import Path

from config import Config
from data_loading import load_datasets, get_augmentation
from model_builder import build_model
from utils.plot_utils import plot_training_history


def main():
    # Reproducibility
    tf.keras.utils.set_random_seed(Config.SEED)

    # Load datasets
    train_ds, val_ds = load_datasets(Config.SUBSET_PATH)

    # Sanity checks: image size and classes
    ds_h, ds_w = train_ds.element_spec[0].shape[1], train_ds.element_spec[0].shape[2]
    if (ds_h, ds_w) != Config.IMG_SIZE:
        raise ValueError(
            f"Dataset image size {(ds_h, ds_w)} does not match Config.IMG_SIZE {Config.IMG_SIZE}."
        )

    class_names = getattr(train_ds, "class_names", None) or []
    inferred_num_classes = len(class_names) if class_names else Config.NUM_CLASSES
    if inferred_num_classes != Config.NUM_CLASSES:
        print(
            f"Warning: Config.NUM_CLASSES={Config.NUM_CLASSES} differs from dataset classes={inferred_num_classes}. Using {inferred_num_classes}."
        )

    if class_names:
        print(f"Classes ({len(class_names)}): {class_names}")

    # Apply data augmentation
    aug_layer = get_augmentation()
    train_ds = train_ds.map(
        lambda x, y: (aug_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prefetch after augmentation
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_model(
        input_shape=Config.IMG_SIZE + (3,), num_classes=inferred_num_classes
    )
    model.summary()

    # Callbacks
    Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(Config.MODEL_DIR) / "cnn_scratch_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    # Train
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=Config.EPOCHS, callbacks=callbacks
    )

    # Save model and plot
    model.save(checkpoint_path)
    plot_training_history(
        history, save_path=Path(Config.PLOTS_DIR) / "cnn_scratch_history.png"
    )


if __name__ == "__main__":
    main()
