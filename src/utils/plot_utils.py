import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable


def _first_available(history_dict: dict, keys: Iterable[str]):
    for k in keys:
        if k in history_dict:
            return history_dict[k]
    raise KeyError(f"None of the keys {list(keys)} were found in training history. Available: {list(history_dict.keys())}")


def plot_training_history(history, save_path: str | Path | None = None):
    """Plot training and validation accuracy and loss.

    Handles both 'accuracy' and 'sparse_categorical_accuracy' metric names.
    """
    hist = history.history

    # Metrics: support both dense and sparse names
    acc = _first_available(hist, ("accuracy", "sparse_categorical_accuracy"))
    val_acc = _first_available(hist, ("val_accuracy", "val_sparse_categorical_accuracy"))
    loss = _first_available(hist, ("loss",))
    val_loss = _first_available(hist, ("val_loss",))

    epochs = range(len(acc))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
