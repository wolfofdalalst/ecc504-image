import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import matplotlib.pyplot as plt


IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def _iter_class_dirs(base_dir: Path) -> Iterable[Path]:
    for p in sorted(base_dir.iterdir()):
        if p.is_dir():
            yield p


def _iter_images(cls_dir: Path, exts: Tuple[str, ...] = IMG_EXTS) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(cls_dir.glob(f"*{ext}")))
    return files


def count_images(base_dir: str | Path) -> Dict[str, int]:
    """Count images per class directory under base_dir.

    Supports common image extensions.
    """
    base = Path(base_dir)
    counts: Dict[str, int] = {}
    for cls_dir in _iter_class_dirs(base):
        counts[cls_dir.name] = len(_iter_images(cls_dir))
    return counts


def show_random_images(base_dir: str | Path, num_images: int = 5, seed: int | None = None):
    """Show a random selection of images across all classes."""
    if seed is not None:
        random.seed(seed)

    base = Path(base_dir)
    all_images: List[Path] = []
    for cls_dir in _iter_class_dirs(base):
        all_images.extend(_iter_images(cls_dir))

    if not all_images:
        print("No images found.")
        return

    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(selected_images):
        img = plt.imread(img_path)
        plt.subplot(1, len(selected_images), i + 1)
        plt.imshow(img)
        plt.title(img_path.parent.name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_random_images_per_class(base_dir: str | Path, seed: int | None = None):
    """Display one random image from each class directory."""
    if seed is not None:
        random.seed(seed)

    base = Path(base_dir)
    classes = list(_iter_class_dirs(base))
    num_classes = len(classes)

    if num_classes == 0:
        print("No class folders found.")
        return

    plt.figure(figsize=(3 * num_classes, 3))
    for i, cls_dir in enumerate(classes):
        images = _iter_images(cls_dir)
        if images:
            random_image = random.choice(images)
            img = plt.imread(random_image)
            plt.subplot(1, num_classes, i + 1)
            plt.imshow(img)
            plt.title(cls_dir.name)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def make_subset(src_dir: str | Path, dst_dir: str | Path, per_class: int = 200, seed: int = 42) -> Tuple[int, int]:
    """Create a balanced subset with per_class images from each class.

    Returns: (total_copied, classes_created)
    """
    random.seed(seed)

    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    classes_created = 0

    for cls_dir in _iter_class_dirs(src_path):
        cls_dst_dir = dst_path / cls_dir.name
        cls_dst_dir.mkdir(exist_ok=True)

        images = _iter_images(cls_dir)
        if not images:
            print(f"Skipping {cls_dir.name}: no images found")
            continue

        num_to_copy = min(per_class, len(images))
        selected_images = random