import os, random, shutil
from pathlib import Path
import matplotlib.pyplot as plt


def count_images(base_dir):
    counts = {}
    for cls in os.listdir(base_dir):
        cls_dir = Path(base_dir) / cls
        if cls_dir.is_dir():
            counts[cls] = len(list(cls_dir.glob("*.jpg")))
    return counts


def show_random_images(base_dir, num_images=5):
    all_images = []
    for cls in os.listdir(base_dir):
        cls_dir = Path(base_dir) / cls
        if cls_dir.is_dir():
            images = list(cls_dir.glob("*.jpg"))
            all_images.extend(images)

    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(selected_images):
        img = plt.imread(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(img_path.parent.name)
        plt.axis("off")
    plt.show()


def show_random_images_per_class(base_dir):
    """Display one random image from each class"""
    classes = [cls for cls in os.listdir(base_dir) if (Path(base_dir) / cls).is_dir()]
    num_classes = len(classes)

    plt.figure(figsize=(15, 3))
    for i, cls in enumerate(classes):
        cls_dir = Path(base_dir) / cls
        images = list(cls_dir.glob("*.jpg"))
        if images:
            random_image = random.choice(images)
            img = plt.imread(random_image)
            plt.subplot(1, num_classes, i + 1)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def make_subset(src_dir, dst_dir, per_class=200, seed=42):
    """Create a balanced subset with per_class images from each class"""
    random.seed(seed)

    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    classes_created = 0

    for cls in os.listdir(src_dir):
        cls_src_dir = src_path / cls
        if cls_src_dir.is_dir():
            cls_dst_dir = dst_path / cls
            cls_dst_dir.mkdir(exist_ok=True)

            # Get all images in the class directory
            images = list(cls_src_dir.glob("*.jpg"))

            # Randomly select per_class images
            num_to_copy = min(per_class, len(images))
            selected_images = random.sample(images, num_to_copy)

            # Copy selected images
            for img_path in selected_images:
                dst_img_path = cls_dst_dir / img_path.name
                shutil.copy2(img_path, dst_img_path)

            total_copied += num_to_copy
            classes_created += 1
            print(f"Created {cls} subset with {num_to_copy} images")

    print(f"\nSubset creation complete!")
    print(f"Total classes: {classes_created}")
    print(f"Total images copied: {total_copied}")

    return total_copied, classes_created


if __name__ == "__main__":
    src = "./data/flower_photos"
    dst = "./subsets/subset_200"

    print("Counting images...")
    print(count_images(src))

    print("\nShowing one image per class:")
    show_random_images_per_class(src)

    print("\nCreating subset...")
    make_subset(src, dst)
