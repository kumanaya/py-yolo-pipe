"""
Dataset preparation and augmentation for YOLO Pipe Counter.

Usage:
    python src/prepare_dataset.py --input path/to/images --labels path/to/labels
    python src/prepare_dataset.py --auto-split  # Split dataset into train/val

Features:
    - Automatic train/val split
    - Data augmentation (flip, rotate, brightness, blur, scaling)
    - Label format validation (YOLO format)
    - Dataset statistics
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import load_config, create_directories


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO Pipe Counter")
    parser.add_argument("--input", type=str, help="Directory containing source images")
    parser.add_argument("--labels", type=str, help="Directory containing source labels (YOLO format)")
    parser.add_argument("--auto-split", action="store_true", help="Auto-split existing dataset/train into train/val")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to config file")
    return parser.parse_args()


def validate_yolo_label(label_path: Path, img_width: int, img_height: int) -> bool:
    """Validate a YOLO format label file."""
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            if not (0 <= cls <= 100):
                return False
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                return False
        return True
    except Exception:
        return False


def split_dataset(config: dict):
    """Split images and labels into train/val sets."""
    data_cfg = config["dataset"]
    seed = data_cfg["seed"]
    split = data_cfg["train_split"]
    random.seed(seed)

    dataset_dir = Path("dataset")
    train_img_dir = dataset_dir / "images" / "train"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    val_lbl_dir = dataset_dir / "labels" / "val"

    source_img_dir = train_img_dir
    source_lbl_dir = train_lbl_dir

    images = sorted(source_img_dir.glob("*.*"))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [img for img in images if img.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {source_img_dir}")
        return

    random.shuffle(images)
    split_idx = int(len(images) * split)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    create_directories([val_img_dir, val_lbl_dir])

    for img in tqdm(train_images, desc="Processing train images"):
        pass

    for img in tqdm(val_images, desc="Moving validation images"):
        label = source_lbl_dir / img.with_suffix(".txt").name
        shutil.move(str(img), str(val_img_dir / img.name))
        if label.exists():
            shutil.move(str(label), str(val_lbl_dir / label.name))

    print(f"Dataset split: {len(train_images)} train, {len(val_images)} validation")


def apply_augmentations(image: np.ndarray, labels: list, config: dict) -> list:
    """
    Apply data augmentations to an image and its labels.
    Returns list of (augmented_image, augmented_labels) tuples.
    """
    aug_cfg = config["augmentation"]
    copies = aug_cfg["copies"]
    h, w = image.shape[:2]
    results = []

    for _ in range(copies):
        aug_img = image.copy()
        aug_labels = [list(l) for l in labels]

        if random.random() < aug_cfg["horizontal_flip"]:
            aug_img = cv2.flip(aug_img, 1)
            for lbl in aug_labels:
                lbl[1] = 1.0 - lbl[1]

        if random.random() < aug_cfg["vertical_flip"]:
            aug_img = cv2.flip(aug_img, 0)
            for lbl in aug_labels:
                lbl[2] = 1.0 - lbl[2]

        angle = random.uniform(-aug_cfg["rotate_limit"], aug_cfg["rotate_limit"])
        if abs(angle) > 1:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += new_w / 2 - w / 2
            M[1, 2] += new_h / 2 - h / 2
            aug_img = cv2.warpAffine(aug_img, M, (new_w, new_h))
            h, w = new_h, new_w

        if random.random() < aug_cfg["brightness_limit"] / 0.5:
            value = random.uniform(1 - aug_cfg["brightness_limit"], 1 + aug_cfg["brightness_limit"])
            aug_img = cv2.convertScaleAbs(aug_img, alpha=value, beta=0)

        if random.random() < aug_cfg["contrast_limit"] / 0.5:
            value = random.uniform(1 - aug_cfg["contrast_limit"], 1 + aug_cfg["contrast_limit"])
            aug_img = cv2.convertScaleAbs(aug_img, alpha=value, beta=0)

        ksize = random.randrange(1, aug_cfg["blur_limit"] * 2 + 1, 2)
        if random.random() < 0.3:
            aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)

        results.append((aug_img, aug_labels))

    return results


def augment_dataset(config: dict):
    """Apply augmentations to all training images."""
    dataset_dir = Path("dataset")
    train_img_dir = dataset_dir / "images" / "train"
    train_lbl_dir = dataset_dir / "labels" / "train"

    images = sorted(train_img_dir.glob("*.*"))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [img for img in images if img.suffix.lower() in image_extensions]

    if not images:
        print(f"No training images found in {train_img_dir}")
        return

    for img_path in tqdm(images, desc="Augmenting dataset"):
        label_path = train_lbl_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        with open(label_path, "r") as f:
            raw_labels = [list(map(float, line.strip().split())) for line in f if line.strip()]

        augmented = apply_augmentations(image, raw_labels, config)
        for i, (aug_img, aug_labels) in enumerate(augmented):
            stem = img_path.stem
            aug_img_path = train_img_dir / f"{stem}_aug_{i}{img_path.suffix}"
            aug_lbl_path = train_lbl_dir / f"{stem}_aug_{i}.txt"
            cv2.imwrite(str(aug_img_path), aug_img)
            with open(aug_lbl_path, "w") as f:
                for lbl in aug_labels:
                    f.write(" ".join(map(str, lbl)) + "\n")

    print("Augmentation complete!")


def compute_dataset_stats(config: dict):
    """Print dataset statistics."""
    dataset_dir = Path("dataset")
    for split_name in ["train", "val"]:
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        images = list(img_dir.glob("*.*"))
        labels = list(lbl_dir.glob("*.txt"))
        print(f"[{split_name}] Images: {len(images)}, Labels: {len(labels)}")

    class_counts = {}
    for split_name in ["train", "val"]:
        lbl_dir = dataset_dir / "labels" / split_name
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    if line.strip():
                        cls = int(line.strip().split()[0])
                        class_counts[cls] = class_counts.get(cls, 0) + 1
    data_cfg = load_config()["dataset"]
    print("\nClass distribution:")
    for cls_id, count in sorted(class_counts.items()):
        name = data_cfg["classes"][cls_id] if cls_id < len(data_cfg["classes"]) else "unknown"
        print(f"  [{cls_id}] {name}: {count} instances")


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.auto_split:
        split_dataset(config)
    elif args.augment:
        augment_dataset(config)
    elif args.input:
        print(f"Processing input: {args.input}")
        print("Use --auto-split to split dataset or --augment to augment")
    else:
        compute_dataset_stats(config)


if __name__ == "__main__":
    main()
