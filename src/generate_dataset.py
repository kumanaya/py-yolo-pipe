"""
Generate dataset: auto-label all images with CV pipeline and split train/val.
"""

import cv2
import numpy as np
from pathlib import Path
import shutil


def generate_labels(img_path, erosion_iters=3, min_area=30, max_area=5000):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=erosion_iters)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)

    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        boxes.append((x_center, y_center, nw, nh))

    return boxes, len(boxes)


def main():
    for d in ["dataset/images/train", "dataset/images/val", "dataset/labels/train", "dataset/labels/val"]:
        p = Path(d)
        if p.exists():
            for f in p.glob("*"):
                f.unlink()

    images = {
        "train": ["input/1.png", "input/2.png", "input/3.png"],
        "val": ["input/4.png"],
    }

    for split, img_list in images.items():
        img_dir = Path(f"dataset/images/{split}")
        lbl_dir = Path(f"dataset/labels/{split}")
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for src_path in img_list:
            src = Path(src_path)
            name = src.stem

            shutil.copy(str(src), str(img_dir / src.name))
            boxes, count = generate_labels(src, erosion_iters=3, min_area=30, max_area=5000)
            print(f"{src.name}: {count} pipes")

            lbl_path = lbl_dir / f"{name}.txt"
            with open(lbl_path, "w") as f:
                for xc, yc, nw, nh in boxes:
                    f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    train_count = len(list(Path("dataset/images/train").glob("*")))
    val_count = len(list(Path("dataset/images/val").glob("*")))
    print(f"\nDataset: {train_count} train, {val_count} val")


if __name__ == "__main__":
    main()
