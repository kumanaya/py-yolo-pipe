"""
Inference and pipe counting script.

Detects, classifies, and counts pipes in images and video streams.

Usage:
    python src/inference.py --source input/image.jpg
    python src/inference.py --source input/video.mp4 --save
    python src/inference.py --source 0  # Webcam
    python src/inference.py --source input/ --stats-only  # Count only

Output:
    - Annotated images/videos with bounding boxes and counts
    - Console summary with per-class pipe counts
    - Optional CSV export of results
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils import load_config, load_data_config, get_class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipe detection and counting")
    parser.add_argument("--source", type=str, default="input", help="Source: image path, video path, directory, or camera ID")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained model weights")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="IoU threshold for NMS (YOLO26 is NMS-free by default)")
    parser.add_argument("--save", action="store_true", help="Save output images/videos")
    parser.add_argument("--show", action="store_true", help="Display results in a window")
    parser.add_argument("--csv", type=str, help="Export results to CSV file")
    parser.add_argument("--stats-only", action="store_true", help="Print stats only, no visualization")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Config file")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Data config file")
    return parser.parse_args()


def draw_pipe_counts(image: np.ndarray, class_counts: dict, class_names: dict, colors: dict):
    """Draw pipe count overlay on the image."""
    overlay = image.copy()
    h, w = image.shape[:2]

    x, y = 15, 30
    line_h = 30
    box_w = 280
    box_h = len(class_counts) * line_h + 50

    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    cv2.putText(image, "Pipe Count Summary", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += line_h + 5

    total = sum(class_counts.values())
    for cls_id, count in sorted(class_counts.items()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        color = colors.get(cls_id, (0, 255, 0))
        cv2.putText(image, f"{name}: {count}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += line_h

    cv2.putText(image, f"Total: {total}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image


def get_color_palette(num_classes: int) -> dict:
    """Generate distinct colors per class."""
    colors = [
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (0, 255, 255),  # yellow
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
    ]
    return {i: colors[i % len(colors)] for i in range(num_classes)}


def process_image(model: YOLO, image_path: Path, class_names: dict,
                  class_colors: dict, conf: float, iou: float,
                  output_dir: Path, show: bool, save: bool) -> dict:
    """Process a single image and return class counts."""
    results = model(str(image_path), conf=conf, iou=iou)[0]

    class_counts = defaultdict(int)
    if results.boxes is not None:
        for cls_id in results.boxes.cls:
            class_counts[int(cls_id)] += 1

    if not show and not save:
        return dict(class_counts)

    annotated = results.plot()
    annotated = draw_pipe_counts(annotated, class_counts, class_names, class_colors)

    if show:
        cv2.imshow("Pipe Detection", annotated)
        cv2.waitKey(0)

    if save:
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), annotated)

    return dict(class_counts)


def process_video(model: YOLO, video_path: Path, class_names: dict,
                  class_colors: dict, conf: float, iou: float,
                  output_dir: Path, show: bool, save: bool) -> list:
    """Process a video file and return per-frame counts."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save:
        out_path = output_dir / video_path.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_counts = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, iou=iou)[0]
        class_counts = defaultdict(int)
        if results.boxes is not None:
            for cls_id in results.boxes.cls:
                class_counts[int(cls_id)] += 1
        frame_counts.append(dict(class_counts))

        annotated = results.plot()
        annotated = draw_pipe_counts(annotated, class_counts, class_names, class_colors)

        cv2.putText(annotated, f"Frame: {frame_idx}/{total_frames}", (15, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if show:
            cv2.imshow("Pipe Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    return frame_counts


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = load_data_config(args.data)

    class_names = get_class_names(data_config)
    class_colors = get_color_palette(len(class_names))
    inf_cfg = config["inference"]
    conf = args.conf or inf_cfg["conf"]
    iou = args.iou or inf_cfg["iou"]

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train a model first: python src/train.py")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    source = Path(args.source)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    print(f"\n{'='*50}")
    print("Pipe Detection & Counting Results")
    print(f"{'='*50}")

    if source.is_file():
        suffix = source.suffix.lower()
        if suffix in image_extensions:
            counts = process_image(model, source, class_names, class_colors,
                                   conf, iou, output_dir, args.show and not args.stats_only,
                                   args.save and not args.stats_only)
            print(f"\nFile: {source.name}")
            for cls_id, count in sorted(counts.items()):
                print(f"  {class_names.get(cls_id, 'unknown')}: {count}")
            print(f"  TOTAL: {sum(counts.values())}")

        elif suffix in video_extensions:
            counts = process_video(model, source, class_names, class_colors,
                                   conf, iou, output_dir,
                                   args.show and not args.stats_only,
                                   args.save and not args.stats_only)
            totals = defaultdict(int)
            for frame_count in counts:
                for cls_id, c in frame_count.items():
                    totals[cls_id] += c
            avg_frame = len(counts)
            print(f"\nVideo: {source.name} ({len(counts)} frames)")
            for cls_id, total in sorted(totals.items()):
                avg = total / avg_frame if avg_frame > 0 else 0
                print(f"  {class_names.get(cls_id, 'unknown')}: {total} total, {avg:.1f}/frame")
            print(f"  TOTAL: {sum(totals.values())} across {avg_frame} frames")

    elif source.is_dir():
        all_images = []
        for ext in image_extensions:
            all_images.extend(source.glob(f"*{ext}"))
        all_images.extend(source.glob("*.jpg"))
        all_images = sorted(set(all_images))

        if not all_images:
            print(f"No images found in {source}")
            return

        print(f"Processing {len(all_images)} images from {source}/")
        all_counts = defaultdict(int)
        for img_path in all_images:
            counts = process_image(model, img_path, class_names, class_colors,
                                   conf, iou, output_dir, False, args.save)
            for cls_id, c in counts.items():
                all_counts[cls_id] += c
            print(f"  {img_path.name}: {sum(counts.values())} pipes")

        print(f"\n{'='*50}")
        print("SUMMARY — All images")
        for cls_id, total in sorted(all_counts.items()):
            print(f"  {class_names.get(cls_id, 'unknown')}: {total}")
        print(f"  TOTAL: {sum(all_counts.values())}")

    elif source.is_dir() and args.stats_only:
        pass

    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["source"] + [class_names[i] for i in sorted(class_names)]
            writer.writerow(header)
            if source.is_file():
                counts = process_image(model, source, class_names, class_colors,
                                       conf, iou, output_dir, False, False)
                row = [source.name] + [counts.get(i, 0) for i in sorted(class_names)]
                writer.writerow(row)
            elif source.is_dir():
                for img_path in sorted(source.glob("*.*")):
                    if img_path.suffix.lower() not in image_extensions:
                        continue
                    counts = process_image(model, img_path, class_names, class_colors,
                                           conf, iou, output_dir, False, False)
                    row = [img_path.name] + [counts.get(i, 0) for i in sorted(class_names)]
                    writer.writerow(row)
        print(f"Results exported to: {csv_path}")


if __name__ == "__main__":
    main()
