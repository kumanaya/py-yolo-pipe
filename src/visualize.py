"""
Visualization and analysis tools for YOLO Pipe Counter.

Provides:
    - Training metrics plots (loss, mAP, precision, recall)
    - Confusion matrix visualization
    - Prediction visualization with class distribution charts
    - Dataset exploration

Usage:
    python src/visualize.py --metrics models/pipe_detector/
    python src/visualize.py --results models/pipe_detector/val_batch0_pred.jpg
    python src/visualize.py --dataset-stats
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics.utils.plotting import plt_settings

from utils import load_config, load_data_config, get_class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization tools for Pipe Counter")
    parser.add_argument("--metrics", type=str, help="Path to training results folder")
    parser.add_argument("--dataset-stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    return parser.parse_args()


def plot_training_metrics(results_dir: str):
    """Plot training metrics from results.csv."""
    results_dir = Path(results_dir)
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        print(f"Results CSV not found: {csv_path}")
        return

    import csv
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    if not data:
        print("No data in results CSV")
        return

    epochs = [int(r.get("epoch", i)) for i, r in enumerate(data)]

    metrics_map = {
        "train/box_loss": ("Box Loss", "red"),
        "train/cls_loss": ("Class Loss", "blue"),
        "train/dfl_loss": ("DFL Loss", "green"),
        "metrics/precision(B)": ("Precision", "purple"),
        "metrics/recall(B)": ("Recall", "orange"),
        "metrics/mAP50(B)": ("mAP@50", "brown"),
        "metrics/mAP50-95(B)": ("mAP@50:95", "pink"),
    }

    n_metrics = len(metrics_map)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, (key, (label, color)) in zip(axes, metrics_map.items()):
        values = [float(r.get(key, 0)) for r in data]
        ax.plot(epochs, values, color=color, linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "training_metrics.png", dpi=150)
    plt.close()
    print(f"Training metrics saved to: {results_dir / 'training_metrics.png'}")


def plot_class_distribution(data_config_path: str):
    """Plot class distribution from training labels."""
    data_config = load_data_config(data_config_path)
    class_names = get_class_names(data_config)
    dataset_dir = Path("dataset")

    counts = {i: 0 for i in range(len(class_names))}
    for split in ["train", "val"]:
        lbl_dir = dataset_dir / "labels" / split
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r") as f:
                for line in f:
                    if line.strip():
                        cls_id = int(line.strip().split()[0])
                        if cls_id in counts:
                            counts[cls_id] += 1

    labels = [class_names[i] for i in sorted(counts)]
    values = [counts[i] for i in sorted(counts)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
    ax.set_title("Class Distribution in Dataset", fontsize=14, fontweight="bold")
    ax.set_ylabel("Instance Count")
    ax.set_xlabel("Pipe Class")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("output/class_distribution.png", dpi=150)
    plt.close()
    print(f"Class distribution saved to: output/class_distribution.png")


def main():
    args = parse_args()

    if args.metrics:
        plot_training_metrics(args.metrics)

    if args.dataset_stars or args.dataset_stats:
        plot_class_distribution(args.data)


if __name__ == "__main__":
    main()
