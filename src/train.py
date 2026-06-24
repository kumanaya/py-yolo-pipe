"""
YOLO training script for pipe detection.

Trains a YOLO model to detect and classify pipe variations.

Usage:
    python src/train.py                          # Train with default config
    python src/train.py --model yolo26m          # Use medium model
    python src/train.py --epochs 150 --batch 8   # Custom params
    python src/train.py --resume                 # Resume from last checkpoint

The model learns to identify:
    - pipe_straight  : straight pipe segments
    - pipe_bent      : bent/elbow pipe sections
    - pipe_connector : T-joints and couplers
    - pipe_valve     : valves and regulators
    - pipe_flange    : flanges and connection plates
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from utils import load_config, load_data_config, create_directories


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for pipe detection")
    parser.add_argument("--model", type=str, help="YOLO model architecture (yolo26n, yolo26s, etc.)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Training image size")
    parser.add_argument("--device", type=str, help="Training device (cpu, 0, 0,1)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = load_data_config(args.data)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]

    create_directories(["models", "runs"])

    arch = args.model or model_cfg["arch"]
    epochs = args.epochs or model_cfg["epochs"]
    batch = args.batch or model_cfg["batch"]
    imgsz = args.imgsz or model_cfg["imgsz"]
    device = args.device or model_cfg["device"]

    print("=" * 60)
    print(f"YOLO Pipe Counter — Training")
    print("=" * 60)
    print(f"Model:       {arch}")
    print(f"Epochs:      {epochs}")
    print(f"Batch size:  {batch}")
    print(f"Image size:  {imgsz}")
    print(f"Device:      {device}")
    print(f"Classes:     {', '.join(dataset_cfg['classes'])}")
    print(f"Data config: {args.data}")
    print("=" * 60)

    if args.resume:
        last_pt = Path("models/last.pt")
        if not last_pt.exists():
            print(f"Checkpoint not found: {last_pt}")
            sys.exit(1)
        model = YOLO(str(last_pt))
        print(f"Resuming from {last_pt}")
    else:
        model = YOLO(f"{arch}.pt" if args.pretrained else f"{arch}.yaml")

    results = model.train(
        data=args.data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=model_cfg["patience"],
        lr0=model_cfg["lr0"],
        project="models",
        name="pipe_detector",
        exist_ok=True,
        pretrained=args.pretrained,
        seed=dataset_cfg["seed"],
        amp=True,
        mosaic=1.0,
        mixup=0.1,
    )

    best_path = Path("models/pipe_detector/weights/best.pt")
    if best_path.exists():
        import shutil
        shutil.copy(str(best_path), "models/best.pt")
        print(f"\nBest model saved to: models/best.pt")

    print("\nTraining complete!")
    print(f"Results saved in: models/pipe_detector/")
    print(f"Best weights:     models/best.pt")


if __name__ == "__main__":
    main()
