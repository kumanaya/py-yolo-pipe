from pathlib import Path
import shutil
from ultralytics import YOLO

def main():
    for d in ['dataset/images/train', 'dataset/images/val', 'dataset/labels/train', 'dataset/labels/val']:
        p = Path(d)
        if p.exists():
            for f in p.glob('*'):
                f.unlink()
        p.mkdir(parents=True, exist_ok=True)

    print("Generating labels from v8 (precision 0.970)...")

    v8 = YOLO('models/best_v8.pt')

    config = [
        ('input/1.png', 79,  'train'),
        ('input/2.png', 10,  'train'),
        ('input/3.png', 36,  'train'),
        ('input/4.png', 89,  'train'),
        ('input/5.png', 147, 'train'),
    ]

    # Also copy img4 to val (same image, just for monitoring)
    val_src = Path('input/4.png')
    shutil.copy(str(val_src), 'dataset/images/val/4.png')

    for path, target, split in config:
        src = Path(path)
        name = src.stem
        img_dir = Path(f'dataset/images/{split}')
        lbl_dir = Path(f'dataset/labels/{split}')

        r = v8(path, conf=0.01)[0]
        confs = r.boxes.conf.tolist() if r.boxes else []
        xywhn = r.boxes.xywhn.tolist() if r.boxes else []
        pairs = sorted(zip(confs, xywhn), key=lambda x: -x[0])[:target]

        print(f'{src.name}: top-{len(pairs)} (min_conf={pairs[-1][0]:.4f}) -> {split}')

        shutil.copy(str(src), str(img_dir / src.name))
        with open(lbl_dir / f'{name}.txt', 'w') as f:
            for conf, box in pairs:
                f.write(f'0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n')

    # Copy same label to val
    shutil.copy('dataset/labels/train/4.txt', 'dataset/labels/val/4.txt')

    model = YOLO('yolo26s.pt')
    print("\nTraining v9: all 5 images + mosaic + heavy aug...")
    model.train(
        data='dataset/data.yaml',
        epochs=300,
        imgsz=640,
        batch=8,
        device=0,
        project='runs/detect/models',
        name='pipe_detector_v9',
        exist_ok=True,
        patience=80,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        workers=0,
        close_mosaic=30,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.15,
        scale=0.7,
        fliplr=0.5,
        flipud=0.5,
    )
    print("Done!")

if __name__ == '__main__':
    main()
