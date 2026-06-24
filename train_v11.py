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

    v1 = YOLO('models/best.pt')
    v8 = YOLO('models/best_v8.pt')

    config = [
        ('input/1.png', 79,  'train', v8),
        ('input/2.png', 10,  'train', v1),
        ('input/3.png', 36,  'train', v1),
        ('input/5.png', 147, 'train', v8),
        ('input/4.png', 89,  'val',   v1),
    ]

    for path, target, split, mdl in config:
        src = Path(path)
        name = src.stem
        img_dir = Path(f'dataset/images/{split}')
        lbl_dir = Path(f'dataset/labels/{split}')

        r = mdl(path, conf=0.01)[0]
        confs = r.boxes.conf.tolist() if r.boxes else []
        xywhn = r.boxes.xywhn.tolist() if r.boxes else []
        pairs = sorted(zip(confs, xywhn), key=lambda x: -x[0])[:target]

        src_name = mdl.model_name if hasattr(mdl, 'model_name') else 'v8' if mdl == v8 else 'v1'
        print(f'{src.name}: top-{len(pairs)} from {"v8" if mdl==v8 else "v1"} (min_conf={pairs[-1][0]:.4f}) -> {split}')

        shutil.copy(str(src), str(img_dir / src.name))
        with open(lbl_dir / f'{name}.txt', 'w') as f:
            for conf, box in pairs:
                f.write(f'0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n')

    model = YOLO('yolo26s.pt')
    print("\nTraining v11: v1 labels for img2/3/4 + v8 labels for img1/5...")
    model.train(
        data='dataset/data.yaml',
        epochs=200,
        imgsz=640,
        batch=8,
        device=0,
        project='runs/detect/models',
        name='pipe_detector_v11',
        exist_ok=True,
        patience=50,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        workers=0,
        close_mosaic=0,
    )
    print("Done!")

if __name__ == '__main__':
    main()
