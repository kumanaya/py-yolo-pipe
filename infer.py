from ultralytics import YOLO
import sys

model = YOLO('models/best_v5.pt')
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.27

results = model(sys.argv[1], conf=conf)[0]
n = len(results.boxes) if results.boxes else 0
print(f'Pipes detected: {n}')
