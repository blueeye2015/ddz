#!/usr/bin/env python3
"""用 yolov8l best.pt 识别 images/ 目录下的扑克牌图片"""

import cv2
import os
import glob
from ultralytics import YOLO

ROBOFLOW_NAMES = [
    '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S',
    '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S',
    '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S',
    '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S',
    '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS',
    'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS',
    'QC', 'QD', 'QH', 'QS'
]
SUIT_MAP = {'C': '♣', 'D': '♦', 'H': '♥', 'S': '♠'}

def roboflow_to_ddz(name):
    if len(name) == 2:
        num, suit = name[0], name[1]
    elif len(name) == 3:
        num, suit = name[:2], name[2]
    else:
        return name
    return f"{num}{SUIT_MAP.get(suit, suit)}"

def main():
    print("Loading yolov8l model (best.pt)...")
    model = YOLO('best.pt')
    
    img_dir = "images"
    if not os.path.exists(img_dir):
        print("images/ not found")
        return
    
    files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + 
                   glob.glob(os.path.join(img_dir, "*.png")))
    
    print(f"Found {len(files)} images, testing first 10...")
    
    os.makedirs("test_images_output", exist_ok=True)
    
    correct = 0
    total = 0
    
    for i, fpath in enumerate(files[:10]):
        img = cv2.imread(fpath)
        if img is None:
            continue
        
        results = model(img, conf=0.25, iou=0.45, verbose=False)
        
        fname = os.path.basename(fpath)
        print(f"\n[{i+1}] {fname} ({img.shape[1]}x{img.shape[0]})")
        
        detected = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[j].cpu().numpy())
                yc = float(r.boxes.conf[j].cpu().numpy())
                cls = int(r.boxes.cls[j].cpu().numpy())
                name = ROBOFLOW_NAMES[cls] if 0 <= cls < len(ROBOFLOW_NAMES) else f"cls_{cls}"
                detected.append({
                    'name': roboflow_to_ddz(name),
                    'raw': name,
                    'conf': yc,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
        
        if not detected:
            print("  No detection")
        else:
            for d in detected:
                print(f"  -> {d['name']:4s} (conf={d['conf']:.2f})")
            total += len(detected)
        
        # Draw debug
        debug = img.copy()
        for d in detected:
            color = (0, 255, 0) if d['conf'] > 0.7 else (0, 165, 255)
            cv2.rectangle(debug, (d['x1'], d['y1']), (d['x2'], d['y2']), color, 2)
            label = f"{d['name']}:{d['conf']:.2f}"
            cv2.putText(debug, label, (d['x1'], max(d['y1']-5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        out_path = os.path.join("test_images_output", fname)
        cv2.imwrite(out_path, debug)
    
    print(f"\nDone. Debug images saved to test_images_output/")
    if total > 0:
        print(f"Total detections in first 10 images: {total}")

if __name__ == "__main__":
    main()
