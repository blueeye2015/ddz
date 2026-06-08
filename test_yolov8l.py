#!/usr/bin/env python3
"""测试 yolov8l 52类模型对斗地主截图的识别效果"""

import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

config_file = "ddz_config.json"
with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

elements = config['elements']
base_w = config['base_width']
base_h = config['base_height']

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

def extract_region(full_img, elem, scale_x, scale_y):
    h, w = full_img.shape[:2]
    x = int(elem['x'] * scale_x)
    y = int(elem['y'] * scale_y)
    ew = int(elem.get('w', 100) * scale_x)
    eh = int(elem.get('h', 100) * scale_y)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    ew = min(ew, w - x)
    eh = min(eh, h - y)
    if ew <= 0 or eh <= 0:
        return np.array([])
    return full_img[y:y+eh, x:x+ew]

def test_model(img_path, model):
    print(f"\n{'='*60}")
    print(f"Test: {img_path}")
    print('='*60)
    
    img = cv2.imread(img_path)
    if img is None:
        print("Cannot read image")
        return
    
    h, w = img.shape[:2]
    scale_x = w / base_w
    scale_y = h / base_h
    
    # 手牌区
    hand_img = extract_region(img, elements['my_hand'], scale_x, scale_y)
    if hand_img.size == 0:
        print("Hand region extraction failed")
        return
    
    print(f"Hand region: {hand_img.shape[1]}x{hand_img.shape[0]}")
    
    # yolov8l detect
    results = model(hand_img, conf=0.05, iou=0.10, imgsz=1280, verbose=False)
    
    cards = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
            yc = float(r.boxes.conf[i].cpu().numpy())
            cls = int(r.boxes.cls[i].cpu().numpy())
            name = ROBOFLOW_NAMES[cls] if 0 <= cls < len(ROBOFLOW_NAMES) else f"cls_{cls}"
            cards.append({
                'name': name,
                'ddz_name': roboflow_to_ddz(name),
                'conf': yc,
                'x': (x1 + x2) // 2,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })
    
    cards.sort(key=lambda c: c['x'])
    
    print(f"Detected {len(cards)} boxes:")
    for c in cards:
        print(f"  {c['ddz_name']:4s} (conf={c['conf']:.2f}, box=({c['x1']},{c['y1']},{c['x2']},{c['y2']}))")
    
    # 画调试图
    debug = hand_img.copy()
    for c in cards:
        color = (0, 255, 0) if c['conf'] > 0.5 else (0, 0, 255)
        cv2.rectangle(debug, (c['x1'], c['y1']), (c['x2'], c['y2']), color, 2)
        label = f"{c['ddz_name']}:{c['conf']:.2f}"
        cv2.putText(debug, label, (c['x1'], max(c['y1']-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    out_name = f"test_y8l_{os.path.basename(img_path)}"
    cv2.imwrite(out_name, debug)
    print(f"Debug saved: {out_name}")

def main():
    print("Loading yolov8l model (best.pt)...")
    model = YOLO('best.pt')
    
    raw_dir = "raw_screenshots"
    if os.path.exists(raw_dir):
        files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.png')])
        for f in files[-3:]:
            test_model(os.path.join(raw_dir, f), model)
    else:
        print("raw_screenshots/ not found")

if __name__ == "__main__":
    main()
