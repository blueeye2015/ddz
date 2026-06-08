#!/usr/bin/env python3
"""滑动窗口分类：不用YOLO框，直接用手牌区整图从左到右滑动切分，每段独立分类"""

import cv2
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 加载配置
config_file = "ddz_config.json"
with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

elements = config['elements']
base_w = config['base_width']
base_h = config['base_height']

CLASS_NAMES = ['10','2','3','4','5','6','7','8','9','A','BJ','J','K','Q','SJ']
SUIT_CHARS = {'♠': '黑桃', '♥': '红心', '♦': '方块', '♣': '梅花'}

# 加载分类器
print("Loading classifier...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = models.mobilenet_v3_small(weights=None)
classifier.classifier[3] = nn.Linear(classifier.classifier[3].in_features, len(CLASS_NAMES))
classifier.load_state_dict(torch.load('card_classifier_best.pth', map_location=device))
classifier.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def classify_crop(crop_bgr):
    """对单张裁切图分类，返回 (name, conf)"""
    if crop_bgr.size == 0:
        return None, 0.0
    # 转RGB PIL
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = classifier(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], conf.item()


def sliding_window_recognize(hand_img, window_w=70, stride=15, min_conf=0.70):
    """滑动窗口识别手牌区整图"""
    h, w = hand_img.shape[:2]
    results = []
    
    # 只取上半部分（数字区域），避免底部花色/按钮干扰
    search_h = min(h, 130)
    search_img = hand_img[0:search_h, :]
    
    x = 0
    while x + window_w <= w:
        crop = search_img[:, x:x+window_w]
        name, conf = classify_crop(crop)
        if name and conf >= min_conf:
            results.append({
                'name': name,
                'conf': conf,
                'x': x + window_w // 2,  # 中心位置
                'x1': x,
                'x2': x + window_w
            })
        x += stride
    
    # 去重：中心距 < 25px 的合并，保留置信度最高的
    results.sort(key=lambda r: r['x'])
    deduped = []
    for r in results:
        dup = False
        for d in deduped:
            if abs(r['x'] - d['x']) < 25:
                dup = True
                if r['conf'] > d['conf']:
                    d.update(r)
                break
        if not dup:
            deduped.append(r)
    
    return deduped


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


def test_file(img_path):
    print(f"\n{'='*60}")
    print(f"测试: {img_path}")
    print('='*60)
    
    img = cv2.imread(img_path)
    if img is None:
        print("❌ 无法读取图片")
        return
    
    h, w = img.shape[:2]
    scale_x = w / base_w
    scale_y = h / base_h
    
    hand_img = extract_region(img, elements['my_hand'], scale_x, scale_y)
    if hand_img.size == 0:
        print("❌ 手牌区提取失败")
        return
    
    print(f"手牌区尺寸: {hand_img.shape[1]}x{hand_img.shape[0]}")
    
    # 滑动窗口识别
    cards = sliding_window_recognize(hand_img, window_w=70, stride=12, min_conf=0.60)
    
    print(f"识别到 {len(cards)} 张牌:")
    for c in cards:
        print(f"  {c['name']:4s} (conf={c['conf']:.2f}, center_x={c['x']})")
    
    # 画调试图
    debug = hand_img.copy()
    for c in cards:
        color = (0, 255, 0) if c['conf'] > 0.80 else (0, 165, 255) if c['conf'] > 0.70 else (0, 0, 255)
        cv2.rectangle(debug, (c['x1'], 0), (c['x2'], min(hand_img.shape[0], 130)), color, 2)
        cv2.putText(debug, f"{c['name']}:{c['conf']:.2f}", (c['x1'], 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    out_name = f"test_slide_{os.path.basename(img_path)}"
    cv2.imwrite(out_name, debug)
    print(f"调试图: {out_name}")


def main():
    raw_dir = "raw_screenshots"
    if os.path.exists(raw_dir):
        files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.png')])
        for f in files[-3:]:
            test_file(os.path.join(raw_dir, f))
    else:
        print("❌ raw_screenshots/ 不存在")


if __name__ == "__main__":
    main()
