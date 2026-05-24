#!/usr/bin/env python3
"""
自动分类新增的 7
从 crops_raw/ 里找出还没分类的图，用模型预测，把 7 按颜色分类到 7_红色/ 或 7_黑色/
"""

import os
import cv2
import numpy as np
import shutil
import glob
from ddz_yolo import YOLORecognizer


def detect_color(img):
    """检测花色颜色"""
    h, w = img.shape[:2]
    if h < 80 or w < 30:
        return 'unknown'

    suit_y1 = min(h, 45)
    suit_y2 = min(h, 95)
    suit_x1 = min(w, 5)
    suit_x2 = max(suit_x1 + 1, w - 5)

    suit_region = img[suit_y1:suit_y2, suit_x1:suit_x2]
    if suit_region.size == 0:
        return 'unknown'

    hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 60, 50]), np.array([180, 255, 255]))
    red_pixels = cv2.countNonZero(red1) + cv2.countNonZero(red2)
    total = suit_region.shape[0] * suit_region.shape[1]

    return 'red' if red_pixels / total > 0.05 else 'black'


def is_classified(fname):
    """检查这张图是否已经被分类到某个文件夹"""
    for base in ['dataset/by_class', 'dataset/by_class_54']:
        if not os.path.exists(base):
            continue
        for cls in os.listdir(base):
            cls_dir = os.path.join(base, cls)
            if os.path.isdir(cls_dir) and os.path.exists(os.path.join(cls_dir, fname)):
                return True
    return False


def main():
    print("[加载] 15类数字识别模型...")
    rec = YOLORecognizer("ddz_yolo.pt", conf_threshold=0.30)

    # 确保目标文件夹存在
    os.makedirs("dataset/by_class_54/7_红色", exist_ok=True)
    os.makedirs("dataset/by_class_54/7_黑色", exist_ok=True)

    files = [f for f in os.listdir("dataset/crops_raw") if f.endswith('.png')]
    print(f"扫描 crops_raw/ 共 {len(files)} 张图...")

    found = 0

    for fname in sorted(files):
        if is_classified(fname):
            continue

        path = os.path.join("dataset/crops_raw", fname)
        img = cv2.imread(path)
        if img is None:
            continue

        # 用模型预测数字
        results = rec.model(img, verbose=False, conf=0.30, iou=0.35)

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            best_idx = int(boxes.conf.argmax())
            conf = float(boxes.conf[best_idx])
            cls_id = int(boxes.cls[best_idx])
            name = rec.ID2NAME.get(cls_id, "?")

            if name == '7' and conf >= 0.30:
                color = detect_color(img)
                if color == 'red':
                    shutil.copy2(path, os.path.join("dataset/by_class_54/7_红色", fname))
                elif color == 'black':
                    shutil.copy2(path, os.path.join("dataset/by_class_54/7_黑色", fname))
                else:
                    # 颜色不明，两边都放
                    shutil.copy2(path, os.path.join("dataset/by_class_54/7_红色", fname))
                    shutil.copy2(path, os.path.join("dataset/by_class_54/7_黑色", fname))

                found += 1
                print(f"  ✅ {fname} → 7_{color} (conf={conf:.3f})")
                break

    print(f"\n{'=' * 50}")
    print(f"共找到 {found} 张 7，已按颜色分类")
    print(f"{'=' * 50}")
    print("下一步：")
    print("  打开 dataset/by_class_54/7_红色/ → 心形拖入 7_红心/，菱形拖入 7_方块/")
    print("  打开 dataset/by_class_54/7_黑色/ → 桃形拖入 7_黑桃/，三叶草拖入 7_梅花/")


if __name__ == "__main__":
    main()
