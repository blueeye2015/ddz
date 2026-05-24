#!/usr/bin/env python3
"""
半自动数据集准备脚本
1. 读取 screenshots/ 下的手牌截图
2. 自动分割并裁剪单张牌到 dataset/crops_raw/
3. 保存元数据到 dataset/crops_meta.json

下一步：用户手动把 crops_raw/ 下的图按类别拖进 dataset/by_class/ 对应文件夹
"""

import os
import json
import cv2
import numpy as np
from glob import glob


def split_cards(hand_img: np.ndarray, y_limit: float = 0.50) -> list:
    """从手牌图中分割出每张牌的区域（简化版，不依赖 ddz.py）"""
    if hand_img.size == 0:
        return []

    h_total, w_total = hand_img.shape[:2]

    # 双通道文字检测：HSV 抓红色数字 + 灰度自适应阈值抓黑字
    hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 80, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)

    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

    combined_mask = cv2.bitwise_or(red_mask, black_mask)
    kernel_open = np.ones((2, 2), np.uint8)
    inv_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 4 <= w <= 50 and 8 <= h <= 60 and area >= 20 and y < h_total * y_limit:
            cx = x + w // 2
            text_boxes.append((cx, x, y, w, h))

    if not text_boxes:
        return []

    text_boxes.sort(key=lambda b: b[0])
    GAP_THRESHOLD = 22

    groups = [[text_boxes[0]]]
    for box in text_boxes[1:]:
        if box[0] - groups[-1][-1][0] > GAP_THRESHOLD:
            groups.append([box])
        else:
            groups[-1].append(box)

    cards = []
    for group in groups:
        min_text_x = min(b[1] for b in group)
        x1 = max(0, min_text_x - 5)
        x2 = min(w_total, x1 + 78)
        cw = x2 - x1
        if cw > 25:
            cards.append((x1, 0, cw, h_total))

    return cards


def main():
    os.makedirs("dataset/crops_raw", exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    screenshots = glob("screenshots/*.png") + glob("screenshots/*.jpg") + glob("screenshots/*.jpeg")
    if not screenshots:
        print("[提示] screenshots/ 目录下未找到截图")
        print("请先运行 ddz.py 保存手牌截图，或手动把截图放到 screenshots/ 目录")
        print("截图命名建议: hand_001.png, hand_002.png ...")
        return

    meta = {}
    total_crops = 0

    for img_path in screenshots:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        h, w = img.shape[:2]
        cards = split_cards(img, y_limit=0.5)

        print(f"[{img_name}] 原图 {w}x{h}，分割出 {len(cards)} 张牌")

        for i, (x, y, cw, ch) in enumerate(cards):
            crop = img[y:y+ch, x:x+cw]
            crop_name = f"{img_name}_{i:02d}.png"
            crop_path = os.path.join("dataset/crops_raw", crop_name)
            cv2.imwrite(crop_path, crop)

            meta[crop_name] = {
                "source": img_path,
                "x": int(x), "y": int(y),
                "w": int(cw), "h": int(ch),
                "img_w": int(w), "img_h": int(h)
            }
            total_crops += 1

    with open("dataset/crops_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！共裁剪 {total_crops} 张牌到 dataset/crops_raw/")
    print("\n下一步操作：")
    print("1. 打开 dataset/crops_raw/ 文件夹")
    print("2. 逐张查看裁剪图，把它们拖进 dataset/by_class/ 下的对应类别文件夹")
    print("   可用类别: 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, SJ(小王), BJ(大王)")
    print("3. 分完后运行: python generate_yolo_labels.py")


if __name__ == "__main__":
    main()
