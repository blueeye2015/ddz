#!/usr/bin/env python3
"""
底牌自动分割工具
用法：
    python extract_bottom_cards.py screenshots/bottom_001.png
    
输出：在 extracted_bottom/ 下生成分割好的单张牌图
你只需要把分好的图拖进 dataset/by_class/ 对应文件夹即可
"""

import cv2
import numpy as np
import sys
import os
from scipy.signal import find_peaks


def split_bottom_cards(image_path, output_dir="extracted_bottom"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取: {image_path}")
        return
    
    h, w = img.shape[:2]
    scale = 3
    img_big = cv2.resize(img, (w * scale, h * scale))
    hb, wb = img_big.shape[:2]
    
    # 找第一个缝隙（跳过文字标签）
    gray = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    proj = np.sum(binary == 255, axis=0)
    creases = np.max(proj) - proj
    peaks, _ = find_peaks(creases, distance=100, prominence=20)
    start = int(peaks[0]) if len(peaks) > 0 else 0
    
    # 有效区域内找缝隙
    sub = img_big[:, start:]
    gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    _, bin_sub = cv2.threshold(gray_sub, 220, 255, cv2.THRESH_BINARY)
    proj_sub = np.sum(bin_sub == 255, axis=0)
    creases_sub = np.max(proj_sub) - proj_sub
    peaks_sub, _ = find_peaks(creases_sub, distance=80, prominence=15)
    splits = [0] + sorted(peaks_sub) + [sub.shape[1]]
    
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    
    saved = 0
    for i in range(len(splits) - 1):
        x1 = start + int(splits[i])
        x2 = start + int(splits[i + 1])
        cw = (x2 - x1) // scale
        # 过滤文字标签(太宽)和噪声(太窄)
        if cw < 25 or cw > 70:
            continue
        
        card = img_big[0:hb, x1:x2]
        # 保存为原图尺寸（方便直接拖入训练集）
        card_resized = cv2.resize(card, (cw, h))
        fname = f"{output_dir}/{base}_card{saved+1:02d}_w{cw}.png"
        cv2.imwrite(fname, card_resized)
        print(f"  保存: {fname}  (宽{cw}px)")
        saved += 1
    
    print(f"\n共分割出 {saved} 张牌，请检查 extracted_bottom/ 文件夹")
    print("把分好的图拖进 dataset/by_class/ 对应数字文件夹，然后跑 train_classifier.py")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_bottom_cards.py <底牌截图路径>")
        print("示例: python extract_bottom_cards.py region_bottom_cards.png")
        sys.exit(1)
    
    for path in sys.argv[1:]:
        print(f"\n处理: {path}")
        split_bottom_cards(path)
