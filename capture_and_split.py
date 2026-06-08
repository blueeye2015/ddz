#!/usr/bin/env python3
"""
自动截图 + 三区域牌面分割工具
用法: python capture_and_split.py
按键: S=截图并分割所有区域, Q=退出

输出:
  captured_cards/hand/    - 手牌区分割图
  captured_cards/bottom/  - 底牌区分割图
  captured_cards/play/    - 出牌区分割图
  captured_cards/debug/   - 带框标注的调试图
"""

import cv2
import numpy as np
import os
import time
import json
import msvcrt
from datetime import datetime
from scipy.signal import find_peaks

# 从 ddz.py 导入（类定义在 if __name__ 之外，安全）
from ddz import WindowFinder, Capture, FixedCoords, ELEMENTS, BASE_WIDTH, BASE_HEIGHT, GAME_REGION
from ddz_yolo_recognizer import TwoStageRecognizerV3


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_hand_cards(img, recognizer, save_dir, prefix):
    """手牌区：YOLO检测 → 扩展窄框 → 裁剪保存"""
    h, w = img.shape[:2]
    results = recognizer.detect_model(img, conf=0.10, iou=0.15, imgsz=1280, verbose=False)
    
    boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
            yc = float(r.boxes.conf[i].cpu().numpy())
            if x1 < 0 or x2 > w + 10 or yc < 0.05:
                continue
            # 扩展窄框
            if x2 - x1 < 70:
                x1 = max(0, x2 - 70)
            boxes.append((x1, y1, x2, y2))
    
    # 按x排序
    boxes.sort(key=lambda b: b[0])
    
    # 去重（15px内）
    deduped = []
    for b in boxes:
        dup = False
        for d in deduped:
            if abs(b[0] - d[0]) < 15:
                dup = True
                break
        if not dup:
            deduped.append(b)
    
    saved = []
    debug = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(deduped):
        # 不扩展，YOLO框本身已足够准确
        sx1 = max(0, x1)
        sx2 = min(w, x2)
        card = img[y1:y2, sx1:sx2]
        if card.size == 0:
            continue
        fname = f"{save_dir}/{prefix}_{i+1:02d}.png"
        cv2.imwrite(fname, card)
        saved.append(fname)
        cv2.rectangle(debug, (sx1, y1), (sx2, y2), (0, 255, 0), 2)
        cv2.putText(debug, str(i+1), (sx1, max(y1-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return saved, debug


def split_bottom_cards(img, detector, save_dir, prefix):
    """底牌区：先YOLO检测（放大2倍），检测不到则回退投影分割"""
    h, w = img.shape[:2]
    
    # ========== 策略1：YOLO检测（放大2倍让小牌更易检出） ==========
    scale_yolo = 2
    img_big = cv2.resize(img, (w * scale_yolo, h * scale_yolo))
    
    if hasattr(detector, 'detect_model'):
        results = detector.detect_model(img_big, conf=0.05, iou=0.15, imgsz=1280, verbose=False)
    else:
        results = detector(img_big, conf=0.05, iou=0.15, imgsz=1280, verbose=False)
    
    boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
            bw, bh = x2 - x1, y2 - y1
            ratio = bw / bh if bh > 0 else 0
            # 底牌小，放宽宽高比和高度限制，但只保留像牌的竖条
            if 0.18 <= ratio <= 0.65 and bh >= 35:
                boxes.append((x1, y1, x2, y2))
    
    # 去重（iou>0.5合并）
    deduped = []
    for b in sorted(boxes, key=lambda b: b[0]):
        x1, y1, x2, y2 = b
        dup = False
        for dx1, dy1, dx2, dy2 in deduped:
            ix1, iy1, ix2, iy2 = max(x1, dx1), max(y1, dy1), min(x2, dx2), min(y2, dy2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            if iw * ih > 0:
                ua = (x2 - x1) * (y2 - y1) + (dx2 - dx1) * (dy2 - dy1) - iw * ih
                if iw * ih / ua > 0.5:
                    dup = True
                    break
        if not dup:
            deduped.append(b)
    
    # YOLO成功检测到2-4张，直接用YOLO框裁剪（不扩展，避免串味）
    if 2 <= len(deduped) <= 4:
        saved = []
        debug = img.copy()
        for i, (x1, y1, x2, y2) in enumerate(sorted(deduped, key=lambda b: b[0])):
            rx1 = max(0, x1 // scale_yolo)
            rx2 = min(w, x2 // scale_yolo)
            ry1 = max(0, y1 // scale_yolo)
            ry2 = min(h, y2 // scale_yolo)
            card = img[ry1:ry2, rx1:rx2]
            if card.size == 0:
                continue
            fname = f"{save_dir}/{prefix}_{i+1:02d}.png"
            cv2.imwrite(fname, card)
            saved.append(fname)
            cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.putText(debug, str(i+1), (rx1, max(ry1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return saved, debug
    
    # ========== 策略2：回退到投影分割 ==========
    scale = 3
    img_big = cv2.resize(img, (w * scale, h * scale))
    hb, wb = img_big.shape[:2]
    
    gray = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    proj = np.sum(binary == 255, axis=0)
    creases = np.max(proj) - proj
    
    left_limit = int(wb * 0.55)
    left_region = creases[:left_limit]
    peaks, props = find_peaks(left_region, distance=50, prominence=20)
    start = 0
    if len(peaks) > 0:
        best_peak = None
        best_prom = 0
        for i, p in enumerate(peaks):
            if wb - p >= 120 * scale and props['prominences'][i] > best_prom:
                best_peak = p
                best_prom = props['prominences'][i]
        if best_peak is not None:
            start = int(best_peak)
    
    right_region = creases[start:]
    peaks_sub, props_sub = find_peaks(right_region, distance=50, prominence=10)
    splits = [0] + sorted(peaks_sub.tolist()) + [right_region.shape[0]]
    
    candidates = []
    for i in range(len(splits) - 1):
        x1 = start + int(splits[i])
        x2 = start + int(splits[i + 1])
        cw = (x2 - x1) // scale
        if 20 <= cw <= 80:
            candidates.append((x1, x2, cw))
    
    merged = []
    for x1, x2, cw in sorted(candidates):
        if not merged or x1 - merged[-1][1] > 12 * scale:
            merged.append((x1, x2, cw))
    candidates = merged
    
    if len(candidates) > 4:
        filtered = [c for c in candidates if c[0] > start + 50 * scale]
        if len(filtered) >= 2:
            candidates = filtered[:3]
        else:
            candidates = []
    
    if len(candidates) < 2 and start > 0:
        est_w = (wb - start) // 3
        candidates = []
        for i in range(3):
            x1 = start + i * est_w
            x2 = min(start + (i + 1) * est_w, wb)
            cw = (x2 - x1) // scale
            if 20 <= cw <= 80:
                candidates.append((x1, x2, cw))
    
    saved = []
    debug = img.copy()
    idx = 0
    for x1, x2, cw in candidates:
        rx1 = max(0, x1 // scale - 3)
        rx2 = min(w, x2 // scale + 3)
        card = img[0:h, rx1:rx2]
        if card.size == 0:
            continue
        fname = f"{save_dir}/{prefix}_{idx+1:02d}.png"
        cv2.imwrite(fname, card)
        saved.append(fname)
        cv2.rectangle(debug, (rx1, 0), (rx2, h), (0, 255, 0), 2)
        cv2.putText(debug, str(idx+1), (rx1, max(h - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        idx += 1
    
    return saved, debug


def split_play_cards(img, recognizer, save_dir, prefix):
    """出牌区：YOLO检测 → 过滤假阳性 → 裁剪保存"""
    h, w = img.shape[:2]
    results = recognizer.detect_model(img, conf=0.10, iou=0.20, imgsz=1280, verbose=False)
    
    boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
            yc = float(r.boxes.conf[i].cpu().numpy())
            bw, bh = x2 - x1, y2 - y1
            ratio = bw / bh if bh > 0 else 0
            # 过滤按钮/文字/边框
            if not (0.25 <= ratio <= 0.55 and bh >= 80 and yc >= 0.10):
                continue
            boxes.append((x1, y1, x2, y2))
    
    boxes.sort(key=lambda b: b[0])
    
    # 去重
    deduped = []
    for b in boxes:
        dup = False
        for d in deduped:
            if abs(b[0] - d[0]) < 15:
                dup = True
                break
        if not dup:
            deduped.append(b)
    
    saved = []
    debug = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(deduped):
        # 不扩展，YOLO框本身已足够准确
        sx1 = max(0, x1)
        sx2 = min(w, x2)
        card = img[y1:y2, sx1:sx2]
        if card.size == 0:
            continue
        fname = f"{save_dir}/{prefix}_{i+1:02d}.png"
        cv2.imwrite(fname, card)
        saved.append(fname)
        cv2.rectangle(debug, (sx1, y1), (sx2, y2), (0, 255, 0), 2)
        cv2.putText(debug, str(i+1), (sx1, max(y1-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return saved, debug


def main():
    print("=" * 50)
    print("自动截图 + 三区域牌面分割工具")
    print("=" * 50)
    
    # 初始化窗口
    finder = WindowFinder()
    hwnd = finder.find()
    if not hwnd:
        print("❌ 未找到游戏窗口")
        return
    
    client = finder.get_client_rect()
    cap = Capture()
    bbox = (client['x'], client['y'],
            client['x'] + client['width'],
            client['y'] + client['height'])
    
    # 加载配置
    config_file = "ddz_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            saved = json.load(f)
            global ELEMENTS, BASE_WIDTH, BASE_HEIGHT, GAME_REGION
            ELEMENTS = saved['elements']
            BASE_WIDTH = saved['base_width']
            BASE_HEIGHT = saved['base_height']
            if 'game_region' in saved:
                GAME_REGION = saved['game_region']
        print(f"✅ 已加载配置: {config_file}")
    
    # 建立坐标系统
    coords = FixedCoords(client)
    
    # DPI校准：截图实际尺寸 vs 客户区逻辑尺寸
    test_img = cap.capture(bbox)
    actual_w, actual_h = test_img.shape[1], test_img.shape[0]
    if abs(actual_w - client['width']) > 10 or abs(actual_h - client['height']) > 10:
        print(f"[DPI校准] 截图尺寸({actual_w}x{actual_h})与客户区({client['width']}x{client['height']})不一致，重新校准坐标")
        coords = FixedCoords(client, actual_size=(actual_w, actual_h))
    
    # 加载YOLO（只用于检测，不需要分类）
    print("📦 加载 YOLO 检测模型...")
    recognizer = TwoStageRecognizerV3()
    
    # 创建输出目录
    ensure_dir("raw_screenshots")
    ensure_dir("captured_cards/hand")
    ensure_dir("captured_cards/bottom")
    ensure_dir("captured_cards/play")
    ensure_dir("captured_cards/debug")
    
    shot_count = 0
    
    print("\n按键说明:")
    print("  S = 截图并分割三区域")
    print("  Q = 退出")
    print()
    
    while True:
        print("等待按键 (S=截图, Q=退出)...")
        key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
        
        if key == 'q':
            print("退出")
            break
        
        if key != 's':
            continue
        
        shot_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{shot_count:03d}_{ts}"
        print(f"\n📸 截图 #{shot_count} ...")
        
        # 截图
        img = cap.capture(bbox)
        
        # 1. 保存完整原图
        raw_path = f"raw_screenshots/{prefix}_full.png"
        cv2.imwrite(raw_path, img)
        print(f"  💾 原图已保存: {raw_path}")
        
        # 2. 分割三个区域
        regions = {
            'hand': ('my_hand', split_hand_cards),
            'bottom': ('bottom_cards', split_bottom_cards),
            'play': ('play_area', split_play_cards),
        }
        
        for region_name, (elem_name, split_fn) in regions.items():
            roi = coords.extract(img, elem_name)
            if roi.size == 0:
                print(f"  ⚠️ {region_name}: 区域提取失败")
                continue
            
            save_dir = f"captured_cards/{region_name}"
            
            saved, debug = split_fn(roi, recognizer, save_dir, prefix)
            
            print(f"  ✅ {region_name}: 分割出 {len(saved)} 张牌")
            
            # 保存调试图
            debug_path = f"captured_cards/debug/{prefix}_{region_name}.png"
            cv2.imwrite(debug_path, debug)
        
        print(f"  📂 调试图: captured_cards/debug/{prefix}_*.png")
        print()
    
    print(f"\n总计截图 {shot_count} 次，所有分割图保存在 captured_cards/")
    print("把分好的图拖进 dataset/by_class/ 对应文件夹，然后跑 train_classifier.py")


if __name__ == "__main__":
    main()
