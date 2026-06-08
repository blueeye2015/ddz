#!/usr/bin/env python3
"""
分割已有截图（支持两种方式）

方式1（默认）: 分割 region_*.png（ddz.py 提取的区域图）
    python split_existing.py

方式2: 分割 raw_screenshots/ 下的完整截图
    python split_existing.py --raw
"""

import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime
from scipy.signal import find_peaks
from ultralytics import YOLO


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_region(full_img, elem, scale_x, scale_y):
    """从完整截图中提取指定区域"""
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


def split_hand_cards(img, yolo, save_dir, prefix):
    h, w = img.shape[:2]
    results = yolo(img, conf=0.10, iou=0.15, imgsz=1280, verbose=False)
    
    boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
            yc = float(r.boxes.conf[i].cpu().numpy())
            if x1 < 0 or x2 > w + 10 or yc < 0.05:
                continue
            if x2 - x1 < 70:
                x1 = max(0, x2 - 70)
            boxes.append((x1, y1, x2, y2))
    
    boxes.sort(key=lambda b: b[0])
    
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


def split_play_cards(img, yolo, save_dir, prefix):
    h, w = img.shape[:2]
    results = yolo(img, conf=0.10, iou=0.20, imgsz=1280, verbose=False)
    
    boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
            yc = float(r.boxes.conf[i].cpu().numpy())
            bw, bh = x2 - x1, y2 - y1
            ratio = bw / bh if bh > 0 else 0
            if not (0.25 <= ratio <= 0.55 and bh >= 80 and yc >= 0.10):
                continue
            boxes.append((x1, y1, x2, y2))
    
    boxes.sort(key=lambda b: b[0])
    
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


def process_region_image(img, region_name, yolo, save_dir, prefix):
    """处理单张区域图"""
    if region_name == 'hand':
        return split_hand_cards(img, yolo, save_dir, prefix)
    elif region_name == 'bottom':
        return split_bottom_cards(img, yolo, save_dir, prefix)
    else:
        return split_play_cards(img, yolo, save_dir, prefix)


def main():
    use_raw = '--raw' in sys.argv
    
    print("=" * 50)
    if use_raw:
        print("分割 raw_screenshots/ 下的完整截图")
    else:
        print("分割 region_*.png 区域图")
    print("=" * 50)
    
    # 加载YOLO
    print("📦 加载 YOLO...")
    yolo = YOLO('ddz_detect_best.pt')
    
    # 创建输出目录
    ensure_dir("captured_cards/hand")
    ensure_dir("captured_cards/bottom")
    ensure_dir("captured_cards/play")
    ensure_dir("captured_cards/debug")
    
    if use_raw:
        # 方式2: 处理 raw_screenshots/ 下的完整截图
        raw_dir = "raw_screenshots"
        if not os.path.exists(raw_dir):
            print(f"❌ {raw_dir}/ 不存在")
            return
        
        # 加载配置获取坐标
        config_file = "ddz_config.json"
        if not os.path.exists(config_file):
            print("❌ 需要 ddz_config.json 才能从完整截图裁剪区域")
            return
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        elements = config['elements']
        base_w = config['base_width']
        base_h = config['base_height']
        
        raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.png')])
        print(f"\n找到 {len(raw_files)} 张完整截图")
        
        for fname in raw_files:
            print(f"\n处理: {fname}")
            img = cv2.imread(os.path.join(raw_dir, fname))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            scale_x = w / base_w
            scale_y = h / base_h
            prefix = os.path.splitext(fname)[0]
            
            regions = {
                'hand': 'my_hand',
                'bottom': 'bottom_cards',
                'play': 'play_area',
            }
            
            for region_name, elem_name in regions.items():
                roi = extract_region(img, elements[elem_name], scale_x, scale_y)
                if roi.size == 0:
                    print(f"  ⚠️ {region_name}: 区域提取失败")
                    continue
                
                save_dir = f"captured_cards/{region_name}"
                saved, debug = process_region_image(roi, region_name, yolo, save_dir, prefix)
                print(f"  ✅ {region_name}: {len(saved)} 张")
                
                debug_path = f"captured_cards/debug/{prefix}_{region_name}.png"
                cv2.imwrite(debug_path, debug)
    
    else:
        # 方式1: 处理 region_*.png
        files = {
            'hand': 'region_my_hand.png',
            'bottom': 'region_bottom_cards.png',
            'play': 'region_play_area.png',
        }
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for region_name, filename in files.items():
            if not os.path.exists(filename):
                print(f"⚠️ 跳过 {filename} (不存在)")
                continue
            
            img = cv2.imread(filename)
            if img is None:
                continue
            
            print(f"\n处理 {filename}...")
            save_dir = f"captured_cards/{region_name}"
            prefix = f"{ts}_{region_name}"
            
            saved, debug = process_region_image(img, region_name, yolo, save_dir, prefix)
            print(f"  ✅ {region_name}: {len(saved)} 张")
            
            debug_path = f"captured_cards/debug/{prefix}.png"
            cv2.imwrite(debug_path, debug)
    
    print(f"\n完成！分割图在 captured_cards/ 下")


if __name__ == "__main__":
    main()
