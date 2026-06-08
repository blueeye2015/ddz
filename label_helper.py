#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注辅助工具 - 生成 YOLO 格式 txt 文件

用法1: 自动预标注（用现有检测模型辅助）
    python label_helper.py --auto image.jpg

用法2: 手动输入坐标
    python label_helper.py --manual image.jpg
    # 然后按提示输入: 类别 x1 y1 x2 y2

用法3: 直接代码调用
    python label_helper.py --code image.jpg "3 850 250 950 400" "3 950 250 1050 400"
"""

import cv2
import numpy as np
import os
import sys
import argparse

# 15 类牌
CARD_NAMES = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','SJ','BJ']

def parse_card_name(name):
    """解析牌名，返回类别索引"""
    name = name.upper().strip()
    if name in CARD_NAMES:
        return CARD_NAMES.index(name)
    # 别名
    aliases = {'小王': 'SJ', '大王': 'BJ', 'JOKER': 'BJ', 'JO': 'BJ'}
    if name in aliases:
        return CARD_NAMES.index(aliases[name])
    return None

def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """像素坐标转 YOLO 归一化坐标"""
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    nw = abs(x2 - x1) / img_w
    nh = abs(y2 - y1) / img_h
    # 限制在 0~1
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    nw = max(0, min(1, nw))
    nh = max(0, min(1, nh))
    return cx, cy, nw, nh

def yolo_to_box(cx, cy, nw, nh, img_w, img_h):
    """YOLO 归一化坐标转像素坐标"""
    x1 = int((cx - nw/2) * img_w)
    y1 = int((cy - nh/2) * img_h)
    x2 = int((cx + nw/2) * img_w)
    y2 = int((cy + nh/2) * img_h)
    return x1, y1, x2, y2

def auto_label(img_path, model_path='ddz_detect_best.pt', conf=0.15):
    """用现有检测模型自动预标注"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("需要安装 ultralytics: pip install ultralytics")
        return []
    
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return []
    
    h, w = img.shape[:2]
    results = model(img, conf=conf, iou=0.15, imgsz=1280, verbose=False)
    
    boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
            boxes.append((x1, y1, x2, y2))
    
    return boxes, w, h

def manual_label(img_path):
    """手动输入坐标交互模式"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return []
    
    h, w = img.shape[:2]
    print(f"\n图片尺寸: {w}x{h}")
    print("输入每张牌的标注，格式: 类别 x1 y1 x2 y2")
    print("类别: 3~10, J, Q, K, A, 2, SJ(小王), BJ(大王)")
    print("坐标: 左上角和右下角的像素坐标")
    print("示例: 3 850 250 950 380")
    print("输入空行结束\n")
    
    labels = []
    while True:
        line = input(f"  牌 {len(labels)+1}: ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 5:
            print("    格式错误，需要: 类别 x1 y1 x2 y2")
            continue
        
        cls_name, x1, y1, x2, y2 = parts[0], parts[1], parts[2], parts[3], parts[4]
        cls_idx = parse_card_name(cls_name)
        if cls_idx is None:
            print(f"    未知类别: {cls_name}")
            continue
        
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except ValueError:
            print("    坐标必须是整数")
            continue
        
        cx, cy, nw, nh = box_to_yolo(x1, y1, x2, y2, w, h)
        labels.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        print(f"    -> {CARD_NAMES[cls_idx]} at ({x1},{y1},{x2},{y2})")
    
    return labels

def code_label(img_path, annotations):
    """代码直接传入标注"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return []
    
    h, w = img.shape[:2]
    labels = []
    
    for ann in annotations:
        parts = ann.split()
        if len(parts) != 5:
            print(f"格式错误: {ann}")
            continue
        cls_name, x1, y1, x2, y2 = parts[0], parts[1], parts[2], parts[3], parts[4]
        cls_idx = parse_card_name(cls_name)
        if cls_idx is None:
            print(f"未知类别: {cls_name}")
            continue
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy, nw, nh = box_to_yolo(x1, y1, x2, y2, w, h)
        labels.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    
    return labels

def save_labels(img_path, labels, output_dir=None):
    """保存标注文件"""
    base = os.path.splitext(os.path.basename(img_path))[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f"{base}.txt")
    else:
        txt_path = os.path.join(os.path.dirname(img_path), f"{base}.txt")
    
    with open(txt_path, 'w') as f:
        f.write('\n'.join(labels))
    
    print(f"\n已保存: {txt_path}")
    print(f"共 {len(labels)} 个标注:")
    for line in labels:
        parts = line.split()
        cls_idx = int(parts[0])
        print(f"  {CARD_NAMES[cls_idx]}: {line}")

def visualize(img_path, labels):
    """可视化标注结果"""
    img = cv2.imread(img_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    debug = img.copy()
    
    for line in labels:
        parts = line.split()
        cls_idx = int(parts[0])
        cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1, y1, x2, y2 = yolo_to_box(cx, cy, nw, nh, w, h)
        
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug, CARD_NAMES[cls_idx], (x1, max(y1-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    out_path = os.path.splitext(img_path)[0] + "_labeled.jpg"
    cv2.imwrite(out_path, debug)
    print(f"可视化图: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLO 标注辅助工具")
    parser.add_argument("image", help="图片路径")
    parser.add_argument("--auto", action="store_true", help="自动预标注模式")
    parser.add_argument("--manual", action="store_true", help="手动输入坐标模式")
    parser.add_argument("--code", nargs="+", help="代码传入标注，如 '3 850 250 950 400'")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--viz", action="store_true", help="生成可视化图")
    
    args = parser.parse_args()
    
    if args.auto:
        print("自动预标注模式...")
        boxes, w, h = auto_label(args.image)
        if not boxes:
            print("未检测到任何框")
            return
        
        print(f"检测到 {len(boxes)} 个框，请为每个框输入类别:")
        labels = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cls_name = input(f"  框 {i+1} ({x1},{y1},{x2},{y2}): 类别? ").strip()
            cls_idx = parse_card_name(cls_name)
            if cls_idx is None:
                print(f"    跳过未知类别: {cls_name}")
                continue
            cx, cy, nw, nh = box_to_yolo(x1, y1, x2, y2, w, h)
            labels.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    
    elif args.manual:
        labels = manual_label(args.image)
    
    elif args.code:
        labels = code_label(args.image, args.code)
    
    else:
        print("请指定模式: --auto, --manual, 或 --code")
        print("示例:")
        print(f"  python label_helper.py image.jpg --auto")
        print(f"  python label_helper.py image.jpg --manual")
        print(f"  python label_helper.py image.jpg --code '3 850 250 950 400' '3 950 250 1050 400'")
        return
    
    if not labels:
        print("没有生成任何标注")
        return
    
    save_labels(args.image, labels, args.output)
    
    if args.viz:
        visualize(args.image, labels)

if __name__ == "__main__":
    main()
