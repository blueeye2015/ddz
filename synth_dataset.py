#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据合成器
从斗地主游戏截图分割出的单牌图，合成类似 Playing-card 的干净训练样本
自动生成 YOLO 格式标注

用法:
    python synth_dataset.py --num 1000 --output dataset_synth
"""

import cv2
import numpy as np
import os
import sys
import random
import argparse
from pathlib import Path
import shutil

# 斗地主牌数字到索引的映射（15类）
CARD_NAMES = ['3','4','5','6','7','8','9','10','J','Q','K','A','2','SJ','BJ']

def load_source_cards(source_dir="dataset/by_class"):
    """从按类别组织的目录加载单牌图"""
    cards = {name: [] for name in CARD_NAMES}
    
    if not os.path.exists(source_dir):
        print(f"源目录不存在: {source_dir}")
        return cards
    
    for class_name in CARD_NAMES:
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(class_dir, fname)
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    cards[class_name].append(img)
    
    total = sum(len(v) for v in cards.values())
    print(f"加载了 {total} 张单牌图")
    for name in CARD_NAMES:
        print(f"  {name}: {len(cards[name])} 张")
    
    return cards


def random_background(size, style="mixed"):
    """生成随机背景"""
    w, h = size
    
    if style == "solid":
        # 纯色背景
        color = [random.randint(0, 255) for _ in range(3)]
        bg = np.full((h, w, 3), color, dtype=np.uint8)
    
    elif style == "gradient":
        # 渐变背景
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        c1 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        c2 = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        for x in range(w):
            ratio = x / w
            color = c1 * (1 - ratio) + c2 * ratio
            bg[:, x] = color
    
    elif style == "noise":
        # 噪点背景
        bg = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (21, 21), 0)
    
    else:  # mixed - 随机选一种
        return random_background(size, random.choice(["solid", "gradient", "noise"]))
    
    return bg


def paste_card(bg, card_img, x, y, angle=0, scale=1.0):
    """把单牌图粘贴到背景上，支持旋转和缩放"""
    h, w = card_img.shape[:2]
    
    # 缩放
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        card_img = cv2.resize(card_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
    
    # 旋转
    if angle != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 计算旋转后的尺寸
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        card_img = cv2.warpAffine(card_img, M, (new_w, new_h), borderValue=(128, 128, 128))
        h, w = new_h, new_w
    
    # 边界检查
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return None, None
    
    # 粘贴（支持透明通道）
    if card_img.shape[2] == 4 if len(card_img.shape) == 3 else False:
        # 有 alpha 通道
        alpha = card_img[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (alpha * card_img[:, :, c] + (1 - alpha) * bg[y:y+h, x:x+w, c]).astype(np.uint8)
    else:
        # 无 alpha，直接覆盖
        bg[y:y+h, x:x+w] = card_img[:, :, :3] if len(card_img.shape) == 3 else cv2.cvtColor(card_img, cv2.COLOR_GRAY2BGR)
    
    return bg, (x, y, w, h)


def synthesize_image(cards, canvas_size=(640, 640), num_cards_range=(1, 5)):
    """合成一张训练图"""
    # 生成背景
    bg = random_background(canvas_size)
    
    # 决定放几张牌
    num_cards = random.randint(*num_cards_range)
    
    # 随机选牌
    selected = []
    for _ in range(num_cards):
        # 随机选一个类别
        available = [k for k, v in cards.items() if len(v) > 0]
        if not available:
            break
        class_name = random.choice(available)
        card_img = random.choice(cards[class_name])
        selected.append((class_name, card_img))
    
    if not selected:
        return None, []
    
    # 计算牌的尺寸（统一缩放）
    target_card_height = random.randint(120, 180)
    
    # 水平排列，有一定重叠
    labels = []
    current_x = random.randint(20, 100)
    base_y = random.randint(50, canvas_size[1] - target_card_height - 50)
    
    for class_name, card_img in selected:
        # 计算缩放比例
        h, w = card_img.shape[:2]
        scale = target_card_height / h
        
        # 添加随机扰动
        angle = random.uniform(-8, 8)  # 轻微旋转
        scale *= random.uniform(0.9, 1.1)  # 轻微缩放
        
        # Y 方向也加点随机偏移
        y_offset = random.randint(-15, 15)
        y = base_y + y_offset
        
        # 粘贴
        result, bbox = paste_card(bg, card_img, current_x, y, angle=angle, scale=scale)
        if result is None:
            continue
        
        bg = result
        x, y, bw, bh = bbox
        
        # YOLO 格式标注: class x_center y_center width height (归一化)
        cx = (x + bw / 2) / canvas_size[0]
        cy = (y + bh / 2) / canvas_size[1]
        nw = bw / canvas_size[0]
        nh = bh / canvas_size[1]
        
        # 类别索引（15类分类器）
        class_idx = CARD_NAMES.index(class_name)
        labels.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        # 下一张牌的位置（有重叠）
        overlap = random.randint(-int(bw * 0.3), int(bw * 0.1))  # 重叠 0~30%
        current_x = x + bw + overlap
        
        # 如果超出画布，停止
        if current_x > canvas_size[0] - 50:
            break
    
    return bg, labels


def main():
    parser = argparse.ArgumentParser(description="合成斗地主训练数据")
    parser.add_argument("--source", type=str, default="dataset/by_class",
                        help="单牌图源目录，默认 dataset/by_class")
    parser.add_argument("--output", type=str, default="dataset_synth",
                        help="输出目录，默认 dataset_synth")
    parser.add_argument("--num", type=int, default=500,
                        help="合成图片数量，默认 500")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="训练集比例，默认 0.8")
    parser.add_argument("--size", type=int, default=640,
                        help="画布尺寸，默认 640")
    parser.add_argument("--min-cards", type=int, default=1,
                        help="最少牌数，默认 1")
    parser.add_argument("--max-cards", type=int, default=5,
                        help="最多牌数，默认 5")
    
    args = parser.parse_args()
    
    # 加载单牌图
    print("Loading source cards...")
    cards = load_source_cards(args.source)
    
    total_cards = sum(len(v) for v in cards.values())
    if total_cards == 0:
        print("没有可用的单牌图，请先运行 capture_and_split.py 生成分割数据")
        sys.exit(1)
    
    # 创建输出目录
    out_dir = args.output
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    
    # 生成数据
    print(f"\nGenerating {args.num} synthetic images...")
    train_num = int(args.num * args.train_ratio)
    
    for i in range(args.num):
        is_train = i < train_num
        split = "train" if is_train else "val"
        
        img, labels = synthesize_image(
            cards,
            canvas_size=(args.size, args.size),
            num_cards_range=(args.min_cards, args.max_cards)
        )
        
        if img is None or not labels:
            continue
        
        fname = f"synth_{i:05d}"
        img_path = os.path.join(out_dir, f"images/{split}/{fname}.jpg")
        label_path = os.path.join(out_dir, f"labels/{split}/{fname}.txt")
        
        cv2.imwrite(img_path, img)
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{args.num}")
    
    # 生成 data.yaml
    yaml_content = f"""train: ../images/train
val: ../images/val

nc: {len(CARD_NAMES)}
names: {CARD_NAMES}
"""
    with open(os.path.join(out_dir, "data.yaml"), 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDone! Output: {out_dir}/")
    print(f"  Train: {train_num} images")
    print(f"  Val: {args.num - train_num} images")
    print(f"\nTo train YOLO on this dataset:")
    print(f"  yolo detect train data={out_dir}/data.yaml model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
