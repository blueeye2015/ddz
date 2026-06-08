#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 训练脚本
模仿 3_yolo_model_training.ipynb 流程，适配到 ultralytics YOLOv8

用法:
    python train_yolov8.py --data labels_my-project-name --epochs 100 --imgsz 640
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# 54 类牌名称（52 标准 + 大小王）
# 前 52 类对应 Roboflow 标准扑克牌顺序
CARD_NAMES_54 = [
    '10C','10D','10H','10S',  # 0-3
    '2C','2D','2H','2S',       # 4-7
    '3C','3D','3H','3S',       # 8-11
    '4C','4D','4H','4S',       # 12-15
    '5C','5D','5H','5S',       # 16-19
    '6C','6D','6H','6S',       # 20-23
    '7C','7D','7H','7S',       # 24-27
    '8C','8D','8H','8S',       # 28-31
    '9C','9D','9H','9S',       # 32-35
    'AC','AD','AH','AS',       # 36-39
    'JC','JD','JH','JS',       # 40-43
    'KC','KD','KH','KS',       # 44-47
    'QC','QD','QH','QS',       # 48-51
    'SJ','BJ'                   # 52-53 (小王、大王)
]

def prepare_dataset(source_dir, output_dir="dataset_yolov8", train_ratio=0.8):
    """
    准备 YOLOv8 标准数据集结构
    输入: source_dir 下的 .png 和 .txt 文件
    输出: dataset_yolov8/images/train, images/val, labels/train, labels/val
    """
    print(f"[1/4] 准备数据集: {source_dir}")
    
    # 查找所有图片和对应的标注
    png_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith('.png')])
    
    pairs = []
    for png in png_files:
        base = os.path.splitext(png)[0]
        txt = base + '.txt'
        txt_path = os.path.join(source_dir, txt)
        if os.path.exists(txt_path):
            pairs.append((png, txt))
    
    if not pairs:
        print(f"错误: {source_dir} 下没有找到配对的 png+txt 文件")
        sys.exit(1)
    
    print(f"  找到 {len(pairs)} 对有效数据")
    
    # 随机打乱（模仿 notebook 中的 shuffle 逻辑）
    random.seed(42)
    random.shuffle(pairs)
    
    # 划分 train/val
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"  Train: {len(train_pairs)} 张")
    print(f"  Val:   {len(val_pairs)} 张")
    
    # 创建目录并复制文件
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = os.path.join(output_dir, f"images/{split_name}")
        lbl_dir = os.path.join(output_dir, f"labels/{split_name}")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for png, txt in split_pairs:
            shutil.copy2(os.path.join(source_dir, png), os.path.join(img_dir, png))
            shutil.copy2(os.path.join(source_dir, txt), os.path.join(lbl_dir, txt))
    
    return output_dir, len(train_pairs), len(val_pairs)

def create_yaml(dataset_dir, num_classes=54):
    """生成 data.yaml"""
    print(f"[2/4] 生成 data.yaml")
    
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    names_str = ", ".join([f"'{n}'" for n in CARD_NAMES_54[:num_classes]])
    
    yaml_content = f"""train: images/train
val: images/val

nc: {num_classes}
names: [{names_str}]
"""
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"  已保存: {yaml_path}")
    return yaml_path

def download_weights(model_name="yolov8n.pt"):
    """下载/检查预训练权重"""
    print(f"[3/4] 检查预训练权重: {model_name}")
    
    if not os.path.exists(model_name):
        print(f"  正在下载 {model_name}...")
        # ultralytics 会在第一次使用时自动下载
        _ = YOLO(model_name)
        print(f"  下载完成")
    else:
        print(f"  已存在")
    
    return model_name

def train(data_yaml, weights="yolov8n.pt", epochs=100, imgsz=640, batch=8):
    """开始训练"""
    print(f"[4/4] 开始训练")
    print(f"  模型: {weights}")
    print(f"  数据: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    
    # 加载模型
    model = YOLO(weights)
    
    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,           # 早停耐心值
        save=True,             # 保存最佳模型
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True
    )
    
    print(f"\n训练完成!")
    # 获取最佳模型路径
    best_path = model.trainer.best if hasattr(model, 'trainer') and hasattr(model.trainer, 'best') else "runs/detect/train/weights/best.pt"
    print(f"最佳模型: {best_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    parser.add_argument("--data", type=str, default="labels_my-project-name",
                        help="数据集目录，默认 labels_my-project-name")
    parser.add_argument("--output", type=str, default="dataset_yolov8",
                        help="输出数据集目录，默认 dataset_yolov8")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数，默认 100")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入尺寸，默认 640")
    parser.add_argument("--batch", type=int, default=8,
                        help="batch size，默认 8")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="训练集比例，默认 0.8")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                        help="预训练权重，默认 yolov8n.pt")
    parser.add_argument("--nc", type=int, default=54,
                        help="类别数，默认 54")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("YOLOv8 训练脚本")
    print("=" * 50)
    
    # 1. 准备数据
    dataset_dir, train_num, val_num = prepare_dataset(
        args.data, args.output, args.train_ratio
    )
    
    # 2. 生成 yaml
    yaml_path = create_yaml(dataset_dir, args.nc)
    
    # 3. 下载权重
    weights_path = download_weights(args.weights)
    
    # 4. 训练
    if train_num == 0:
        print("错误: 训练集为空")
        sys.exit(1)
    
    print()
    train(yaml_path, weights_path, args.epochs, args.imgsz, args.batch)

if __name__ == "__main__":
    main()
