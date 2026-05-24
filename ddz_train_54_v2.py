#!/usr/bin/env python3
"""54 类优化训练：更多轮数 + 更强增强 + 更低学习率"""

import os
import sys


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        sys.exit(1)

    if not os.path.exists("dataset_54/images/train"):
        print("先运行: python fix_path.py")
        sys.exit(1)

    print("=" * 50)
    print("54 类优化训练 v2")
    print("=" * 50)

    model = YOLO("yolov8n.pt")

    # 优化参数
    model.train(
        data="ddz_54_fixed.yaml",
        epochs=200,           # 增加到 200 轮
        imgsz=640,
        batch=8,
        workers=0,
        patience=50,          # 早停耐心加大，让模型充分学习
        device="cpu",
        project="ddz_runs_54",
        name="train_v2",
        exist_ok=True,
        verbose=True,
        # 数据增强
        hsv_h=0.015,          # 色调增强（灰度图调小）
        hsv_s=0.7,            # 饱和度增强
        hsv_v=0.4,            # 亮度增强
        degrees=5,            # 随机旋转 ±5度
        translate=0.1,        # 平移
        scale=0.5,            # 缩放
        shear=2,              # 剪切
        flipud=0.0,           # 不上下翻转（牌不能倒）
        fliplr=0.5,           # 左右翻转
        mosaic=1.0,           # Mosaic 增强
        mixup=0.1,            # Mixup 增强
        copy_paste=0.1,       # Copy-Paste 增强
        # 学习率
        lr0=0.001,            # 初始学习率降低
        lrf=0.01,             # 最终学习率
        # 权重衰减
        weight_decay=0.0005,
    )

    possible_paths = [
        os.path.join("runs", "detect", "ddz_runs_54", "train_v2", "weights", "best.pt"),
        os.path.join("ddz_runs_54", "train_v2", "weights", "best.pt"),
    ]

    best_path = None
    for p in possible_paths:
        if os.path.exists(p):
            best_path = p
            break

    if best_path:
        print(f"\n✅ 训练完成！最佳模型: {best_path}")
        print(f"复制到项目根目录: copy {best_path} ddz_yolo_54.pt")
    else:
        print("[警告] 未找到最佳模型")


if __name__ == "__main__":
    main()
