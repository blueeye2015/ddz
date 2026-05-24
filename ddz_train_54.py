#!/usr/bin/env python3
"""54 类 YOLOv8 训练脚本"""

import os
import sys


def check_dataset():
    train_img = os.path.exists("dataset/images_54/train") and len(os.listdir("dataset/images_54/train")) > 0
    val_img = os.path.exists("dataset/images_54/val") and len(os.listdir("dataset/images_54/val")) > 0
    if not all([train_img, val_img]):
        print("[错误] 54 类数据集不完整")
        print("  1. python prepare_54.py")
        print("  2. 手动按花色分类")
        print("  3. python generate_yolo_labels_54.py")
        return False
    return True


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        sys.exit(1)

    if not check_dataset():
        sys.exit(1)

    print("=" * 50)
    print("YOLOv8 54 类扑克牌检测训练")
    print("=" * 50)

    model = YOLO("yolov8n.pt")

    print("\n开始训练...")
    model.train(
        data="ddz_54_fixed.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=0,
        patience=20,
        device="cpu",
        project="ddz_runs_54",
        name="train",
        exist_ok=True,
        verbose=True
    )

    possible_paths = [
        os.path.join("runs", "detect", "ddz_runs_54", "train", "weights", "best.pt"),
        os.path.join("ddz_runs_54", "train", "weights", "best.pt"),
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
