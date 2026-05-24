#!/usr/bin/env python3
"""
YOLOv8 训练脚本
使用前请确保已安装 ultralytics:
    pip install ultralytics

首次运行会自动下载 yolov8n.pt 预训练权重（约 6MB）
"""

import os
import sys


def check_dataset():
    """检查数据集是否已准备好"""
    train_images = os.path.exists("dataset/images/train") and len(os.listdir("dataset/images/train")) > 0
    val_images = os.path.exists("dataset/images/val") and len(os.listdir("dataset/images/val")) > 0
    train_labels = os.path.exists("dataset/labels/train") and len(os.listdir("dataset/labels/train")) > 0
    val_labels = os.path.exists("dataset/labels/val") and len(os.listdir("dataset/labels/val")) > 0

    if not all([train_images, val_images, train_labels, val_labels]):
        print("[错误] 数据集不完整，请按以下步骤操作:")
        print("  1. python prepare_dataset.py")
        print("  2. 手动分类 crops_raw/ 下的图片到 by_class/ 对应文件夹")
        print("  3. python generate_yolo_labels.py")
        return False
    return True


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics，请运行:")
        print("  pip install ultralytics")
        sys.exit(1)

    if not check_dataset():
        sys.exit(1)

    print("=" * 50)
    print("YOLOv8 斗地主牌检测训练")
    print("=" * 50)

    # 加载预训练模型（nano 版，最小最快，CPU 友好）
    model = YOLO("yolov8n.pt")

    # 开始训练
    # epochs: 训练轮数，100 张图建议 50-100 轮
    # imgsz: 输入尺寸，640 是标准
    # batch: 批次大小，CPU 建议 4-8
    # workers: 数据加载线程，Windows 建议 0 避免多进程问题
    # patience: 早停耐心值，10 轮不提升就停止
    print("\n开始训练...")
    print("参数: epochs=100, imgsz=640, batch=8, workers=0")
    print("按 Ctrl+C 可随时中断\n")

    model.train(
        data="ddz.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=0,
        patience=20,
        device="cpu",           # 明确指定 CPU，如果你有 GPU 可改成 0
        project="ddz_runs",
        name="train",
        exist_ok=True,
        verbose=True
    )

    # YOLO 实际保存路径可能是 runs/detect/ddz_runs/train/weights/best.pt
    # 或者 ddz_runs/train/weights/best.pt（取决于 ultralytics 版本）
    possible_paths = [
        os.path.join("runs", "detect", "ddz_runs", "train", "weights", "best.pt"),
        os.path.join("ddz_runs", "train", "weights", "best.pt"),
    ]

    best_path = None
    for p in possible_paths:
        if os.path.exists(p):
            best_path = p
            break

    if best_path:
        print(f"\n{'=' * 50}")
        print(f"✅ 训练完成！最佳模型: {best_path}")
        print(f"{'=' * 50}")
        print("\n下一步: 把 best.pt 复制到项目根目录")
        print(f"  copy {best_path} ddz_yolo.pt")
        print("\n然后运行测试: python ddz_yolo.py screenshots\\hand_xxx.png")
    else:
        print("\n[警告] 未找到最佳模型，可能保存路径和预期不同")
        print("  请检查 runs/detect/ddz_runs/train/weights/ 或 ddz_runs/train/weights/")


if __name__ == "__main__":
    main()
