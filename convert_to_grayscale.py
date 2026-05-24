#!/usr/bin/env python3
"""
把数据集所有图像转换为灰度（保持3通道，YOLO需要）
消除红/黑颜色差异，让模型只学习形状
"""

import os
import cv2
import glob


def convert_dir(directory):
    files = glob.glob(os.path.join(directory, "**", "*.png"), recursive=True)
    print(f"处理 {directory}: {len(files)} 张图")

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue

        # BGR -> 灰度 -> BGR（保持3通道）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(path, gray_bgr)

    print(f"  完成")


def main():
    print("=" * 50)
    print("数据集灰度化")
    print("=" * 50)

    convert_dir("dataset/images/train")
    convert_dir("dataset/images/val")

    # 同时把 crops_raw/ 也转灰度（方便后续分类时统一）
    convert_dir("dataset/crops_raw")

    print("\n✅ 全部转灰度完成！")
    print("下一步: python ddz_train.py 重新训练")
    print("训练完成后，ddz_yolo.py 推理时也会自动转灰度")


if __name__ == "__main__":
    main()
