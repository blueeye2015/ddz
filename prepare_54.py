#!/usr/bin/env python3
"""
54 类扑克牌数据集准备
基于现有的 dataset/by_class/（15类数字），自动按颜色分组

流程：
1. 读取现有数字分类文件夹里的裁剪图
2. 颜色检测判断红/黑（基于花色区域）
3. 自动放入 dataset/by_class_54/数字_红色/ 或 数字_黑色/
4. 用户只需区分形状：
   - 红色内：心形→红心，菱形→方块
   - 黑色内：桃形→黑桃，三叶草→梅花
"""

import os
import cv2
import numpy as np
import shutil
import glob


# 54 个类别文件夹
NUMBERS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
SUITS = ['红心', '方块', '黑桃', '梅花']


def detect_color(img):
    """检测裁剪图中花色的颜色（红/黑）"""
    h, w = img.shape[:2]
    if h < 80 or w < 30:
        return 'unknown'

    # 提取花色区域：数字下方，牌面中央偏左
    # 数字通常在 y=0:40，花色在 y=40:90，x=5:w-5
    suit_y1 = min(h, 45)
    suit_y2 = min(h, 95)
    suit_x1 = min(w, 5)
    suit_x2 = max(suit_x1 + 1, w - 5)

    suit_region = img[suit_y1:suit_y2, suit_x1:suit_x2]
    if suit_region.size == 0:
        return 'unknown'

    hsv = cv2.cvtColor(suit_region, cv2.COLOR_BGR2HSV)

    # 红色范围（包含红桃和方块）
    red1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 60, 50]), np.array([180, 255, 255]))
    red_pixels = cv2.countNonZero(red1) + cv2.countNonZero(red2)

    # 如果红色像素足够多，判定为红色花色
    total_pixels = suit_region.shape[0] * suit_region.shape[1]
    red_ratio = red_pixels / total_pixels

    return 'red' if red_ratio > 0.05 else 'black'


def main():
    base = "dataset/by_class_54"
    os.makedirs(base, exist_ok=True)

    # 创建 54 个目标文件夹
    for num in NUMBERS:
        for suit in SUITS:
            os.makedirs(os.path.join(base, f"{num}_{suit}"), exist_ok=True)
        # 创建临时分组文件夹
        os.makedirs(os.path.join(base, f"{num}_红色"), exist_ok=True)
        os.makedirs(os.path.join(base, f"{num}_黑色"), exist_ok=True)

    os.makedirs(os.path.join(base, "SJ"), exist_ok=True)
    os.makedirs(os.path.join(base, "BJ"), exist_ok=True)

    print("=" * 60)
    print("54 类数据集准备")
    print("=" * 60)

    # 处理每个数字类别
    for num in NUMBERS:
        src_dir = f"dataset/by_class/{num}"
        if not os.path.exists(src_dir):
            continue

        files = glob.glob(os.path.join(src_dir, "*.png"))
        if not files:
            continue

        red_count = 0
        black_count = 0

        for path in files:
            img = cv2.imread(path)
            if img is None:
                continue

            color = detect_color(img)
            fname = os.path.basename(path)

            if color == 'red':
                dst = os.path.join(base, f"{num}_红色", fname)
                shutil.copy2(path, dst)
                red_count += 1
            elif color == 'black':
                dst = os.path.join(base, f"{num}_黑色", fname)
                shutil.copy2(path, dst)
                black_count += 1
            else:
                # 颜色不明，两边都放一份
                shutil.copy2(path, os.path.join(base, f"{num}_红色", fname))
                shutil.copy2(path, os.path.join(base, f"{num}_黑色", fname))

        print(f"{num}: 红色 {red_count} 张, 黑色 {black_count} 张 → 待确认")

    # 处理 SJ 和 BJ（没有花色）
    for special in ['SJ', 'BJ']:
        src_dir = f"dataset/by_class/{special}"
        if os.path.exists(src_dir):
            for path in glob.glob(os.path.join(src_dir, "*.png")):
                shutil.copy2(path, os.path.join(base, special, os.path.basename(path)))
            print(f"{special}: 直接复制完成")

    print(f"\n{'=' * 60}")
    print("✅ 颜色分组完成！")
    print(f"{'=' * 60}")
    print("\n下一步：手动区分形状")
    print("打开 dataset/by_class_54/ 下的文件夹：")
    print("  • *_红色/ 里的图 → 心形拖入 *_红心/，菱形拖入 *_方块/")
    print("  • *_黑色/ 里的图 → 桃形拖入 *_黑桃/，三叶草拖入 *_梅花/")
    print("\n用缩略图模式查看，一眼就能看出形状，批量拖动很快。")
    print("分完后运行: python generate_yolo_labels_54.py")


if __name__ == "__main__":
    main()
