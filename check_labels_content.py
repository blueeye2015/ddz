#!/usr/bin/env python3
"""随机抽查标注文件内容，看坐标是否正常"""

import os
import random
import cv2


def main():
    lbl_dir = "dataset/labels_54/train"
    img_dir = "dataset/images_54/train"

    lbl_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    sample = random.sample(lbl_files, min(5, len(lbl_files)))

    print("=" * 60)
    print("标注文件抽查")
    print("=" * 60)

    for lbl_name in sample:
        base = os.path.splitext(lbl_name)[0]
        lbl_path = os.path.join(lbl_dir, lbl_name)

        # 找对应图片
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(img_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break

        # 读取图片尺寸
        img_w, img_h = 0, 0
        if img_path:
            img = cv2.imread(img_path)
            if img is not None:
                img_h, img_w = img.shape[:2]

        # 读取标注
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

        print(f"\n{lbl_name} (图: {img_w}x{img_h}):")
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"  ❌ 格式错误: {line}")
                continue

            cid, cx, cy, nw, nh = parts
            cx, cy, nw, nh = float(cx), float(cy), float(nw), float(nh)

            # 反算像素坐标
            px = cx * img_w
            py = cy * img_h
            pw = nw * img_w
            ph = nh * img_h

            flag = ""
            if cx == 0 and cy == 0 and nw == 0 and nh == 0:
                flag = " ❌ 全零！"
            elif cx > 1 or cy > 1 or nw > 1 or nh > 1:
                flag = " ❌ 越界！"
            elif pw < 10 or ph < 10:
                flag = " ⚠️ 框太小"

            print(f"  class={cid}, cx={cx:.4f}, cy={cy:.4f}, w={nw:.4f}, h={nh:.4f}"
                  f" → 像素: ({px:.0f}, {py:.0f}) {pw:.0f}x{ph:.0f}{flag}")

    print("\n" + "=" * 60)
    print("判断标准:")
    print("  • 全零 → 标注生成有 bug，坐标没写进去")
    print("  • 越界 → 原图尺寸和 img_w/img_h 不匹配")
    print("  • 框太小(<10px) → _split_cards 框宽异常")
    print("  • 正常 → 问题在 YOLO 训练配置，继续排查")


if __name__ == "__main__":
    main()
