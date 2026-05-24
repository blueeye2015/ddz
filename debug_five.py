#!/usr/bin/env python3
"""
诊断 5 为什么检测不到
1. 用极低阈值(conf=0.01)跑整行手牌，看 5 的置信度是多少
2. 对单张已分类的 5 做推理，看模型对 isolate 的 5 识别率如何
"""

import os
import cv2
import glob
from ddz_yolo import YOLORecognizer


def test_full_hand(img_path):
    """测试整行手牌，极低阈值"""
    print("=" * 50)
    print(f"测试整行手牌: {img_path}")
    print("=" * 50)

    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片")
        return

    rec = YOLORecognizer("ddz_yolo.pt", conf_threshold=0.01)

    # 临时把模型阈值降到 0.01
    results = rec.model(img, verbose=False, conf=0.01, iou=0.35)

    all_dets = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            name = rec.ID2NAME.get(cls_id, f"?{cls_id}")
            x = float(boxes.xyxy[i][0])
            all_dets.append((name, conf, x))

    all_dets.sort(key=lambda d: d[2])

    print(f"共检测到 {len(all_dets)} 个框（阈值 0.01）:")
    for name, conf, x in all_dets:
        marker = " <-- 5!" if name == '5' else ""
        print(f"  {name} (conf={conf:.3f}, x={x:.0f}){marker}")

    fives = [d for d in all_dets if d[0] == '5']
    print(f"\n其中 5 的数量: {len(fives)}")
    if fives:
        print(f"5 的置信度范围: {min(f[1] for f in fives):.3f} ~ {max(f[1] for f in fives):.3f}")
    else:
        print("❌ 即使阈值降到 0.01，仍然没有检测到 5")
        print("   说明模型完全没学到 5 的特征，需要补强训练数据")


def test_single_fives():
    """测试单张已分类的 5"""
    print("\n" + "=" * 50)
    print("测试单张已分类的 5（isolate 环境）")
    print("=" * 50)

    rec = YOLORecognizer("ddz_yolo.pt", conf_threshold=0.01)

    five_files = glob.glob("dataset/by_class/5/*.png")
    if not five_files:
        print("dataset/by_class/5/ 为空，没有单张 5 可测")
        return

    print(f"找到 {len(five_files)} 张已分类的 5")

    for i, path in enumerate(five_files[:5]):  # 只测前 5 张
        img = cv2.imread(path)
        if img is None:
            continue

        results = rec.model(img, verbose=False, conf=0.01)
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                print(f"  {os.path.basename(path)}: 未检测到任何牌")
                continue

            conf = float(boxes.conf[0])
            cls_id = int(boxes.cls[0])
            name = rec.ID2NAME.get(cls_id, f"?{cls_id}")
            print(f"  {os.path.basename(path)}: 预测为 {name} (conf={conf:.3f})")


if __name__ == "__main__":
    import sys

    # 测试单张 5
    test_single_fives()

    # 测试整行手牌
    if len(sys.argv) > 1:
        test_full_hand(sys.argv[1])
    else:
        # 自动找 screenshots 下的第一张图
        screenshots = glob.glob("screenshots/*.png")
        if screenshots:
            test_full_hand(screenshots[0])
        else:
            print("\n未找到 screenshots/ 下的图，请传参:")
            print("  python debug_five.py screenshots\\hand_xxx.png")
