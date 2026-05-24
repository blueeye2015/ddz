#!/usr/bin/env python3
"""
分析 dataset/by_class/5/ 的数据质量
1. 统计各花色比例（红 vs 黑）
2. 用模型对每张 5 做推理，找出模型"不认"的样本
3. 检查裁剪图是否包含相邻牌的边缘
"""

import os
import cv2
import numpy as np
import glob
from collections import Counter
from ddz_yolo import YOLORecognizer


def count_red_black():
    """统计 5/ 文件夹里红色和黑色 5 的数量（通过像素颜色判断）"""
    files = glob.glob("dataset/by_class/5/*.png")
    red_count = 0
    black_count = 0
    uncertain = 0

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue

        # 提取左上角数字区域（红色数字在HSV里H接近0或180）
        digit = img[:60, :40]
        hsv = cv2.cvtColor(digit, cv2.COLOR_BGR2HSV)

        # 红色范围
        red1 = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 80, 50]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(red1) + cv2.countNonZero(red2)

        if red_pixels > 50:
            red_count += 1
        else:
            black_count += 1

    print(f"[花色统计] 红色 5: {red_count}, 黑色 5: {black_count}")
    if red_count + black_count > 0:
        ratio = red_count / (red_count + black_count)
        print(f"[花色统计] 红色占比: {ratio:.1%}")
        if ratio < 0.3:
            print("  ⚠️ 红色 5 太少，建议补充红桃 5 和方块 5")
        elif ratio > 0.7:
            print("  ⚠️ 黑色 5 太少，建议补充黑桃 5 和梅花 5")


def analyze_predictions():
    """用模型对每张 5 做推理，找出低置信度和错分的样本"""
    rec = YOLORecognizer("ddz_yolo.pt", conf_threshold=0.01)
    files = sorted(glob.glob("dataset/by_class/5/*.png"))

    print(f"\n[推理测试] 共 {len(files)} 张，测试模型对每张 5 的识别能力...")

    pred_as_5 = []
    pred_as_other = []
    pred_none = []

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue

        fname = os.path.basename(path)
        results = rec.model(img, verbose=False, conf=0.01)

        found = False
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            best_idx = int(boxes.conf.argmax())
            conf = float(boxes.conf[best_idx])
            cls_id = int(boxes.cls[best_idx])
            name = rec.ID2NAME.get(cls_id, f"?{cls_id}")

            if name == '5':
                pred_as_5.append((fname, conf))
                found = True
            else:
                pred_as_other.append((fname, name, conf))
                found = True
            break

        if not found:
            pred_none.append(fname)

    print(f"\n[结果]")
    print(f"  正确识别为 5: {len(pred_as_5)} 张")
    if pred_as_5:
        confs = [c for _, c in pred_as_5]
        print(f"    置信度范围: {min(confs):.3f} ~ {max(confs):.3f}, 平均: {np.mean(confs):.3f}")
        low_conf = [f for f, c in pred_as_5 if c < 0.30]
        if low_conf:
            print(f"    ⚠️ 低置信度(<0.30): {len(low_conf)} 张")
            for f in low_conf[:5]:
                print(f"      - {f}")

    print(f"  错分为其他: {len(pred_as_other)} 张")
    if pred_as_other:
        counter = Counter([name for _, name, _ in pred_as_other])
        print(f"    错分分布: {dict(counter)}")
        print(f"    示例（前 5 张）:")
        for f, name, conf in pred_as_other[:5]:
            print(f"      - {f}: 被认成 {name}(conf={conf:.3f})")

    if pred_none:
        print(f"  完全检测不到: {len(pred_none)} 张")


def check_edge_contamination():
    """检查裁剪图是否包含相邻牌的边缘"""
    files = glob.glob("dataset/by_class/5/*.png")
    contaminated = 0

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        # 如果裁剪图右侧包含了明显不属于当前牌的边缘（颜色突变），可能有污染
        right_edge = img[:, -5:]  # 最右侧 5 像素
        gray = cv2.cvtColor(right_edge, cv2.COLOR_BGR2GRAY)
        std = np.std(gray)

        # 标准差大说明右侧有内容（可能是相邻牌），标准差小说明是纯色边缘
        if std > 30:
            contaminated += 1

    print(f"\n[边缘污染检查]")
    print(f"  右侧可能包含相邻牌的: {contaminated}/{len(files)} 张")
    if contaminated > len(files) * 0.3:
        print("  ⚠️ 超过 30% 的裁剪图包含相邻牌边缘，建议重新裁剪")


def main():
    print("=" * 60)
    print("dataset/by_class/5/ 数据质量分析报告")
    print("=" * 60)

    count_red_black()
    analyze_predictions()
    check_edge_contamination()

    print("\n" + "=" * 60)
    print("建议")
    print("=" * 60)
    print("1. 如果红色 5 占比 < 30%，从 crops_raw/ 找红桃 5 和方块 5 补充")
    print("2. 如果错分为 8/6 的很多，说明 5 和这些数字太像，需要更多高质量样本区分")
    print("3. 如果边缘污染严重，说明 _split_cards 的框太宽，需要调窄")


if __name__ == "__main__":
    main()
