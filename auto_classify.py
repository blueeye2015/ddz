#!/usr/bin/env python3
"""
自动预分类脚本
用现有的模板匹配对 crops_raw/ 下的裁剪图做初步分类，
减少人工分类工作量。

高置信度(score>0.55) → 自动放入对应类别
低置信度或不确定   → 放入 review/ 文件夹待人工检查
明显错误的裁剪    → 直接丢弃
"""

import os
import cv2
import numpy as np
import shutil


def load_templates(templates_dir="templates"):
    """加载现有模板"""
    templates = {}
    if not os.path.exists(templates_dir):
        return templates

    for f in sorted(os.listdir(templates_dir)):
        if f.endswith('.png'):
            name = os.path.splitext(f)[0]
            tmpl = cv2.imread(os.path.join(templates_dir, f), cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                templates[name] = tmpl
    return templates


def extract_digit(card_img):
    """简化版数字提取（固定区域）"""
    h, w = card_img.shape[:2]
    return card_img[2:min(h, 102), 2:min(w, 62)]


def preprocess_for_match(card_img):
    """预处理为匹配用的二值图"""
    digit = extract_digit(card_img)
    if digit.size == 0:
        return None
    gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.resize(bw, (64, 64), interpolation=cv2.INTER_AREA)
    return bw


def match_card(card_img, templates):
    """模板匹配，返回 (best_name, best_score)"""
    if not templates:
        return None, 0.0

    bw = preprocess_for_match(card_img)
    if bw is None:
        return None, 0.0

    best_name, best_score = None, -1.0
    for name, tmpl in templates.items():
        resized = cv2.resize(bw, (tmpl.shape[1], tmpl.shape[0]), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score, best_name = max_val, name

    return best_name, best_score


def is_valid_crop(img):
    """过滤明显错误的裁剪"""
    h, w = img.shape[:2]
    # 尺寸太小（可能是噪声碎片）
    if h < 60 or w < 35:
        return False, "too_small"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)

    # 几乎全白（空白区域）
    if mean > 245:
        return False, "too_white"

    # 几乎全黑
    if mean < 30:
        return False, "too_black"

    return True, "ok"


def main():
    templates = load_templates()
    print(f"[加载] {len(templates)} 个模板: {list(templates.keys())}")

    if not templates:
        print("[错误] 未找到模板，请先运行: python ddz.py --capture-templates")
        return

    os.makedirs("dataset/by_class/review", exist_ok=True)

    raw_files = [f for f in os.listdir("dataset/crops_raw") if f.endswith('.png')]
    print(f"[处理] 共 {len(raw_files)} 张裁剪图\n")

    auto_count = 0
    review_count = 0
    skip_count = 0

    for i, fname in enumerate(raw_files, 1):
        path = os.path.join("dataset/crops_raw", fname)
        img = cv2.imread(path)
        if img is None:
            continue

        valid, reason = is_valid_crop(img)
        if not valid:
            skip_count += 1
            continue

        name, score = match_card(img, templates)

        if name and score > 0.55:
            # 高置信度，自动分类
            target_dir = f"dataset/by_class/{name}"
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(target_dir, fname))
            auto_count += 1
        else:
            # 低置信度，放入 review
            shutil.copy2(path, os.path.join("dataset/by_class/review", fname))
            review_count += 1

        if i % 200 == 0:
            print(f"  进度: {i}/{len(raw_files)}  自动:{auto_count}  待审:{review_count}  跳过:{skip_count}")

    print(f"\n{'=' * 50}")
    print(f"✅ 自动预分类完成")
    print(f"   自动分类: {auto_count} 张 (已按模板匹配放入对应文件夹)")
    print(f"   待人工检查: {review_count} 张 (在 dataset/by_class/review/)")
    print(f"   过滤丢弃: {skip_count} 张 (空白/碎片)")
    print(f"{'=' * 50}")

    if review_count > 0:
        print(f"\n下一步:")
        print(f"  1. 打开 dataset/by_class/review/")
        print(f"  2. 逐张查看，拖入正确的类别文件夹")
        print(f"  3. 分完后运行: python generate_yolo_labels.py")
    else:
        print(f"\n全部自动分类完成，直接运行: python generate_yolo_labels.py")


if __name__ == "__main__":
    main()
