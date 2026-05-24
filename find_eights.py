#!/usr/bin/env python3
"""
从 crops_raw/ 里挑出所有可能是 8 的图，放到 8_candidates/
增加诊断输出，帮助排查为什么找不到候选
"""

import os
import cv2
import numpy as np
import shutil


def load_template_8(templates_dir="templates"):
    path = os.path.join(templates_dir, "8.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 templates/8.png")
    tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise ValueError(f"无法读取 templates/8.png")
    return tmpl


def preprocess(card_img):
    h, w = card_img.shape[:2]
    digit = card_img[2:min(h, 102), 2:min(w, 62)]
    if digit.size == 0:
        return None
    gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.resize(bw, (64, 64), interpolation=cv2.INTER_AREA)


def main():
    tmpl = load_template_8()
    print(f"[模板] 8.png 尺寸: {tmpl.shape[1]}x{tmpl.shape[0]}")
    print(f"[模板] 像素均值: {np.mean(tmpl):.1f} (>127=亮底暗字, <127=暗底亮字)")
    print(f"[模板] 像素范围: {tmpl.min()}~{tmpl.max()}")

    os.makedirs("dataset/by_class/8_candidates", exist_ok=True)

    files = [f for f in os.listdir("dataset/crops_raw") if f.endswith('.png')]
    print(f"\n扫描 crops_raw/ 共 {len(files)} 张图...")

    if not files:
        print("[错误] crops_raw/ 目录为空！")
        return

    candidates = []
    scores = []  # 收集所有分数用于诊断

    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join("dataset/crops_raw", fname))
        if img is None:
            continue

        bw = preprocess(img)
        if bw is None:
            continue

        resized = cv2.resize(bw, (tmpl.shape[1], tmpl.shape[0]), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        score = cv2.minMaxLoc(res)[1]
        scores.append(score)

        if score >= 0.0:
            candidates.append((score, fname))

    if scores:
        scores_arr = np.array(scores)
        print(f"\n[诊断] 匹配分数统计:")
        print(f"  最高: {scores_arr.max():.3f}")
        print(f"  最低: {scores_arr.min():.3f}")
        print(f"  平均: {scores_arr.mean():.3f}")
        print(f"  中位数: {np.median(scores_arr):.3f}")
        print(f"  >=0.0 的数量: {(scores_arr >= 0.0).sum()}/{len(scores)}")
        print(f"  >=0.3 的数量: {(scores_arr >= 0.3).sum()}/{len(scores)}")
        print(f"  >=0.5 的数量: {(scores_arr >= 0.5).sum()}/{len(scores)}")

        # 显示分数最高的几张图的信息
        top_indices = np.argsort(scores_arr)[-5:][::-1]
        print(f"\n[诊断] 分数最高的 5 张:")
        for idx in top_indices:
            print(f"  {files[idx]}: score={scores_arr[idx]:.3f}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    print(f"\n[结果] 共 {len(candidates)} 张候选图")

    for score, fname in candidates:
        shutil.copy2(os.path.join("dataset/crops_raw", fname),
                     os.path.join("dataset/by_class/8_candidates", fname))

    if len(candidates) == 0:
        print("\n[警告] 一个候选都没找到！可能原因:")
        print("  1. templates/8.png 内容不对（只截到了花色？）")
        print("  2. 模板和裁剪图颜色相反（黑底白字 vs 白底黑字）")
        print("  3. crops_raw/ 里确实没有 8（被彻底删除了）")
        print("\n请检查 templates/8.png 的内容，确认它显示的是数字 8 而不是花色")


if __name__ == "__main__":
    main()
