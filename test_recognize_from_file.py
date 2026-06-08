#!/usr/bin/env python3
"""从截图文件直接测试手牌/底牌/出牌识别，不需要开游戏窗口"""

import cv2
import json
import os
import sys
import numpy as np

# 加载 ddz.py 中的类和配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddz_yolo_recognizer import TwoStageRecognizerV3

# 加载配置
config_file = "ddz_config.json"
if not os.path.exists(config_file):
    print(f"❌ 需要 {config_file}")
    sys.exit(1)

with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

elements = config['elements']
base_w = config['base_width']
base_h = config['base_height']

# 加载识别器
print("📦 加载 YOLO 识别器...")
recognizer = TwoStageRecognizerV3()

# 支持的截图文件路径（按优先级）
candidates = [
    "raw_screenshots",
]

def extract_region(full_img, elem, scale_x, scale_y):
    """从完整截图中提取指定区域"""
    h, w = full_img.shape[:2]
    x = int(elem['x'] * scale_x)
    y = int(elem['y'] * scale_y)
    ew = int(elem.get('w', 100) * scale_x)
    eh = int(elem.get('h', 100) * scale_y)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    ew = min(ew, w - x)
    eh = min(eh, h - y)
    if ew <= 0 or eh <= 0:
        return np.array([])
    return full_img[y:y+eh, x:x+ew]


def test_file(img_path):
    """测试单张截图"""
    print(f"\n{'='*50}")
    print(f"测试: {img_path}")
    print('='*50)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图片: {img_path}")
        return
    
    h, w = img.shape[:2]
    scale_x = w / base_w
    scale_y = h / base_h
    print(f"图片尺寸: {w}x{h}, 缩放比例: {scale_x:.3f}x{scale_y:.3f}")
    
    # 手牌识别
    print("\n[手牌识别]")
    hand_img = extract_region(img, elements['my_hand'], scale_x, scale_y)
    if hand_img.size > 0:
        cv2.imwrite("test_debug_hand_region.png", hand_img)
        print(f"  手牌区尺寸: {hand_img.shape[1]}x{hand_img.shape[0]}")
        cards = recognizer.recognize(hand_img.copy())
        print(f"  结果: {cards} (共{len(cards)}张)")
    else:
        print("  ❌ 手牌区提取失败")
    
    # 底牌识别
    print("\n[底牌识别]")
    bottom_img = extract_region(img, elements['bottom_cards'], scale_x, scale_y)
    if bottom_img.size > 0:
        cv2.imwrite("test_debug_bottom_region.png", bottom_img)
        print(f"  底牌区尺寸: {bottom_img.shape[1]}x{bottom_img.shape[0]}")
        bottom_cards = recognizer.recognize_bottom(bottom_img.copy())
        names = [n for n, c in bottom_cards]
        confs = [f"{c:.0%}" for n, c in bottom_cards]
        print(f"  结果: {names} (共{len(names)}张)")
        print(f"  置信度: {confs}")
    else:
        print("  ❌ 底牌区提取失败")
    
    # 出牌识别
    print("\n[出牌识别]")
    play_img = extract_region(img, elements['play_area'], scale_x, scale_y)
    if play_img.size > 0:
        cv2.imwrite("test_debug_play_region.png", play_img)
        print(f"  出牌区尺寸: {play_img.shape[1]}x{play_img.shape[0]}")
        play_cards = recognizer.recognize_play(play_img.copy())
        print(f"  结果: {play_cards} (共{len(play_cards)}张)")
    else:
        print("  ❌ 出牌区提取失败")


def main():
    # 如果有命令行参数，直接用参数指定的文件
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            if os.path.exists(path):
                test_file(path)
            else:
                print(f"❌ 文件不存在: {path}")
        return
    
    # 自动查找 raw_screenshots 下的最新截图
    raw_dir = "raw_screenshots"
    if os.path.exists(raw_dir):
        files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.png')])
        if files:
            # 测最新的 3 张
            for f in files[-3:]:
                test_file(os.path.join(raw_dir, f))
            return
    
    # 查找当前目录下的 debug_*.png 或 region_*.png
    for pattern in ["debug_*.png", "region_*.png", "*_full.png"]:
        import glob
        files = sorted(glob.glob(pattern))
        if files:
            test_file(files[-1])
            return
    
    print("❌ 找不到测试图片。请指定图片路径：")
    print("  python test_recognize_from_file.py <图片路径>")


if __name__ == "__main__":
    main()
