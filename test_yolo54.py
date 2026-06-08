#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新训练的 54 类 YOLO 模型 (best.pt) 在游戏截图上的识别效果
用法:
    python test_yolo54.py
"""

import os
import sys
import cv2
import numpy as np
import time
import json
import ctypes
from typing import Optional, Tuple, Dict, List
from collections import defaultdict

try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("警告: win32gui 不可用，无法截图")

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("错误: ultralytics 未安装")
    sys.exit(1)


# ============== 配置 ==============

BASE_WIDTH = 2071
BASE_HEIGHT = 1231

# 默认坐标（会被 ddz_config.json 覆盖）
ELEMENTS = {
    'my_hand': {'x': 100, 'y': 980, 'w': 1870, 'h': 180},
    'bottom_cards': {'x': 50, 'y': 50, 'w': 500, 'h': 130},
    'play_area': {'x': 400, 'y': 280, 'w': 1250, 'h': 600},
}

GAME_REGION = {'x_offset': 0, 'y_offset': 0, 'width': 0, 'height': 0}

# 54 类牌名
CARD_NAMES = [
    '10C','10D','10H','10S',
    '2C','2D','2H','2S',
    '3C','3D','3H','3S',
    '4C','4D','4H','4S',
    '5C','5D','5H','5S',
    '6C','6D','6H','6S',
    '7C','7D','7H','7S',
    '8C','8D','8H','8S',
    '9C','9D','9H','9S',
    'AC','AD','AH','AS',
    'JC','JD','JH','JS',
    'KC','KD','KH','KS',
    'QC','QD','QH','QS',
    'SJ','BJ'
]

# 斗地主排序权重（用于输出排序）
DDZ_RANK = {
    '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15,
    'SJ': 16, 'BJ': 17
}


def load_config():
    """加载 ddz_config.json 覆盖默认坐标"""
    global ELEMENTS, BASE_WIDTH, BASE_HEIGHT, GAME_REGION
    if os.path.exists("ddz_config.json"):
        with open("ddz_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        BASE_WIDTH = cfg.get("base_width", BASE_WIDTH)
        BASE_HEIGHT = cfg.get("base_height", BASE_HEIGHT)
        GAME_REGION = cfg.get("game_region", GAME_REGION)
        if "elements" in cfg:
            for k, v in cfg["elements"].items():
                if k in ELEMENTS:
                    ELEMENTS[k].update(v)
                else:
                    ELEMENTS[k] = v
        print(f"[配置] 已加载 ddz_config.json")


class WindowFinder:
    def __init__(self):
        self.hwnd = None
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    def find(self, title="腾讯欢乐斗地主") -> Optional[int]:
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                if title in t:
                    extra.append(hwnd)
            return True
        handles = []
        win32gui.EnumWindows(callback, handles)
        if handles:
            self.hwnd = handles[0]
            placement = win32gui.GetWindowPlacement(self.hwnd)
            if placement[1] == win32con.SW_SHOWMINIMIZED:
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.2)
            return self.hwnd
        return None

    def get_client_rect(self) -> Dict[str, int]:
        cr = win32gui.GetClientRect(self.hwnd)
        left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        right, bottom = win32gui.ClientToScreen(self.hwnd, (cr[2], cr[3]))
        return {'x': left, 'y': top, 'width': right - left, 'height': bottom - top}


class Capture:
    def __init__(self):
        self.mss = mss.mss()

    def capture(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        monitor = {
            "left": bbox[0], "top": bbox[1],
            "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]
        }
        img = np.array(self.mss.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


class FixedCoords:
    def __init__(self, client_rect: Dict[str, int], actual_size: Tuple[int, int] = None):
        self.client = client_rect
        self.game_x = GAME_REGION.get('x_offset', 0)
        self.game_y = GAME_REGION.get('y_offset', 0)
        if actual_size:
            actual_w, actual_h = actual_size
        else:
            actual_w = client_rect['width']
            actual_h = client_rect['height']
        self.game_w = GAME_REGION.get('width', 0) or (actual_w - self.game_x)
        self.game_h = GAME_REGION.get('height', 0) or (actual_h - self.game_y)
        self.scale_x = self.game_w / BASE_WIDTH
        self.scale_y = self.game_h / BASE_HEIGHT
        print(f"[坐标] 客户区: {client_rect['width']}x{client_rect['height']}")
        print(f"[坐标] 截图实际: {actual_w}x{actual_h}")
        print(f"[坐标] 游戏区域: {self.game_w}x{self.game_h}")
        print(f"[坐标] 缩放: X={self.scale_x:.3f}, Y={self.scale_y:.3f}")

    def get_rel(self, name: str) -> Dict[str, int]:
        if name not in ELEMENTS:
            raise ValueError(f"未知元素: {name}")
        elem = ELEMENTS[name]
        result = {}
        for key in ['x', 'count_x']:
            if key in elem:
                result[key] = int(self.game_x + elem[key] * self.scale_x)
        for key in ['y', 'count_y']:
            if key in elem:
                result[key] = int(self.game_y + elem[key] * self.scale_y)
        if 'w' in elem:
            result['w'] = int(elem['w'] * self.scale_x)
        if 'h' in elem:
            result['h'] = int(elem['h'] * self.scale_y)
        return result

    def extract(self, screenshot: np.ndarray, name: str) -> np.ndarray:
        rect = self.get_rel(name)
        x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
        h_s, w_s = screenshot.shape[:2]
        x = max(0, min(x, w_s - 1))
        y = max(0, min(y, h_s - 1))
        w = min(w, w_s - x)
        h = min(h, h_s - y)
        if w <= 0 or h <= 0:
            return np.array([])
        return screenshot[y:y+h, x:x+w]


def ddz_sort_key(card_name: str) -> int:
    """斗地主牌面大小排序"""
    if card_name in DDZ_RANK:
        return DDZ_RANK[card_name]
    # 标准牌如 "3C", "AH"
    rank = card_name[:-1]
    return DDZ_RANK.get(rank, 99)


def detect_region(model, image: np.ndarray, region_name: str, conf: float = 0.25) -> List[Dict]:
    """
    对单个区域做 YOLO 检测，返回检测到的牌列表
    """
    if image.size == 0:
        return []

    results = model.predict(image, conf=conf, verbose=False)
    detections = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            cx = (xyxy[0] + xyxy[2]) / 2
            card_name = CARD_NAMES[cls_id] if cls_id < len(CARD_NAMES) else f"cls_{cls_id}"
            detections.append({
                'name': card_name,
                'conf': conf_val,
                'cx': cx,
                'x1': int(xyxy[0]), 'y1': int(xyxy[1]),
                'x2': int(xyxy[2]), 'y2': int(xyxy[3])
            })

    # 按中心 x 坐标从左到右排序
    detections.sort(key=lambda d: d['cx'])
    return detections


def visualize(image: np.ndarray, detections: List[Dict], title: str = "") -> np.ndarray:
    """在图像上画框和标签"""
    vis = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        label = f"{d['name']} {d['conf']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if title:
        cv2.putText(vis, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def main():
    print("=" * 50)
    print("54 类 YOLO 模型测试 (best.pt)")
    print("=" * 50)

    # 加载配置
    load_config()

    # 加载模型
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型 {model_path}")
        sys.exit(1)

    print(f"\n加载模型: {model_path}")
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    device_str = "GPU" if device != "cpu" else "CPU"
    print(f"设备: {device_str} ({model.device})")

    # 找窗口
    if not WIN32_AVAILABLE:
        print("错误: win32gui 不可用")
        sys.exit(1)

    finder = WindowFinder()
    hwnd = finder.find("腾讯欢乐斗地主")
    if not hwnd:
        print("错误: 找不到游戏窗口")
        sys.exit(1)

    # 截图
    rect = finder.get_client_rect()
    cap = Capture()
    screenshot = cap.capture((rect['x'], rect['y'],
                            rect['x'] + rect['width'],
                            rect['y'] + rect['height']))
    print(f"\n截图尺寸: {screenshot.shape[1]}x{screenshot.shape[0]}")

    # 初始化坐标系统
    coords = FixedCoords(rect, (screenshot.shape[1], screenshot.shape[0]))

    # 创建输出目录
    os.makedirs("test_output", exist_ok=True)
    ts = time.strftime("%m%d_%H%M%S")

    # 测试三个区域
    regions = ['my_hand', 'play_area', 'bottom_cards']
    all_results = {}

    for region in regions:
        print(f"\n--- {region} ---")
        roi = coords.extract(screenshot, region)
        if roi.size == 0:
            print("  ROI 为空")
            continue

        # 检测（conf 设低一些，因为小数据集模型置信度可能不高）
        dets = detect_region(model, roi, region, conf=0.15)

        # 去重：同一 x 位置附近的多个框只保留置信度最高的
        filtered = []
        used_x = []
        for d in dets:
            dup = False
            for ux in used_x:
                if abs(d['cx'] - ux) < 30:  # 30px 内认为是同一张牌
                    dup = True
                    break
            if not dup:
                filtered.append(d)
                used_x.append(d['cx'])

        cards = [d['name'] for d in filtered]
        cards_sorted = sorted(cards, key=ddz_sort_key)
        print(f"  检测到: {cards} (排序后: {cards_sorted})")
        for d in filtered:
            print(f"    {d['name']} conf={d['conf']:.3f} cx={d['cx']:.0f}")

        all_results[region] = {
            'cards': cards,
            'cards_sorted': cards_sorted,
            'detections': filtered
        }

        # 保存可视化结果
        vis = visualize(roi, filtered, region)
        out_path = f"test_output/{region}_{ts}.png"
        cv2.imwrite(out_path, vis)
        print(f"  可视化保存: {out_path}")

    # 保存完整截图（带区域框）
    full_vis = screenshot.copy()
    for region in regions:
        r = coords.get_rel(region)
        cv2.rectangle(full_vis, (r['x'], r['y']),
                      (r['x'] + r['w'], r['y'] + r['h']), (255, 0, 0), 2)
        cards = all_results.get(region, {}).get('cards_sorted', [])
        text = f"{region}: {' '.join(cards)}"
        cv2.putText(full_vis, text, (r['x'], r['y'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    full_path = f"test_output/full_{ts}.png"
    cv2.imwrite(full_path, full_vis)
    print(f"\n完整截图保存: {full_path}")

    print("\n测试完成!")


if __name__ == "__main__":
    main()
