#!/usr/bin/env python3
"""
54 类扑克牌检测推理模块
支持数字+花色识别
"""

import os
import cv2
import numpy as np


class YOLORecognizer54:
    """54 类扑克牌识别器"""

    ID2NAME = {
        0: '3S', 1: '3H', 2: '3C', 3: '3D',
        4: '4S', 5: '4H', 6: '4C', 7: '4D',
        8: '5S', 9: '5H', 10: '5C', 11: '5D',
        12: '6S', 13: '6H', 14: '6C', 15: '6D',
        16: '7S', 17: '7H', 18: '7C', 19: '7D',
        20: '8S', 21: '8H', 22: '8C', 23: '8D',
        24: '9S', 25: '9H', 26: '9C', 27: '9D',
        28: '10S', 29: '10H', 30: '10C', 31: '10D',
        32: 'JS', 33: 'JH', 34: 'JC', 35: 'JD',
        36: 'QS', 37: 'QH', 38: 'QC', 39: 'QD',
        40: 'KS', 41: 'KH', 42: 'KC', 43: 'KD',
        44: 'AS', 45: 'AH', 46: 'AC', 47: 'AD',
        48: '2S', 49: '2H', 50: '2C', 51: '2D',
        52: 'SJ', 53: 'BJ'
    }

    # 英文到中文的显示映射
    EN2CN = {
        '3S': '3♠', '3H': '3♥', '3C': '3♣', '3D': '3♦',
        '4S': '4♠', '4H': '4♥', '4C': '4♣', '4D': '4♦',
        '5S': '5♠', '5H': '5♥', '5C': '5♣', '5D': '5♦',
        '6S': '6♠', '6H': '6♥', '6C': '6♣', '6D': '6♦',
        '7S': '7♠', '7H': '7♥', '7C': '7♣', '7D': '7♦',
        '8S': '8♠', '8H': '8♥', '8C': '8♣', '8D': '8♦',
        '9S': '9♠', '9H': '9♥', '9C': '9♣', '9D': '9♦',
        '10S': '10♠', '10H': '10♥', '10C': '10♣', '10D': '10♦',
        'JS': 'J♠', 'JH': 'J♥', 'JC': 'J♣', 'JD': 'J♦',
        'QS': 'Q♠', 'QH': 'Q♥', 'QC': 'Q♣', 'QD': 'Q♦',
        'KS': 'K♠', 'KH': 'K♥', 'KC': 'K♣', 'KD': 'K♦',
        'AS': 'A♠', 'AH': 'A♥', 'AC': 'A♣', 'AD': 'A♦',
        '2S': '2♠', '2H': '2♥', '2C': '2♣', '2D': '2♦',
        'SJ': '小王', 'BJ': '大王'
    }

    def __init__(self, model_path: str = "ddz_yolo_54.pt", conf_threshold: float = 0.30):
        self.conf_threshold = conf_threshold
        self.model = None

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[YOLO-54] 模型加载成功: {model_path}")
        except ImportError:
            raise ImportError("pip install ultralytics")

    @staticmethod
    def to_grayscale(img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

    def recognize(self, hand_img: np.ndarray) -> list:
        if hand_img.size == 0 or self.model is None:
            return []

        hand_img = self.to_grayscale(hand_img)
        results = self.model(hand_img, verbose=False, conf=0.25, iou=0.35)

        raw_dets = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                name = self.ID2NAME.get(cls_id, f"?{cls_id}")
                x = float(boxes.xyxy[i][0])
                raw_dets.append((name, conf, x))

        if not raw_dets:
            return []

        raw_dets.sort(key=lambda d: d[2])

        # 去重
        cards = [raw_dets[0]]
        for det in raw_dets[1:]:
            last = cards[-1]
            if det[2] - last[2] < 40:
                if det[1] > last[1]:
                    cards[-1] = det
            else:
                cards.append(det)

        cards = [c for c in cards if c[1] >= self.conf_threshold]
        return [(c[0], c[1], c[2]) for c in cards]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python ddz_yolo_54.py <截图路径>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("无法读取图片")
        sys.exit(1)

    try:
        rec = YOLORecognizer54()
        cards = rec.recognize(img)
        print(f"识别到 {len(cards)} 张牌:")
        for name, conf, x in cards:
            cn = rec.EN2CN.get(name, name)
            print(f"  {cn} (conf={conf:.3f}, x={x:.0f})")
    except Exception as e:
        print(f"[错误] {e}")
