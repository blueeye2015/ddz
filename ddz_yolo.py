#!/usr/bin/env python3
"""
YOLOv8 斗地主牌检测推理模块

使用方法:
    from ddz_yolo import YOLORecognizer
    recognizer = YOLORecognizer("ddz_yolo.pt")
    cards = recognizer.recognize(hand_img)
    # cards = [("K", 0.94), ("Q", 0.91), ...]  # 已按 x 坐标排序
"""

import os
import cv2
import numpy as np


class YOLORecognizer:
    """基于 YOLOv8 的扑克牌识别器"""

    # 类别 ID -> 牌面名称
    ID2NAME = {
        0: '3', 1: '4', 2: '5', 3: '6', 4: '7', 5: '8', 6: '9',
        7: '10', 8: 'J', 9: 'Q', 10: 'K', 11: 'A', 12: '2',
        13: 'SJ', 14: 'BJ'
    }

    def __init__(self, model_path: str = "ddz_yolo.pt", conf_threshold: float = 0.5):
        """
        加载 YOLO 模型

        Args:
            model_path: 训练好的模型路径，默认 ddz_yolo.pt
            conf_threshold: 置信度阈值，低于此值的检测框会被过滤
        """
        self.conf_threshold = conf_threshold
        self.model = None

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请先运行训练: python ddz_train.py\n"
                f"然后把 best.pt 复制为 {model_path}"
            )

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[YOLO] 模型加载成功: {model_path}")
        except ImportError:
            raise ImportError(
                "未安装 ultralytics，请运行: pip install ultralytics"
            )

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """转灰度（保持3通道，YOLO需要）"""
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

    def recognize(self, hand_img: np.ndarray) -> list:
        """
        识别手牌区域中的所有牌，带后处理去重
        训练时用了灰度图，推理时也要转灰度
        """
        if hand_img.size == 0 or self.model is None:
            return []

        # 转灰度，消除红/黑颜色差异
        hand_img = self.to_grayscale(hand_img)

        # YOLO 推理：降低 conf 减少漏检，降低 iou 让重叠牌更容易分开
        results = self.model(hand_img, verbose=False,
                             conf=0.25, iou=0.35)

        raw_dets = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                name = self.ID2NAME.get(cls_id, f"?{cls_id}")
                xyxy = boxes.xyxy[i].cpu().numpy()
                x = float(xyxy[0])
                w = float(xyxy[2] - xyxy[0])
                raw_dets.append((name, conf, x, w))

        if not raw_dets:
            return []

        # 按 x 排序
        raw_dets.sort(key=lambda d: d[2])

        # 后处理：对相邻框去重，只保留置信度最高的
        # 斗地主牌间距通常 > 40px，< 40px 视为同一位置的重复检测
        cards = [raw_dets[0]]
        for det in raw_dets[1:]:
            last = cards[-1]
            # 如果两个框中心距 < 40px，视为重叠，保留置信度高的
            if det[2] - last[2] < 40:
                if det[1] > last[1]:
                    cards[-1] = det
            else:
                cards.append(det)

        # 再过滤一轮：置信度低于 0.30 的丢弃
        # 实测 5 的置信度在 0.30-0.72 之间，0.30 能保留 5 同时过滤掉 8(0.37) 这种假阳性
        cards = [c for c in cards if c[1] >= 0.30]

        # 返回 (name, conf, x)
        return [(c[0], c[1], c[2]) for c in cards]

    def recognize_all(self, img: np.ndarray, elements: dict, coords) -> dict:
        """
        一次性识别手牌、底牌、出牌区域

        Args:
            img: 完整截图
            elements: 配置中的区域坐标
            coords: FixedCoords 实例，用于提取 ROI

        Returns:
            {"hand": [...], "bottom": [...], "play": [...]}
        """
        result = {}

        for key in ["my_hand", "bottom_cards", "play_area"]:
            if key not in elements:
                continue

            roi = coords.extract(img, key)
            if roi.size == 0:
                continue

            cards = self.recognize(roi)
            result[key] = cards

        return result


# 简单的命令行测试
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python ddz_yolo.py <截图路径>")
        print("示例: python ddz_yolo.py region_my_hand.png")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)

    try:
        rec = YOLORecognizer()
        cards = rec.recognize(img)
        print(f"识别到 {len(cards)} 张牌:")
        for name, conf, x in cards:
            print(f"  {name} (置信度: {conf:.3f}, x={x:.0f})")
    except Exception as e:
        print(f"[错误] {e}")
