#!/usr/bin/env python3
"""
两阶段 YOLO + MobileNetV3 卡牌识别器
用法：
    from ddz_yolo_recognizer import TwoStageRecognizerV3
    rec = TwoStageRecognizerV3()
    cards = rec.recognize(hand_img)  # 返回 ['3','4','5',...]
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO


class TwoStageRecognizerV3:
    """
    两阶段识别器 V3
    - 阶段1: YOLO 检测牌的位置（1类: card）
    - 阶段2: MobileNetV3 分类数字（15类）
    - 优化: 框微调 + 边缘镜像 + 窄框扩展
    """

    # ImageFolder 按字母排序的 15 个类别
    CLASS_NAMES = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'BJ', 'J', 'K', 'Q', 'SJ']
    MIN_CARD_WIDTH = 70  # YOLO 密集场景下框偏窄，强制扩展到 70px

    def __init__(self,
                 detect_model_path='ddz_detect_best.pt',
                 classify_model_path='card_classifier_best.pth',
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ========== 阶段1: YOLO 定位 ==========
        self.detect_model = YOLO(detect_model_path)

        # ========== 阶段2: MobileNetV3 分类 ==========
        self.classifier = models.mobilenet_v3_small(pretrained=False)
        in_f = self.classifier.classifier[3].in_features
        self.classifier.classifier[3] = nn.Linear(in_f, len(self.CLASS_NAMES))
        self.classifier.load_state_dict(
            torch.load(classify_model_path, map_location=self.device)
        )
        self.classifier.to(self.device)
        self.classifier.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _classify(self, card_img: np.ndarray):
        if card_img.size == 0:
            return None, 0.0
        rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.classifier(tensor)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
        return self.CLASS_NAMES[pred.item()], conf.item()

    def _classify_shift(self, img, x1, y1, x2, y2, shifts=(-10, -5, 0, 5, 10)):
        """框左右微调，取最高置信度"""
        h, w = img.shape[:2]
        best = (None, 0.0, 0)
        for dx in shifts:
            nx1, nx2 = max(0, x1 + dx), min(w, x2 + dx)
            name, conf = self._classify(img[y1:y2, nx1:nx2])
            if name and conf > best[1]:
                best = (name, conf, dx)
        return best

    def _classify_mirror(self, img, x1, y1, x2, y2):
        """边缘框镜像填充后分类"""
        h, w = img.shape[:2]
        card = img[y1:y2, x1:x2]
        if card.size == 0:
            return None, 0.0
        # 左边填充
        if x1 < 20:
            card = cv2.copyMakeBorder(card, 0, 0, 20 - x1, 0, cv2.BORDER_REFLECT_101)
        # 右边填充
        if x2 > w - 20:
            card = cv2.copyMakeBorder(card, 0, 0, 0, x2 - (w - 20), cv2.BORDER_REFLECT_101)
        return self._classify(card)

    def recognize(self, hand_img: np.ndarray, min_conf=0.70):
        """
        识别整行手牌/底牌/出牌
        返回: list[str] 按从左到右排序
        """
        h, w = hand_img.shape[:2]

        # ---- 阶段1: YOLO 检测 ----
        results = self.detect_model(
            hand_img,
            conf=0.10,
            iou=0.15,
            imgsz=1280,
            verbose=False
        )

        raw = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                yc = float(r.boxes.conf[i].cpu().numpy())
                # 过滤极端边缘垃圾框，但保留截断的牌
                if x1 < 0 or x2 > w + 10 or yc < 0.05:
                    continue
                raw.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'cx': (x1 + x2) // 2, 'yc': yc})

        raw.sort(key=lambda c: c['cx'])

        # ---- 阶段2: 分类 ----
        cards = []
        for c in raw:
            # 优化1: 强制扩展窄框（密集场景 YOLO 框宽 57-67px，标准 78px）
            if c['x2'] - c['x1'] < self.MIN_CARD_WIDTH:
                c['x1'] = max(0, c['x2'] - self.MIN_CARD_WIDTH)

            edge = (c['x1'] < 25) or (c['x2'] > w - 25)

            if edge:
                # 优化2: 边缘框先做镜像填充分类
                name, conf = self._classify_mirror(
                    hand_img, c['x1'], c['y1'], c['x2'], c['y2']
                )
                # 优化3: 再做框微调
                name2, conf2, _ = self._classify_shift(
                    hand_img, c['x1'], c['y1'], c['x2'], c['y2']
                )
                if conf2 > conf:
                    name, conf = name2, conf2
            else:
                card_img = hand_img[c['y1']:c['y2'], c['x1']:c['x2']]
                name, conf = self._classify(card_img)
                # 优化3: 置信度低则微调
                if conf < 0.90:
                    name2, conf2, _ = self._classify_shift(
                        hand_img, c['x1'], c['y1'], c['x2'], c['y2']
                    )
                    if conf2 > conf:
                        name, conf = name2, conf2

            cards.append({'name': name, 'conf': conf, 'x': c['cx']})

        # ---- 去重: 同一位置保留置信度最高的 ----
        deduped = []
        for c in cards:
            dup = False
            for d in deduped:
                if abs(c['x'] - d['x']) < 15:
                    dup = True
                    if c['conf'] > d['conf']:
                        d.update(c)
                    break
            if not dup:
                deduped.append(c)

        deduped.sort(key=lambda c: c['x'])

        # 过滤低置信度，但边缘牌适当放宽
        result = []
        for c in deduped:
            thr = 0.60 if (c['x'] < 30 or c['x'] > w - 30) else min_conf
            if c['conf'] >= thr:
                result.append(c['name'])
        return result
