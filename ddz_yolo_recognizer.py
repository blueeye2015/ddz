#!/usr/bin/env python3
"""
两阶段 YOLO + MobileNetV3 卡牌识别器
支持：手牌区 / 出牌区 / 底牌区
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
from scipy.signal import find_peaks


class TwoStageRecognizerV3:
    CLASS_NAMES = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'BJ', 'J', 'K', 'Q', 'SJ']
    MIN_CARD_WIDTH = 70

    def __init__(self,
                 detect_model_path='ddz_detect_best.pt',
                 classify_model_path='card_classifier_best.pth',
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detect_model = YOLO(detect_model_path)

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

    def _classify(self, card_img):
        if card_img.size == 0:
            return None, 0.0
        rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.classifier(tensor)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
        return self.CLASS_NAMES[pred.item()], conf.item()

    def _classify_shift(self, img, x1, y1, x2, y2, shifts=(-12, -6, 0, 6, 12)):
        h, w = img.shape[:2]
        best = (None, 0.0, 0)
        for dx in shifts:
            nx1, nx2 = max(0, x1 + dx), min(w, x2 + dx)
            name, conf = self._classify(img[y1:y2, nx1:nx2])
            if name and conf > best[1]:
                best = (name, conf, dx)
        return best

    def _classify_mirror(self, img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        card = img[y1:y2, x1:x2]
        if card.size == 0:
            return None, 0.0
        if x1 < 20:
            card = cv2.copyMakeBorder(card, 0, 0, 20 - x1, 0, cv2.BORDER_REFLECT_101)
        if x2 > w - 20:
            card = cv2.copyMakeBorder(card, 0, 0, 0, x2 - (w - 20), cv2.BORDER_REFLECT_101)
        return self._classify(card)

    def recognize(self, hand_img, min_conf=0.70):
        """识别手牌区（横排密集牌）"""
        h, w = hand_img.shape[:2]
        results = self.detect_model(hand_img, conf=0.10, iou=0.15, imgsz=1280, verbose=False)

        raw = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                yc = float(r.boxes.conf[i].cpu().numpy())
                if x1 < 0 or x2 > w + 10 or yc < 0.05:
                    continue
                raw.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'cx': (x1 + x2) // 2, 'yc': yc})

        raw.sort(key=lambda c: c['cx'])
        return self._classify_boxes(hand_img, raw, min_conf)

    def recognize_play(self, play_img, min_conf=0.70):
        """
        识别出牌区（多组牌分散在桌面）
        过滤假阳性：按钮、文字、边框等
        """
        h, w = play_img.shape[:2]
        results = self.detect_model(play_img, conf=0.10, iou=0.20, imgsz=1280, verbose=False)

        raw = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())
                yc = float(r.boxes.conf[i].cpu().numpy())
                bw, bh = x2 - x1, y2 - y1
                ratio = bw / bh if bh > 0 else 0
                # 核心过滤：牌的宽高比约0.37，高度>80，过滤按钮/文字/边框
                if not (0.25 <= ratio <= 0.55 and bh >= 80 and yc >= 0.10):
                    continue
                raw.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'cx': (x1 + x2) // 2, 'yc': yc})

        raw.sort(key=lambda c: c['cx'])
        return self._classify_boxes(play_img, raw, min_conf)

    def recognize_bottom(self, bottom_img, min_conf=0.60):
        """
        识别底牌区（小牌密集排列）
        策略：放大3倍 -> 垂直投影分割 -> 跳过文字标签 -> 分类
        返回: list[str] 或 list[(str, float)] 带置信度
        """
        h, w = bottom_img.shape[:2]
        scale = 3
        img_big = cv2.resize(bottom_img, (w * scale, h * scale))
        hb, wb = img_big.shape[:2]

        # 找第一个缝隙（跳过左边的"底牌"文字标签）
        gray = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        proj = np.sum(binary == 255, axis=0)
        creases = np.max(proj) - proj
        peaks, _ = find_peaks(creases, distance=100, prominence=20)
        start = int(peaks[0]) if len(peaks) > 0 else 0

        # 在有效区域内找所有缝隙
        sub = img_big[:, start:]
        gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        _, bin_sub = cv2.threshold(gray_sub, 220, 255, cv2.THRESH_BINARY)
        proj_sub = np.sum(bin_sub == 255, axis=0)
        creases_sub = np.max(proj_sub) - proj_sub
        peaks_sub, _ = find_peaks(creases_sub, distance=80, prominence=15)
        splits = [0] + sorted(peaks_sub) + [sub.shape[1]]

        results = []
        for i in range(len(splits) - 1):
            x1 = start + int(splits[i])
            x2 = start + int(splits[i + 1])
            cw = (x2 - x1) // scale
            # 过滤文字标签(太宽)和噪声(太窄)
            if cw < 25 or cw > 70:
                continue
            card = img_big[0:hb, x1:x2]
            name, conf = self._classify(card)
            # 低置信度微调
            if conf < 0.85:
                name2, conf2, _ = self._classify_shift(img_big, x1, 0, x2, hb)
                if conf2 > conf:
                    name, conf = name2, conf2
            results.append((name, conf))
        return results

    def _classify_boxes(self, img, raw_boxes, min_conf):
        """对检测框进行分类（通用逻辑）"""
        h, w = img.shape[:2]
        cards = []
        for c in raw_boxes:
            # 扩展窄框
            if c['x2'] - c['x1'] < self.MIN_CARD_WIDTH:
                c['x1'] = max(0, c['x2'] - self.MIN_CARD_WIDTH)

            edge = (c['x1'] < 25) or (c['x2'] > w - 25)
            if edge:
                name, conf = self._classify_mirror(img, c['x1'], c['y1'], c['x2'], c['y2'])
                name2, conf2, _ = self._classify_shift(img, c['x1'], c['y1'], c['x2'], c['y2'])
                if conf2 > conf:
                    name, conf = name2, conf2
            else:
                card_img = img[c['y1']:c['y2'], c['x1']:c['x2']]
                name, conf = self._classify(card_img)
                if conf < 0.90:
                    name2, conf2, _ = self._classify_shift(img, c['x1'], c['y1'], c['x2'], c['y2'])
                    if conf2 > conf:
                        name, conf = name2, conf2

            cards.append({'name': name, 'conf': conf, 'x': c['cx']})

        # 去重
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

        result = []
        for c in deduped:
            thr = 0.60 if (c['x'] < 30 or c['x'] > w - 30) else min_conf
            if c['conf'] >= thr:
                result.append(c['name'])
        return result
