#!/usr/bin/env python3
"""
两阶段识别器 V3：
- 框微调重分类（解决框偏移问题）
- 边缘镜像填充（解决窗口截断问题）
- 动态阈值（70%为底线，但保留所有结果供上层过滤）
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO


class TwoStageRecognizerV3:
    CLASS_NAMES = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'BJ', 'J', 'K', 'Q', 'SJ']
    
    def __init__(self, 
                 detect_model_path='ddz_detect_best.pt',
                 classify_model_path='card_classifier_best.pth',
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        print(f"Loading detect model: {detect_model_path}")
        self.detect_model = YOLO(detect_model_path)
        
        print(f"Loading classify model: {classify_model_path}")
        self.classifier = models.mobilenet_v3_small(pretrained=False)
        in_features = self.classifier.classifier[3].in_features
        self.classifier.classifier[3] = nn.Linear(in_features, len(self.CLASS_NAMES))
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
        img_rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.classifier(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        
        return self.CLASS_NAMES[pred.item()], conf.item()
    
    def _classify_with_shift(self, hand_img, x1, y1, x2, y2, shifts=[-10, -5, 0, 5, 10]):
        """框微调重分类"""
        h, w = hand_img.shape[:2]
        best_name, best_conf, best_shift = None, 0.0, 0
        
        for dx in shifts:
            nx1 = max(0, x1 + dx)
            nx2 = min(w, x2 + dx)
            card = hand_img[y1:y2, nx1:nx2]
            name, conf = self._classify(card)
            if name and conf > best_conf:
                best_name, best_conf, best_shift = name, conf, dx
        
        return best_name, best_conf, best_shift
    
    def _classify_with_mirror(self, hand_img, x1, y1, x2, y2):
        """
        对边缘框做镜像填充后分类
        解决：窗口截断导致牌只有一半可见的问题
        """
        h, w = hand_img.shape[:2]
        card = hand_img[y1:y2, x1:x2]
        if card.size == 0:
            return None, 0.0
        
        ch, cw = card.shape[:2]
        
        # 如果框靠近左边缘，左边镜像填充
        if x1 < 20:
            pad_left = 20 - x1
            card = cv2.copyMakeBorder(card, 0, 0, pad_left, 0, cv2.BORDER_REFLECT_101)
        
        # 如果框靠近右边缘，右边镜像填充
        if x2 > w - 20:
            pad_right = x2 - (w - 20)
            card = cv2.copyMakeBorder(card, 0, 0, 0, pad_right, cv2.BORDER_REFLECT_101)
        
        return self._classify(card)
    
    def recognize(self, hand_img, min_conf=0.70, return_all=False):
        """
        识别整行手牌
        
        Args:
            hand_img: numpy BGR
            min_conf: 最低分类置信度（默认70%，边缘牌适当放宽）
            return_all: 是否返回所有结果（包括低置信度，供上层二次判断）
        
        Returns:
            list[str] 或 list[dict]
        """
        h, w = hand_img.shape[:2]
        
        # 1. YOLO 检测
        results = self.detect_model(
            hand_img,
            conf=0.10,
            iou=0.15,
            imgsz=1280,
            verbose=False
        )
        
        raw_cards = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                yolo_conf = float(boxes.conf[i].cpu().numpy())
                cx = (x1 + x2) // 2
                
                # 过滤窗口边缘垃圾框（只过滤明显超出窗口的，截断的牌保留）
                if x1 < 0 or x2 > w + 10 or yolo_conf < 0.05:
                    continue
                
                raw_cards.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': cx, 'yolo_conf': yolo_conf
                })
        
        raw_cards.sort(key=lambda c: c['cx'])
        
        # 2. 分类（对窄框强制扩展宽度，解决密集场景下YOLO框偏窄问题）
        MIN_CARD_WIDTH = 70  # 标准牌宽78px，密集场景下检测框偏窄
        
        cards = []
        for c in raw_cards:
            # 强制扩展框宽度到最小70px（向左扩展，不超出图像边界）
            actual_width = c['x2'] - c['x1']
            if actual_width < MIN_CARD_WIDTH:
                c['x1'] = max(0, c['x2'] - MIN_CARD_WIDTH)
            
            is_edge = (c['x1'] < 25) or (c['x2'] > w - 25)
            
            if is_edge:
                # 边缘框：先镜像填充分类
                name, conf = self._classify_with_mirror(
                    hand_img, c['x1'], c['y1'], c['x2'], c['y2']
                )
                # 再尝试微调
                name2, conf2, shift = self._classify_with_shift(
                    hand_img, c['x1'], c['y1'], c['x2'], c['y2']
                )
                if conf2 > conf:
                    name, conf = name2, conf2
            else:
                # 普通框：先直接分类
                card_img = hand_img[c['y1']:c['y2'], c['x1']:c['x2']]
                name, conf = self._classify(card_img)
                
                # 置信度低则微调
                if conf < 0.90:
                    name2, conf2, shift = self._classify_with_shift(
                        hand_img, c['x1'], c['y1'], c['x2'], c['y2']
                    )
                    if conf2 > conf:
                        name, conf = name2, conf2
            
            cards.append({
                'name': name,
                'conf': conf,
                'x': c['cx'],
                'yolo_conf': c['yolo_conf'],
                'is_edge': is_edge
            })
        
        # 3. 去重
        deduped = []
        for c in cards:
            duplicate = False
            for d in deduped:
                if abs(c['x'] - d['x']) < 15:
                    duplicate = True
                    if c['conf'] > d['conf']:
                        d.update(c)
                    break
            if not duplicate:
                deduped.append(c)
        
        deduped.sort(key=lambda c: c['x'])
        
        if return_all:
            return deduped
        
        # 4. 过滤，但边缘牌适当放宽
        filtered = []
        for c in deduped:
            threshold = 0.60 if c.get('is_edge') else min_conf
            if c['conf'] >= threshold:
                filtered.append(c)
        
        return [c['name'] for c in filtered]


def test_on_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return
    
    recognizer = TwoStageRecognizerV3()
    result = recognizer.recognize(img, return_all=True)
    
    print(f"\nImage: {image_path}")
    print(f"Total detected: {len(result)} cards")
    
    # 显示所有结果（不过滤）
    print("\n--- All results (with confidence) ---")
    for c in result:
        edge = "[EDGE]" if c.get('is_edge') else ""
        bar = '█' * int(c['conf'] * 10) + '░' * (10 - int(c['conf'] * 10))
        flag = "LOW" if c['conf'] < 0.70 else "OK"
        print(f"  {c['name']:>3s}  {bar}  {c['conf']:>5.1%}  yolo={c['yolo_conf']:>5.1%} {flag} {edge}")
    
    # 过滤后
    filtered = recognizer.recognize(img, min_conf=0.70)
    print(f"\n--- Filtered (>=70%): {len(filtered)} cards ---")
    print(' '.join(filtered))
    
    return result, filtered


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_on_image(sys.argv[1])
    else:
        print("Usage: python ddz_two_stage_v3.py <screenshot.png>")
