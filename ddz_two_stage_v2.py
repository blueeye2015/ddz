#!/usr/bin/env python3
"""
两阶段识别器 V2：YOLO 定位 + MobileNetV3 分类 + 框微调重分类
输入：整行手牌截图  输出：按位置排序的牌面列表
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO


class TwoStageRecognizerV2:
    """
    改进版两阶段识别器
    - 过滤窗口边缘垃圾框
    - 对低置信度框做左右微调重分类
    """
    
    CLASS_NAMES = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'BJ', 'J', 'K', 'Q', 'SJ']
    
    def __init__(self, 
                 detect_model_path='ddz_detect_best.pt',
                 classify_model_path='card_classifier_best.pth',
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # 第一阶段：YOLO 检测
        print(f"Loading detect model: {detect_model_path}")
        self.detect_model = YOLO(detect_model_path)
        
        # 第二阶段：分类
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
        """对单张牌图片分类，返回 (name, conf)"""
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
    
    def _classify_with_shift(self, hand_img, x1, y1, x2, y2, shifts=[-8, -4, 0, 4, 8]):
        """
        对框做左右微调，取最高置信度的分类结果
        解决：框稍微偏左/偏右导致裁剪到相邻牌的问题
        """
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
    
    def recognize(self, hand_img, return_conf=False):
        """识别整行手牌"""
        h, w = hand_img.shape[:2]
        
        # 1. YOLO 检测（低 conf 确保不漏，低 iou 减少合并）
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
                
                # 过滤窗口边缘垃圾框（x<5 或 x>w-5）
                if x1 < 5 or x2 > w - 5:
                    continue
                
                raw_cards.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': cx, 'yolo_conf': yolo_conf
                })
        
        # 按 x 排序
        raw_cards.sort(key=lambda c: c['cx'])
        
        # 2. 分类 + 框微调
        cards = []
        for c in raw_cards:
            # 第一次分类
            card_img = hand_img[c['y1']:c['y2'], c['x1']:c['x2']]
            name, conf = self._classify(card_img)
            shift = 0
            
            # 如果置信度低，做框微调重分类
            if conf < 0.85:
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
                'shift': shift
            })
        
        # 3. 去重：同一位置保留置信度最高的
        deduped = []
        for c in cards:
            duplicate = False
            for d in deduped:
                if abs(c['x'] - d['x']) < 15:
                    duplicate = True
                    # 如果新框置信度更高，替换
                    if c['conf'] > d['conf']:
                        d.update(c)
                    break
            if not duplicate:
                deduped.append(c)
        
        # 4. 过滤低置信度（<80% 的很可能是不确定的误检）
        filtered = [c for c in deduped if c['conf'] >= 0.80]
        filtered.sort(key=lambda c: c['x'])
        
        if return_conf:
            return [(c['name'], c['conf'], c.get('shift', 0)) for c in filtered]
        return [c['name'] for c in filtered]


def test_on_image(image_path):
    """测试单张图片"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return
    
    recognizer = TwoStageRecognizerV2()
    result = recognizer.recognize(img, return_conf=True)
    
    print(f"\nImage: {image_path}")
    print(f"Detected: {len(result)} cards")
    for name, conf, shift in result:
        shift_str = f"(shift={shift:+d})" if shift != 0 else ""
        bar = '█' * int(conf * 10) + '░' * (10 - int(conf * 10))
        print(f"  {name:>3s}  {bar}  {conf:.1%} {shift_str}")
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_on_image(sys.argv[1])
    else:
        print("Usage: python ddz_two_stage_v2.py <screenshot.png>")
