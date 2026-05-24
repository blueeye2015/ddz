#!/usr/bin/env python3
"""
两阶段识别器：YOLO 定位 + MobileNetV3 分类
输入：整行手牌截图  输出：按位置排序的牌面列表
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO


class TwoStageRecognizer:
    """
    两阶段识别器
    
    使用方式：
        recognizer = TwoStageRecognizer()
        cards = recognizer.recognize(hand_img)  # hand_img: numpy BGR
        print(cards)  # ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    """
    
    # ImageFolder 按字母排序的类别名
    CLASS_NAMES = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'BJ', 'J', 'K', 'Q', 'SJ']
    
    def __init__(self, 
                 detect_model_path='ddz_detect_best.pt',
                 classify_model_path='card_classifier_best.pth',
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        # ========== 第一阶段：YOLO 检测 ==========
        print(f"📦 加载检测模型: {detect_model_path}")
        self.detect_model = YOLO(detect_model_path)
        
        # ========== 第二阶段：分类 ==========
        print(f"📦 加载分类模型: {classify_model_path}")
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
        
        # 缓存机制：同一位置 +/- 10px 的牌复用上一次的分类结果
        self._cache = {}  # {(x_center_rounded, y_center_rounded): name}
        self.cache_tolerance = 15
    
    def _get_cache_key(self, x, y):
        """将坐标离散化为缓存键"""
        return (round(x / self.cache_tolerance), round(y / self.cache_tolerance))
    
    def _classify(self, card_img):
        """对单张牌图片进行分类"""
        # BGR -> RGB
        img_rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.classifier(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        
        return self.CLASS_NAMES[pred.item()], conf.item()
    
    def recognize(self, hand_img, use_cache=True, return_conf=False):
        """
        识别整行手牌
        
        Args:
            hand_img: numpy array, BGR 格式
            use_cache: 是否启用坐标缓存
            return_conf: 是否返回置信度
        
        Returns:
            list[str] 或 list[tuple[str, float]]
        """
        # 1. YOLO 检测所有牌的位置
        results = self.detect_model(
            hand_img,
            conf=0.15,      # 进一步放宽，确保不漏框
            iou=0.2,        # NMS 放宽，密集牌减少合并
            imgsz=1280,     # 高分辨率检测小目标
            verbose=False
        )
        
        cards = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # 裁剪单张牌
                card_img = hand_img[max(0, y1):y2, max(0, x1):x2]
                if card_img.size == 0:
                    continue
                
                # 缓存检查
                cache_key = self._get_cache_key(cx, cy)
                if use_cache and cache_key in self._cache:
                    name = self._cache[cache_key]
                    conf = 1.0  # 缓存命中视为高置信度
                    print(f"  💾 缓存命中: {name} @ ({cx}, {cy})")
                else:
                    # 分类器识别
                    name, conf = self._classify(card_img)
                    if use_cache:
                        self._cache[cache_key] = name
                
                cards.append({
                    'name': name,
                    'conf': conf,
                    'x': cx,
                    'y': cy,
                    'box': (x1, y1, x2, y2)
                })
        
        # 按 x 坐标排序（从左到右）
        cards.sort(key=lambda c: c['x'])
        
        # 去重：同一 x 位置只保留置信度最高的（处理重叠框）
        deduped = []
        for c in cards:
            # 检查是否已有相近位置的牌
            duplicate = False
            for d in deduped:
                if abs(c['x'] - d['x']) < 12:  # 12px 内视为同一张牌（牌宽约78px，间距约15px）
                    duplicate = True
                    if c['conf'] > d['conf']:
                        d.update(c)
                    break
            if not duplicate:
                deduped.append(c)
        
        # 硬阈值：只保留高置信度结果（<85%的几乎全是误检）
        filtered = [c for c in deduped if c['conf'] >= 0.85]
        if return_conf:
            return [(c['name'], c['conf']) for c in filtered]
        return [c['name'] for c in filtered]
    
    def clear_cache(self):
        """清空坐标缓存"""
        self._cache.clear()
        print("🗑️  缓存已清空")


def test_on_image(image_path):
    """测试单张图片"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    recognizer = TwoStageRecognizer()
    result = recognizer.recognize(img, return_conf=True)
    
    print(f"\n🖼️  图片: {image_path}")
    print(f"🃏 识别结果 ({len(result)} 张):")
    for name, conf in result:
        bar = '█' * int(conf * 10) + '░' * (10 - int(conf * 10))
        print(f"   {name:>3s}  {bar}  {conf:.2%}")
    
    # 可视化
    vis = img.copy()
    # 需要重新运行以获取 box 信息，这里简化为只打印结果
    
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_on_image(sys.argv[1])
    else:
        print("用法: python ddz_two_stage.py <手牌截图路径>")
        print("\n请先完成以下步骤:")
        print("  1. python prepare_detect.py")
        print("  2. python ddz_train_detect.py")
        print("  3. python train_classifier.py")
        print("  4. python ddz_two_stage.py your_screenshot.png")
