#!/usr/bin/env python3
"""
清洗 dataset/by_class/5/ 里的错误样本
用模型对每张图做推理，预测不是 5 的（或置信度 < 0.3）移到 review/ 待检查
"""

import os
import shutil
import glob
import cv2
from ddz_yolo import YOLORecognizer


def main():
    rec = YOLORecognizer("ddz_yolo.pt", conf_threshold=0.01)

    five_files = glob.glob("dataset/by_class/5/*.png")
    print(f"扫描 dataset/by_class/5/ 共 {len(five_files)} 张")

    os.makedirs("dataset/by_class/review_from_5", exist_ok=True)

    keep = 0
    move = 0

    for path in five_files:
        img = cv2.imread(path)
        if img is None:
            continue

        # 单张推理
        results = rec.model(img, verbose=False, conf=0.01)

        pred_name = "?"
        pred_conf = 0.0

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            # 取置信度最高的预测
            best_idx = int(boxes.conf.argmax())
            pred_conf = float(boxes.conf[best_idx])
            cls_id = int(boxes.cls[best_idx])
            pred_name = rec.ID2NAME.get(cls_id, f"?{cls_id}")

        fname = os.path.basename(path)

        # 如果预测不是 5，或者置信度 < 0.3，移到 review
        if pred_name != '5' or pred_conf < 0.30:
            shutil.move(path, os.path.join("dataset/by_class/review_from_5", fname))
            move += 1
            if move <= 10:
                print(f"  [移出] {fname}: 模型认为它是 {pred_name}(conf={pred_conf:.3f})")
        else:
            keep += 1

    print(f"\n{'=' * 50}")
    print(f"清洗完成:")
    print(f"  保留: {keep} 张（模型确信是 5）")
    print(f"  移出: {move} 张（模型认为不是 5 或不够确信）")
    print(f"{'=' * 50}")
    print(f"\n请检查 dataset/by_class/review_from_5/")
    print(f"  - 如果里面确实是 5（模型看错了），拖回 dataset/by_class/5/")
    print(f"  - 如果不是 5，删掉或拖进正确类别")
    print(f"\n清理完后重新生成标注并训练:")
    print(f"  python generate_yolo_labels.py")
    print(f"  python ddz_train.py")


if __name__ == "__main__":
    main()
