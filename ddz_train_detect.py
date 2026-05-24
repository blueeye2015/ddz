#!/usr/bin/env python3
"""
YOLO 1 类检测模型训练（只检测牌的位置）
输入: dataset_detect/  输出: ddz_detect_best.pt
"""

from ultralytics import YOLO


def main():
    model = YOLO('yolov8n.pt')
    
    # 训练配置
    model.train(
        data='ddz_detect.yaml',
        epochs=100,
        imgsz=1280,          # 高分辨率，小目标定位更准
        batch=4,             # 1280 分辨率内存占用大，batch 调小
        workers=0,           # Windows 下设为 0 避免多进程问题
        device='cpu',        # 如有 GPU 可改为 '0'
        patience=20,         # 早停
        project='ddz_detect',
        name='train',
        exist_ok=True,
        # 增强策略（数据少，增强别太猛）
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=3,
        translate=0.05,
        scale=0.3,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.05,
    )
    
    # 保存最佳权重到项目根目录
    import shutil
    best = 'runs/detect/ddz_detect/train/weights/best.pt'
    shutil.copy(best, 'ddz_detect_best.pt')
    print(f"\n✅ 检测模型已保存: ddz_detect_best.pt")


if __name__ == "__main__":
    main()
