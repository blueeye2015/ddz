#!/usr/bin/env python3
"""
根据用户分类结果生成 YOLO 格式的标注文件

前置步骤：
1. python prepare_dataset.py         # 自动裁剪
2. 手动把 crops_raw/ 下的图分类到 by_class/ 下
3. python generate_yolo_labels.py    # 生成本脚本
"""

import os
import json
import shutil
import random
from glob import glob


CLASS_MAP = {
    '3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6,
    '10': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12,
    'SJ': 13, 'BJ': 14
}

VAL_RATIO = 0.2  # 20% 做验证集


def main():
    if not os.path.exists("dataset/crops_meta.json"):
        print("[错误] 找不到 dataset/crops_meta.json，请先运行 prepare_dataset.py")
        return

    with open("dataset/crops_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 清空旧数据
    for d in ["dataset/images/train", "dataset/images/val",
              "dataset/labels/train", "dataset/labels/val"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 收集所有标注: source_name -> list of (class_id, cx, cy, nw, nh)
    labels_by_image = {}

    for class_name, class_id in CLASS_MAP.items():
        class_dir = f"dataset/by_class/{class_name}"
        if not os.path.exists(class_dir):
            continue

        images = glob(os.path.join(class_dir, "*.png")) + \
                 glob(os.path.join(class_dir, "*.jpg")) + \
                 glob(os.path.join(class_dir, "*.jpeg"))

        for img_path in images:
            crop_name = os.path.basename(img_path)
            if crop_name not in meta:
                print(f"[警告] {crop_name} 不在元数据中，跳过")
                continue

            info = meta[crop_name]
            source_name = os.path.splitext(os.path.basename(info["source"]))[0]

            # 转换为 YOLO 归一化格式
            cx = (info["x"] + info["w"] / 2.0) / info["img_w"]
            cy = (info["y"] + info["h"] / 2.0) / info["img_h"]
            nw = info["w"] / float(info["img_w"])
            nh = info["h"] / float(info["img_h"])

            # 裁剪到 [0, 1] 范围内
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            if source_name not in labels_by_image:
                labels_by_image[source_name] = []

            labels_by_image[source_name].append((class_id, cx, cy, nw, nh))

    if not labels_by_image:
        print("[错误] 没有收集到任何标注，请检查 dataset/by_class/ 是否已分类")
        return

    # 划分 train/val
    all_names = list(labels_by_image.keys())
    random.shuffle(all_names)
    val_size = max(1, int(len(all_names) * VAL_RATIO))
    val_names = set(all_names[:val_size])
    train_names = set(all_names[val_size:])

    print(f"总图片数: {len(all_names)}，训练集: {len(train_names)}，验证集: {len(val_names)}")

    # 复制原图并保存标注
    for source_name, labels in labels_by_image.items():
        split = "val" if source_name in val_names else "train"

        # 复制原图
        src_path = meta.get(f"{source_name}_00.png", {}).get("source", "")
        if not src_path:
            # 尝试从任意一个 crop 找 source
            for k, v in meta.items():
                if v["source"].endswith(f"{source_name}.png") or \
                   v["source"].endswith(f"{source_name}.jpg"):
                    src_path = v["source"]
                    break

        if src_path and os.path.exists(src_path):
            ext = os.path.splitext(src_path)[1]
            dst_img = f"dataset/images/{split}/{source_name}{ext}"
            shutil.copy2(src_path, dst_img)

        # 保存标注
        label_path = f"dataset/labels/{split}/{source_name}.txt"
        with open(label_path, "w") as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

    print(f"\n✅ YOLO 数据集生成完成！")
    print(f"   图片: dataset/images/")
    print(f"   标注: dataset/labels/")
    print(f"\n下一步: 运行 python ddz_train.py 开始训练")


if __name__ == "__main__":
    main()
