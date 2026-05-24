#!/usr/bin/env python3
"""
54 类 YOLO 标注生成
基于 dataset/by_class_54/ 的分类结果和 crops_meta.json 元数据
"""

import os
import json
import shutil
import random
import glob


# 54 类映射
CLASS_MAP = {}
idx = 0
for num in ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']:
    for suit in ['黑桃', '红心', '梅花', '方块']:
        CLASS_MAP[f"{num}_{suit}"] = idx
        idx += 1
CLASS_MAP['SJ'] = 52
CLASS_MAP['BJ'] = 53

VAL_RATIO = 0.2


def main():
    if not os.path.exists("dataset/crops_meta.json"):
        print("[错误] 找不到 dataset/crops_meta.json")
        return

    with open("dataset/crops_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 清空旧数据
    for d in ["dataset/images_54/train", "dataset/images_54/val",
              "dataset/labels_54/train", "dataset/labels_54/val"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    labels_by_image = {}

    for class_name, class_id in CLASS_MAP.items():
        class_dir = f"dataset/by_class_54/{class_name}"
        if not os.path.exists(class_dir):
            continue

        images = glob.glob(os.path.join(class_dir, "*.png"))
        for path in images:
            fname = os.path.basename(path)
            if fname not in meta:
                continue

            info = meta[fname]
            source_name = os.path.splitext(os.path.basename(info["source"]))[0]

            cx = (info["x"] + info["w"] / 2.0) / info["img_w"]
            cy = (info["y"] + info["h"] / 2.0) / info["img_h"]
            nw = info["w"] / float(info["img_w"])
            nh = info["h"] / float(info["img_h"])

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            if source_name not in labels_by_image:
                labels_by_image[source_name] = []

            labels_by_image[source_name].append((class_id, cx, cy, nw, nh))

    if not labels_by_image:
        print("[错误] 没有收集到任何标注")
        return

    # 划分 train/val
    all_names = list(labels_by_image.keys())
    random.shuffle(all_names)
    val_size = max(1, int(len(all_names) * VAL_RATIO))
    val_names = set(all_names[:val_size])

    print(f"总图片: {len(all_names)}, 训练: {len(all_names)-len(val_names)}, 验证: {len(val_names)}")

    for source_name, labels in labels_by_image.items():
        split = "val" if source_name in val_names else "train"

        # 复制原图
        src_path = None
        for k, v in meta.items():
            if v["source"].endswith(f"{source_name}.png") or v["source"].endswith(f"{source_name}.jpg"):
                src_path = v["source"]
                break

        if src_path and os.path.exists(src_path):
            ext = os.path.splitext(src_path)[1]
            dst_img = f"dataset/images_54/{split}/{source_name}{ext}"
            shutil.copy2(src_path, dst_img)

        # 保存标注
        label_path = f"dataset/labels_54/{split}/{source_name}.txt"
        with open(label_path, "w") as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

    print(f"\n✅ 54 类 YOLO 数据集生成完成！")
    print(f"   图片: dataset/images_54/")
    print(f"   标注: dataset/labels_54/")


if __name__ == "__main__":
    main()
