#!/usr/bin/env python3
"""诊断 54 类数据集是否有问题"""

import os
import glob


def check_labels():
    print("=" * 60)
    print("54 类数据集诊断")
    print("=" * 60)

    # 1. 检查目录结构
    for split in ['train', 'val']:
        img_dir = f"dataset/images_54/{split}"
        lbl_dir = f"dataset/labels_54/{split}"

        if not os.path.exists(img_dir):
            print(f"[错误] 目录不存在: {img_dir}")
            continue
        if not os.path.exists(lbl_dir):
            print(f"[错误] 目录不存在: {lbl_dir}")
            continue

        img_files = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png'))
        lbl_files = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt'))

        print(f"\n[{split}]")
        print(f"  图片: {len(img_files)} 张")
        print(f"  标注: {len(lbl_files)} 个")

        # 检查是否有图片没有对应标注
        no_label = img_files - lbl_files
        if no_label:
            print(f"  ⚠️ 有 {len(no_label)} 张图片没有对应标注")
            for name in sorted(no_label)[:5]:
                print(f"    - {name}.png")

        # 检查是否有标注没有对应图片
        no_img = lbl_files - img_files
        if no_img:
            print(f"  ⚠️ 有 {len(no_img)} 个标注没有对应图片")

        # 检查标注文件内容
        empty_labels = 0
        invalid_labels = 0
        class_ids = set()

        for lbl_name in sorted(lbl_files)[:100]:  # 只检查前 100 个
            lbl_path = os.path.join(lbl_dir, f"{lbl_name}.txt")
            with open(lbl_path, 'r') as f:
                lines = f.read().strip().split('\n')

            if not lines or lines == ['']:
                empty_labels += 1
                continue

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_labels += 1
                    continue
                try:
                    cid = int(parts[0])
                    class_ids.add(cid)
                    if cid < 0 or cid > 53:
                        invalid_labels += 1
                except ValueError:
                    invalid_labels += 1

        print(f"  空标注文件: {empty_labels}")
        print(f"  无效标注: {invalid_labels}")
        print(f"  类别 ID 范围: {min(class_ids) if class_ids else 'N/A'} ~ {max(class_ids) if class_ids else 'N/A'}")
        if class_ids - set(range(54)):
            print(f"  ⚠️ 发现越界类别 ID: {sorted(class_ids - set(range(54)))}")

    # 2. 检查 ddz_54.yaml
    print("\n[配置文件]")
    if os.path.exists("ddz_54.yaml"):
        with open("ddz_54.yaml", 'r') as f:
            content = f.read()
        if "nc: 54" in content:
            print("  ✅ nc: 54")
        else:
            print("  ❌ nc 不是 54")

        name_count = content.count(": '")
        print(f"  names 条目数: {name_count}")
        if name_count != 54:
            print(f"  ⚠️ 应该有 54 个 names，实际 {name_count}")
    else:
        print("  ❌ ddz_54.yaml 不存在")

    print("\n" + "=" * 60)


def show_sample_label():
    """显示几个标注文件样本"""
    print("\n[标注样本]")
    lbl_files = glob.glob("dataset/labels_54/train/*.txt")
    if not lbl_files:
        print("  未找到标注文件")
        return

    for path in sorted(lbl_files)[:3]:
        name = os.path.basename(path)
        with open(path, 'r') as f:
            content = f.read().strip()
        print(f"  {name}:")
        for line in content.split('\n')[:5]:
            print(f"    {line}")


if __name__ == "__main__":
    check_labels()
    show_sample_label()
