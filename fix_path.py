#!/usr/bin/env python3
"""
把 54 类数据复制到标准路径 dataset_54/images/ 和 dataset_54/labels/
绕过 ultralytics 对带下划线路径名的解析 bug
"""

import os
import shutil


def copy_dir(src, dst):
    if not os.path.exists(src):
        return
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copy_dir(s, d)
        else:
            shutil.copy2(s, d)


def main():
    # 创建标准路径
    copy_dir("dataset/images_54/train", "dataset_54/images/train")
    copy_dir("dataset/images_54/val", "dataset_54/images/val")
    copy_dir("dataset/labels_54/train", "dataset_54/labels/train")
    copy_dir("dataset/labels_54/val", "dataset_54/labels/val")

    # 重写 YAML 使用标准路径
    yaml_content = """path: ./dataset_54
train: images/train
val: images/val

nc: 54

names:
  - '3S'
  - '3H'
  - '3C'
  - '3D'
  - '4S'
  - '4H'
  - '4C'
  - '4D'
  - '5S'
  - '5H'
  - '5C'
  - '5D'
  - '6S'
  - '6H'
  - '6C'
  - '6D'
  - '7S'
  - '7H'
  - '7C'
  - '7D'
  - '8S'
  - '8H'
  - '8C'
  - '8D'
  - '9S'
  - '9H'
  - '9C'
  - '9D'
  - '10S'
  - '10H'
  - '10C'
  - '10D'
  - 'JS'
  - 'JH'
  - 'JC'
  - 'JD'
  - 'QS'
  - 'QH'
  - 'QC'
  - 'QD'
  - 'KS'
  - 'KH'
  - 'KC'
  - 'KD'
  - 'AS'
  - 'AH'
  - 'AC'
  - 'AD'
  - '2S'
  - '2H'
  - '2C'
  - '2D'
  - 'SJ'
  - 'BJ'
"""

    with open("ddz_54_fixed.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # 修改训练脚本
    with open("ddz_train_54.py", "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace('data="ddz_54.yaml"', 'data="ddz_54_fixed.yaml"')
    with open("ddz_train_54.py", "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ 已修复路径问题")
    print("  数据集: dataset_54/images/ 和 dataset_54/labels/")
    print("  配置: ddz_54_fixed.yaml")
    print("  训练脚本已更新")
    print("\n下一步: python ddz_train_54.py")


if __name__ == "__main__":
    main()
