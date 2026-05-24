#!/usr/bin/env python3
"""
把 54 类标注转换为 1 类检测数据集
所有框的类别 ID 改为 0（card）
"""

import os
import shutil
import glob


def main():
    for split in ['train', 'val']:
        os.makedirs(f"dataset_detect/images/{split}", exist_ok=True)
        os.makedirs(f"dataset_detect/labels/{split}", exist_ok=True)
        
        # 复制图片
        src_img = f"dataset_54/images/{split}"
        dst_img = f"dataset_detect/images/{split}"
        for f in os.listdir(src_img):
            shutil.copy2(os.path.join(src_img, f), os.path.join(dst_img, f))
        
        # 转换标注：所有类别改为 0
        src_lbl = f"dataset_54/labels/{split}"
        dst_lbl = f"dataset_detect/labels/{split}"
        for lbl_path in glob.glob(os.path.join(src_lbl, "*.txt")):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # 只保留坐标，类别统一为 0
                    new_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
            
            with open(os.path.join(dst_lbl, os.path.basename(lbl_path)), 'w') as f:
                f.writelines(new_lines)
    
    print("✅ 1 类检测数据集准备完成！")
    print("   图片: dataset_detect/images/")
    print("   标注: dataset_detect/labels/")
    print("   类别: card (0)")


if __name__ == "__main__":
    main()
