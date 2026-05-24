#!/usr/bin/env python3
"""打印训练结果摘要"""

import os
import glob
import csv


def main():
    # 找 results.csv
    paths = [
        "runs/detect/ddz_runs_54/train/results.csv",
        "ddz_runs_54/train/results.csv",
    ]
    csv_path = None
    for p in paths:
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        print("未找到 results.csv")
        return

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("results.csv 为空")
        return

    last = rows[-1]
    print("=" * 60)
    print("训练结果摘要")
    print("=" * 60)
    print(f"总轮数: {len(rows)}")
    print(f"最终 mAP50: {float(last.get('metrics/mAP50(B)', 0)):.4f}")
    print(f"最终 mAP50-95: {float(last.get('metrics/mAP50-95(B)', 0)):.4f}")
    print(f"最终 box_loss: {float(last.get('train/box_loss', 0)):.4f}")
    print(f"最终 cls_loss: {float(last.get('train/cls_loss', 0)):.4f}")

    train_dir = os.path.dirname(csv_path)
    print(f"\n结果目录: {train_dir}")
    print("请把以下文件发给我:")
    print(f"  1. {train_dir}/results.png")
    print(f"  2. {train_dir}/confusion_matrix.png")
    print(f"  3. {train_dir}/F1_curve.png")


if __name__ == "__main__":
    main()
