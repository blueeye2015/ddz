#!/usr/bin/env python3
"""直接读取 results.csv，显示真实数值"""

import os
import glob
import csv


def main():
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

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    print("列名:", headers)
    print(f"\n总行数: {len(rows)}")

    if not rows:
        return

    # 显示前 3 行和后 3 行
    print("\n前 3 行:")
    for i, row in enumerate(rows[:3]):
        print(f"  epoch {i+1}: {dict(zip(headers, row))}")

    print("\n最后 3 行:")
    for i, row in enumerate(rows[-3:]):
        print(f"  epoch {len(rows)-2+i}: {dict(zip(headers, row))}")

    # 检查 box_loss 和 mAP50 的列名
    box_col = None
    map_col = None
    for h in headers:
        if 'box' in h.lower():
            box_col = h
        if 'mAP50' in h and '95' not in h:
            map_col = h

    print(f"\nbox_loss 列名: {box_col}")
    print(f"mAP50 列名: {map_col}")

    if box_col and rows:
        last_box = rows[-1][headers.index(box_col)]
        print(f"最后一轮 box_loss: {last_box}")
    if map_col and rows:
        last_map = rows[-1][headers.index(map_col)]
        print(f"最后一轮 mAP50: {last_map}")


if __name__ == "__main__":
    main()
