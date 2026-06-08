# 欢乐斗地主 AI Bot - Agent 指南

## 项目概述

基于 **两阶段识别架构** 的斗地主辅助 bot：
- **阶段一**：YOLOv8n 1-class 检测器定位每张牌的位置
- **阶段二**：MobileNetV3-Small 15-class 分类器识别牌面数字

> 单阶段 54-class YOLO 已验证不可行（30 张图训练 mAP@0.5 仅 1.5%），所有生产代码均围绕两阶段方案。

---

## 核心系统

### `ddz.py` - 主程序
欢乐斗地主主 bot，截图、识别、输出手牌/底牌/出牌区结果。

```bash
# 运行主程序（需要游戏窗口已打开）
python ddz.py

# 保存截图到 screenshots/（用于数据集准备）
python ddz.py --save-screenshot
```

**依赖**：`ddz_config.json`（坐标配置）、`ddz_yolo_recognizer.py`

### `ddz_yolo_recognizer.py` - 两阶段识别器 V3
核心识别模块，封装 `TwoStageRecognizerV3`：
- `recognize(hand_img)` — 手牌识别（重叠牌，支持 shift/mirror 边缘处理）
- `recognize_play(play_img)` — 出牌区识别（带宽高比过滤）
- `recognize_bottom(bottom_img)` — 底牌识别（3x 上采样 + 垂直投影）

**类名顺序**：`['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'BJ', 'J', 'K', 'Q', 'SJ']`

**模型文件**：
- `ddz_detect_best.pt` — 1-class YOLO 检测器
- `card_classifier_best.pth` — MobileNetV3 分类器

---

## 数据采集与标注

### `auto_capture.py` - 自动截图
用 `mss`（默认）或 `win32gui` 后端，定时截取游戏窗口。

```bash
# 默认 mss 后端，每秒 1 张，保存到 raw_auto/
python auto_capture.py

# win32gui 后端（BitBlt），自定义间隔和窗口
python auto_capture.py --backend win32 --interval 0.5 --window "腾讯欢乐斗地主"

# 修正坐标偏移（DPI 或窗口边框问题）
python auto_capture.py --offset-x 8 --offset-y 31
```

**参数**：`--backend`, `--interval`, `--output`, `--window`, `--offset-x`, `--offset-y`

### `capture_and_split.py` - 实时截图分割
按键截图，自动分割手牌/底牌/出牌区并保存单张牌图。

```bash
python capture_and_split.py
# 按键: S=截图并分割, Q=退出
```

**输出**：
- `captured_cards/hand/` — 手牌分割图
- `captured_cards/bottom/` — 底牌分割图
- `captured_cards/play/` — 出牌区分割图
- `captured_cards/debug/` — 带框调试图

### `split_existing.py` - 批量分割已有截图
对 `region_*.png`（ddz.py 提取的区域图）或 `raw_screenshots/` 下的完整截图做批量分割。

```bash
# 分割 region_*.png（默认）
python split_existing.py

# 分割 raw_screenshots/ 下的完整截图
python split_existing.py --raw
```

### `label_helper.py` - YOLO 标注辅助
为图片生成 YOLO 格式 `.txt` 标注文件。

```bash
# 代码传入模式（直接给像素坐标）
python label_helper.py image.png --code "3 850 250 950 400" "5 960 250 1060 400"

# 自动预标注（用已有 ddz_detect_best.pt 预打标，人工复核）
python label_helper.py image.png --auto --viz

# 手动交互模式
python label_helper.py image.png --manual
```

**参数**：`image`, `--code`, `--auto`, `--manual`, `--output`, `--viz`

---

## 数据预处理

### `prepare_dataset.py` - 半自动数据集准备
读取 `screenshots/` 下手牌截图，自动分割裁剪到 `dataset/crops_raw/`，保存元数据。

```bash
python prepare_dataset.py
# 下一步：手动把 crops_raw/ 下的图按类别拖进 dataset/by_class/ 对应文件夹
```

### `generate_yolo_labels.py` - 生成 YOLO 标注
根据 `by_class/` 分类结果，反向生成 YOLO `.txt` 标注文件。

```bash
# 前置：prepare_dataset.py + 手动分类
python generate_yolo_labels.py
```

### `extract_bottom_cards.py` - 底牌分割
对底牌区域图做垂直投影分割，输出单张底牌。

```bash
python extract_bottom_cards.py screenshots/bottom_001.png
# 输出到 extracted_bottom/
```

---

## 模型训练

### `ddz_train_detect.py` - 训练 1-class YOLO 检测器

```bash
python ddz_train_detect.py
```

- **输入**：`dataset_detect/`（由 `prepare_detect.py` 生成）
- **输出**：`ddz_detect_best.pt`
- **配置**：`imgsz=1280`, `epochs=100`, `batch=4`

前置：
```bash
python prepare_detect.py    # 把 54 类标注转 1 类
```

### `train_classifier.py` - 训练 MobileNetV3 分类器

```bash
python train_classifier.py
```

- **输入**：`dataset/by_class/`（15 个文件夹：`10`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `A`, `BJ`, `J`, `K`, `Q`, `SJ`）
- **输出**：`card_classifier_best.pth`
- **模型**：MobileNetV3-Small

### `train_yolov8.py` - YOLOv8 训练脚本（54 类，已废弃）

```bash
python train_yolov8.py --data labels_my-project-name --epochs 100
```

> ⚠️ 已验证效果极差（mAP@0.5 = 1.5%），仅作参考保留。生产环境请用两阶段方案。

---

## 测试与验证

### `test_recognize_from_file.py` - 离线测试两阶段识别
不需要开游戏窗口，直接对截图文件测试手牌/底牌/出牌识别。

```bash
# 测试单张截图
python test_recognize_from_file.py raw_screenshots/001_xxx_full.png

# 测试多张
python test_recognize_from_file.py raw_screenshots/*.png
```

### `test_on_images.py` - Roboflow 模型测试
用 `yolov8l best.pt`（52-class Roboflow 模型）测试标准扑克牌图片。

```bash
python test_on_images.py images/
```

> 该模型在标准牌上 conf>0.9，在游戏截图上完全失效（domain gap）。

### `test_roboflow_model.py` - Roboflow 模型评估
对比 Roboflow `best.pt` 在游戏截图 vs 标准牌上的效果。

```bash
python test_roboflow_model.py
```

### `test_yolo54.py` - 54 类 YOLO 测试（已废弃）

```bash
# 截图测试（需开游戏窗口）
python test_yolo54.py

# 或用 --image 指定文件（脚本内有该参数）
```

---

## 配置说明

### `ddz_config.json`
坐标配置文件，覆盖 `ddz.py` 中的默认 `ELEMENTS`：

```json
{
  "base_width": 2071,
  "base_height": 1231,
  "game_region": { "x_offset": 0, "y_offset": 0, "width": 0, "height": 0 },
  "elements": {
    "my_hand": { "x": 10, "y": 850, "w": 2051, "h": 230 },
    "bottom_cards": { "x": 25, "y": 175, "w": 500, "h": 130 },
    "play_area": { "x": 350, "y": 245, "w": 1350, "h": 620 }
  }
}
```

- `game_region`：游戏画面在客户区内的偏移（网页/小程序嵌套时需要）
- `elements`：各 ROI 相对于游戏区域的坐标（基于 2071×1231 基准分辨率）
- 实际运行时会根据截图尺寸自动缩放

---

## 目录结构速查

```
doudizhubot/
├── ddz.py                          # 主程序
├── ddz_yolo_recognizer.py          # 两阶段识别器 V3
├── ddz_config.json                 # 坐标配置
│
├── ddz_detect_best.pt              # 1-class 检测模型（生产）
├── card_classifier_best.pth        # 15-class 分类模型（生产）
│
├── auto_capture.py                 # 自动截图工具
├── capture_and_split.py            # 实时截图分割
├── split_existing.py               # 批量分割已有截图
├── label_helper.py                 # 标注辅助
│
├── prepare_dataset.py              # 数据集准备
├── prepare_detect.py               # 54类→1类转换
├── generate_yolo_labels.py         # 生成 YOLO 标签
├── extract_bottom_cards.py         # 底牌分割
│
├── ddz_train_detect.py             # 训练检测器
├── train_classifier.py             # 训练分类器
├── train_yolov8.py                 # 54类训练（废弃）
│
├── test_recognize_from_file.py     # 离线测试两阶段识别
├── test_on_images.py               # Roboflow 模型测试
│
├── raw_screenshots/                # 原始完整截图
├── screenshots/                    # ddz.py --save-screenshot 输出
├── captured_cards/                 # capture_and_split.py 输出
├── dataset/
│   ├── by_class/                   # 15 类分类训练数据
│   └── crops_raw/                  # 自动裁剪的原始牌图
├── dataset_detect/                 # 1-class 检测训练数据
├── labels_my-project-name/         # 54类标注数据（30张）
└── test_output/                    # 测试结果可视化
```

---

## 开发备忘

- **DPI 问题**：Windows DPI 缩放会导致 `win32gui.GetClientRect` 与 `mss` 截图尺寸不匹配。已在 `WindowFinder` 中调用 `SetProcessDPIAware()`，并在 `FixedCoords` 中用实际截图尺寸做校准。
- **GPU 训练**：`ddz` conda 环境已安装 CUDA 版 PyTorch（`torch 2.6.0+cu124`），训练脚本会自动检测 GPU。
- **红色 5 问题**：分类器对红心 5/方块 5 容易误判，训练时建议加入 `ColorJitter` 数据增强。
- **底牌识别**：底牌尺寸约 50px，检测器容易漏检，当前用 3x 上采样 + 垂直投影作为 fallback。
