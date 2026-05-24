# 两阶段识别方案 - 执行指南

## 原理

| 阶段 | 任务 | 模型 | 类别数 | 作用 |
|------|------|------|--------|------|
| 第一阶段 | 定位 | YOLOv8n | 1 类 (card) | 把每张牌框出来 |
| 第二阶段 | 分类 | MobileNetV3-Small | 15 类 (3-10,J,Q,K,A,2,SJ,BJ) | 识别框里是什么数字 |

**核心优势**：
- 定位模型只用学"牌长什么样"，不区分数字，极容易学
- 分类模型只在裁剪好的单张牌上做判断，背景干净、干扰少
- 两阶段解耦，各自做擅长的事，准确率质变

---

## 执行步骤

### Step 0: 环境检查

```bash
# 检查 PyTorch（训练分类器需要）
python -c "import torch; print(torch.__version__)"

# 如果没有，安装 CPU 版（约 200MB）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 1: 准备检测数据集

把 54 类标注转换为 1 类（所有框标签改为 `card`）：

```bash
python prepare_detect.py
```

输出：`dataset_detect/images/` + `dataset_detect/labels/`

### Step 2: 训练 YOLO 定位模型

```bash
python ddz_train_detect.py
```

- 输入：`dataset_detect/`
- 输出：`ddz_detect_best.pt`（自动复制到根目录）
- 配置：`imgsz=1280`（高分辨率定位小目标）

训练完成后，验证检测效果：
```python
from ultralytics import YOLO
model = YOLO('ddz_detect_best.pt')
model('test.png', show=True)  # 看框是否准确
```

### Step 3: 训练分类模型

```bash
python train_classifier.py
```

- 输入：`dataset/by_class/` 下的 15 类裁剪图
- 输出：`card_classifier_best.pth`
- 模型：MobileNetV3-Small（CPU 推理极快）

> 注意：`dataset/by_class/` 下必须有 15 个文件夹：`10`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `A`, `BJ`, `J`, `K`, `Q`, `SJ`

### Step 4: 集成推理

```bash
python ddz_two_stage.py your_screenshot.png
```

或在代码中使用：
```python
from ddz_two_stage import TwoStageRecognizer

rec = TwoStageRecognizer()
result = rec.recognize(hand_img)  # numpy BGR
print(result)  # ['3', '4', '5', '6', '7', ...]
```

---

## 预期效果

| 指标 | 54 类单模型 | 两阶段方案 |
|------|-----------|-----------|
| 定位 mAP50 | 0.22（数据不足） | >0.85（1 类极易学） |
| 分类准确率 | — | >95%（单张牌干净背景） |
| 综合准确率 | ~22% | **>80%，有望突破 95%** |
| 推理速度 | 中等 | YOLO 10fps + 分类 1ms/张 |

---

## 进阶优化

1. **红色 5 问题**：分类模型训练时加入 `ColorJitter`，对红色牌做数据增强
2. **缓存机制**：`TwoStageRecognizer` 已内置坐标缓存，同一位置复用结果
3. **预处理**：分类前对裁剪图做灰度化或对比度增强，消除光影干扰
4. **更大数据**：若分类准确率不够，用 YOLO 检测模型在实际环境中自动截取更多牌片，扩充 `dataset/by_class/`
