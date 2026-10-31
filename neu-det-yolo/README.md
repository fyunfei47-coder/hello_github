# 基于 YOLO 的 NEU-DET 钢材表面缺陷检测

目标：在普通电脑（CPU）上跑通一个完整的工业表面缺陷检测流程：
下载数据 → 转成 YOLO 格式 → 训练 nano 模型 → 推理并输出检测结果。

## 数据集：NEU-DET

- 来源：东北大学热轧带钢表面缺陷数据库，检测版 NEU-DET
- 规模：1800 张灰度图，200×200 像素，6 类缺陷，每类 300 张
- 类别（本项目统一用这个顺序）：
  1. `crazing`（裂纹）
  2. `inclusion`（夹杂）
  3. `patches`（斑块）
  4. `pitted_surface`（麻点）
  5. `rolled-in_scale`（轧入氧化皮）
  6. `scratches`（划伤）
- 原始标注：VOC XML 框标注，需要用 `scripts/convert_neu_det_to_yolo.py` 转成 YOLO txt

下载地址：https://www.kaggle.com/datasets/bobweng/neu-surface-defect-dataset/data

把下载解压后的内容放到 `data_raw/`，转换脚本支持两种常见结构：

```
data_raw/
  train/images/*.jpg
  train/annotations/*.xml
  validation/images/*.jpg
  validation/annotations/*.xml
```

或

```
data_raw/
  IMAGES/*.jpg
  ANNOTATIONS/*.xml
```

## 目录结构

```
neu-det-yolo/
  README.md
  requirements.txt
  data.yaml
  scripts/
    convert_neu_det_to_yolo.py
    train.py
    predict.py
  data_raw/            # 你下载的原始数据（不提交）
  dataset/             # 转换后的 YOLO 数据（不提交）
  runs/                # 训练输出（不提交）
```

## 1. 安装（普通电脑，CPU）

先装 CPU 版 PyTorch，再装 ultralytics，避免默认装上很大的 CUDA 版：

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

注意：torchvision 一定要和 torch 一起从 CPU 源装，不要让它从默认 PyPI 装成不匹配的版本，否则训练时验证器初始化会报 `torchvision::nms does not exist`。

## 2. 转换数据

```bash
python scripts/convert_neu_det_to_yolo.py \
  --src data_raw \
  --out dataset
```

转换后会生成：

```
dataset/images/train/*.jpg
dataset/images/val/*.jpg
dataset/labels/train/*.txt
dataset/labels/val/*.txt
```

以及 `data.yaml`（已随项目提供，路径指向 `dataset/`）。

## 3. 训练（CPU 先跑通）

```bash
python scripts/train.py --model yolov8n.pt --epochs 20 --imgsz 320 --batch 16
```

- NEU-DET 原图只有 200×200，`imgsz 320` 已够用，不用一上来就用 640。
- CPU 训练速度取决于你的电脑，先用 20 轮跑通全流程，再根据 mAP 决定是否加轮数或换 `yolo11n.pt`。
- 训练结果在 `runs/detect/neu-det-n/`，最好的权重是 `runs/detect/neu-det-n/weights/best.pt`。

## 4. 推理

单张图：

```bash
python scripts/predict.py --model runs/detect/neu-det-n/weights/best.pt --source path/to/test.jpg
```

整个文件夹，并把检测结果写成 CSV：

```bash
python scripts/predict.py \
  --model runs/detect/neu-det-n/weights/best.pt \
  --source dataset/images/val \
  --save-csv runs/predictions.csv
```

标注图默认保存到 `runs/predict/`，CSV 里有图片名、类别、置信度、框坐标。

## 评估重点

只看平均 mAP 不够，重点看这几项：

- 每类的 mAP50、召回率，哪个缺陷最容易漏检
- 小缺陷、弱对比缺陷的表现
- 误检主要来自哪类背景或哪类缺陷混淆

下一步可做的事：增加难例、调 `imgsz`、换 nano→small 模型、做数据增强，都是在跑通这个基线之后再做。
