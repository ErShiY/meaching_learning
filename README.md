# Human Protein Atlas Image Classification

本仓库是 Kaggle [Human Protein Atlas Image Classification](https://www.kaggle.com/competitions/human-protein-atlas-image-classification) 比赛的小组项目。

比赛任务是根据四通道荧光显微图像预测蛋白质所在的细胞结构，属于 **28 类多标签图像分类**。一张图像可以同时对应多个类别。

当前代码是项目的第一版 Baseline，主要用于跑通数据读取、训练、验证、结果分析和 Kaggle 提交流程。后续实验会在此基础上继续更新模型、数据划分、类别不平衡处理和推理策略。

## 当前 Baseline

- Backbone：DenseNet121
- 输入通道：Blue、Green、Red、Yellow
- 输入形式：四张单通道图像按固定顺序堆叠为四通道图像
- 预训练：支持加载本地 ImageNet 预训练权重
- 第四通道初始化：使用原始三个通道卷积权重的均值
- 特征聚合：Global Average Pooling
- 分类头：全连接层输出28个 logits
- 损失函数：`BCEWithLogitsLoss`
- 验证指标：Macro-F1 和 Micro-F1
- 最优模型：按照验证集 Macro-F1 保存
- 默认预测阈值：0.5

模型内部不使用 Sigmoid。训练时直接将 logits 传给 `BCEWithLogitsLoss`，计算指标和推理时才使用 Sigmoid 转换为概率。

## 项目结构

```text
project-root/
├── dataset/
│   ├── train.csv
│   ├── sample_submission.csv
│   ├── train/
│   └── test/
├── load_checkpoint/
│   └── densenet121-a639ec97.pth
├── outputs/
│   ├── baseline/
│   └── debug/
├── dataset.py
├── engine.py
├── inference.py
├── model.py
├── train.py
└── README.md
```

代码文件说明：

- `train.py`：参数设置、数据划分、训练、验证和测试主流程
- `inference.py`：加载训练好的模型并生成 Kaggle 提交文件
- `dataset.py`：读取 CSV、加载四通道图像、数据增强、数据划分和标签数量统计
- `model.py`：DenseNet121 四通道模型
- `engine.py`：单个 Epoch 的训练与验证、F1 计算、Smoke Test 和结果保存

## 准备运行环境

建议为项目创建独立的 Python 虚拟环境。代码主要依赖：

- PyTorch
- torchvision
- NumPy
- Pillow
- Matplotlib

请根据本机的操作系统和 CUDA 版本安装合适的 PyTorch。安装完成后，可以运行下面的命令检查环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果系统使用 `python3` 命令，请将本文命令中的 `python` 替换为 `python3`。

## 准备数据集

Kaggle 数据集和训练权重通常不直接提交到 Git 仓库。请先从比赛页面下载数据，并整理为下面的结构：

```text
dataset/
├── train.csv
├── sample_submission.csv
├── train/
│   ├── <image_id>_blue.png
│   ├── <image_id>_green.png
│   ├── <image_id>_red.png
│   └── <image_id>_yellow.png
└── test/
    ├── <image_id>_blue.png
    ├── <image_id>_green.png
    ├── <image_id>_red.png
    └── <image_id>_yellow.png
```

同一个样本的四个通道必须具有相同的图像尺寸和像素位置。代码按照下面的固定顺序读取：

```text
blue, green, red, yellow
```

训练 CSV 的 `Target` 是以空格分隔的类别编号，例如：

```csv
Id,Target
000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0,16 0
```

代码会将 `Target` 转换为长度为28的 multi-hot 向量。

## 准备预训练权重

代码不会自动下载预训练模型。请将 torchvision 的 DenseNet121 ImageNet 权重放在：

```text
load_checkpoint/densenet121-a639ec97.pth
```

也可以通过 `--pretrained-weights` 指定其他本地路径。

加载权重时出现下面的信息属于当前设计下的正常情况：

```text
Missing keys: 2, unexpected keys: 0
```

缺少的两个参数来自原始 ImageNet 分类器。本项目使用新的28类别分类头，因此不会加载原分类器权重。

## 运行方法

所有命令都需要在项目根目录执行。

### 1. Smoke Test

修改数据、模型或训练代码后，建议先进行快速检查：

```bash
python train.py --smoke-test --batch-size 2 --image-size 64 --device cpu
```

Smoke Test 只执行一个训练 batch 和一个验证 batch，用于检查：

- 四通道图像能否正常读取
- 模型前向传播是否正常
- Loss 是否能够计算
- 反向传播和参数更新是否正常

它不会完成完整 Epoch，也不会保存最优模型和验证报告。

### 2. 小数据 Debug 训练

调试完整训练、模型保存和结果输出流程时，可以只抽取少量数据：

```bash
python train.py --debug --debug-samples 1000 --epochs 1 --image-size 128 --batch-size 8
```

Debug 模式会：

- 使用固定随机种子抽取指定数量的有标签图片
- 将抽取的数据按照 70%/20%/10% 划分
- 执行完整的训练、验证、模型保存和结果统计流程
- 自动跳过 Kaggle 测试集推理
- 默认将结果保存到 `outputs/debug/`，避免覆盖正式训练结果

快速检查代码时可以使用200～500个样本，检查完整流程时可以使用1000～3000个样本。由于小数据可能不包含某些稀有类别，Debug 模式的 F1 不能代表正式模型效果。

### 3. 正式训练

使用默认参数训练：

```bash
python train.py
```

显式指定常用参数：

```bash
python train.py --epochs 10 --image-size 256 --batch-size 16 --learning-rate 0.0001 --threshold 0.5 --device auto
```

训练完成后，程序会：

1. 加载验证集 Macro-F1 最好的模型；
2. 保存验证集分类报告、预测概率和混淆矩阵；
3. 在 Holdout 数据上计算最终指标；
4. 默认预测 Kaggle 测试集并生成 `submission.csv`。

如果只训练和评估，不生成 Kaggle 提交文件：

```bash
python train.py --skip-submission
```

### 4. 单独运行推理

使用已经训练好的模型生成提交文件：

```bash
python inference.py --checkpoint outputs/baseline/best_model.pth --submission-path outputs/baseline/submission.csv --image-size 256 --batch-size 16 --threshold 0.5
```

推理时应使用与训练时一致的图像尺寸和预处理设置。程序会选择概率大于等于阈值的类别；如果没有任何类别超过阈值，则选择概率最高的类别，避免产生空预测。

## 数据处理与划分

当前基础增强包括：

- 随机水平翻转
- 随机垂直翻转

增强在四个通道堆叠后执行，因此四个通道始终应用相同的空间变换。

有标签数据使用固定随机种子划分为：

- Train：70%
- Validation：20%
- Holdout：10%

默认随机种子为42。Holdout 是从 Kaggle 训练数据中划出的本地留出集，不是 `dataset/test/` 中的 Kaggle 测试集。

当前版本采用普通随机划分，还没有实现多标签分层抽样。每次划分后会生成 `label_distribution.csv`，用于检查28个类别在 Train、Validation 和 Holdout 中的样本数量。

## 输出文件

正式训练结果默认保存在 `outputs/baseline/`，Debug 结果默认保存在 `outputs/debug/`。

```text
outputs/baseline/
├── best_model.pth
├── training_log.csv
├── label_distribution.csv
├── classification_report.csv
├── confusion_matrix_per_class.csv
├── confusion_matrices.png
├── val_predictions.npz
└── submission.csv
```

文件说明：

- `best_model.pth`：验证集 Macro-F1 最好的模型权重及训练信息
- `training_log.csv`：每个 Epoch 的 Train/Validation Loss、Macro-F1 和 Micro-F1
- `label_distribution.csv`：各类别在 Total、Train、Validation、Holdout 中的阳性样本数
- `classification_report.csv`：验证集每个类别的 Precision、Recall、F1 和 Support
- `confusion_matrix_per_class.csv`：验证集每个类别 one-vs-rest 的 TN、FP、FN、TP
- `confusion_matrices.png`：28个类别的 one-vs-rest 2×2 混淆矩阵
- `val_predictions.npz`：验证集图片 ID、真实标签、预测概率和预测阈值
- `submission.csv`：可上传到 Kaggle 的预测结果

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--data-dir` | `dataset` | 数据集目录 |
| `--output-dir` | `outputs/baseline` | 结果保存目录 |
| `--image-size` | `256` | 输入图像尺寸 |
| `--batch-size` | `16` | Batch Size |
| `--epochs` | `10` | 训练 Epoch 数量 |
| `--learning-rate` | `1e-4` | Adam 学习率 |
| `--num-workers` | `0` | DataLoader 工作进程数 |
| `--threshold` | `0.5` | 多标签预测阈值 |
| `--seed` | `42` | 随机种子 |
| `--device` | `auto` | 自动选择 CUDA 或 CPU |
| `--pretrained-weights` | `load_checkpoint/densenet121-a639ec97.pth` | 本地预训练权重路径 |
| `--debug-samples` | `1000` | Debug 模式抽取的样本数量 |

显存不足时，可以先减小 `--batch-size`，再根据需要减小 `--image-size`。

## 当前待改进内容

当前版本是便于理解和修改的基础实现，尚未加入以下优化：

- 多标签分层数据划分
- 针对类别不平衡的 Loss 或采样策略
- 每个类别独立阈值搜索
- 更强的数据增强
- 学习率调度器
- 更大的输入尺寸或多尺度训练
- TTA、模型集成和交叉验证

小组成员进行新实验时，建议同时记录运行参数、随机种子、验证集指标和 Kaggle 分数，并及时更新本 README。
