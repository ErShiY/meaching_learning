# Human Protein Atlas Image Classification

本项目用四通道荧光显微图像预测蛋白质的亚细胞定位。每张图像可以有多个标签，模型输出 28 类预测。项目比较了 ResNet18 通道编码与融合、DenseNet121、SE-ResNeXt50，以及 Bestfitting DenseNet121 微调后的损失函数、阈值和少样本增强。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| [train.py](train.py) | 模型训练、留出评估和五折交叉验证 |
| [inference.py](inference.py) | 从训练权重导出一个或多个阈值的 Kaggle 提交文件 |
| [fold_manifest.py](fold_manifest.py) | 生成固定的多标签分层五折 |
| [models/](models/) | 模型结构和通道融合、注意力、池化模块 |
| [scripts/](scripts/) | 权重转换、实验批量训练和提交导出 |
| [tests/](tests/) | 数据划分、模型和实验流程的单元测试 |

## 环境与数据

在项目根目录执行下列命令。原实验使用 Python 3.12、PyTorch 2.5.1、torchvision 0.20.1 和 CUDA 12.1。批量实验脚本使用 CUDA。可按设备选择匹配的 PyTorch 和 torchvision 后再安装其余依赖。 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) · [原实验版本的安装命令](https://pytorch.org/get-started/previous-versions/#v251)

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
~~~

从比赛数据页下载原始数据并按以下目录解压。每张图像对应四个 PNG 文件，文件名为 `<Id>_blue.png`、`<Id>_green.png`、`<Id>_red.png` 和 `<Id>_yellow.png`。 [Kaggle 比赛数据](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/data)

~~~text
dataset/
├── train.csv
├── sample_submission.csv
├── train/
└── test/
~~~

使用原始 `train.csv` 和种子 42 生成实验使用的五折划分。程序将 `fold_manifest.json` 写到 `dataset/`。

~~~bash
python fold_manifest.py --data-dir dataset --seed 42
~~~

## 预训练权重

按实验需要下载权重并放到对应路径。

| 权重 | 本地路径 | 下载来源 |
| --- | --- | --- |
| DenseNet121 ImageNet | `load_checkpoint/densenet121-a639ec97.pth` | [Torchvision 权重](https://download.pytorch.org/models/densenet121-a639ec97.pth) |
| ResNet18 ImageNet | `load_checkpoint/resnet18-f37072fd.pth` | [Torchvision 权重](https://download.pytorch.org/models/resnet18-f37072fd.pth) |
| SE-ResNeXt50 ImageNet | `load_checkpoint/SE/pytorch_model.bin` | [timm RACM 权重](https://huggingface.co/timm/seresnext50_32x4d.racm_in1k/resolve/1b040f7c2d6e6b4d4a1467b29a25e62c58f817bc/pytorch_model.bin) |
| Bestfitting HPA | `load_checkpoint/bestfitting-densenet121.pth` | [Bestfitting v0.1.0 发布页](https://github.com/CellProfiling/densenet/releases/tag/v0.1.0) |

从 Bestfitting 发布页下载 `external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds_fold0_final.pth` 到项目根目录，再运行转换脚本。转换器会校验原文件的 SHA-256。

~~~bash
python scripts/convert_bestfitting_weights.py external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds_fold0_final.pth load_checkpoint/bestfitting-densenet121.pth
~~~

## 最终单模型

最终提交采用 Bestfitting DenseNet121、ASL、普通增强、20 个 epoch，Kaggle 推理阈值固定为 `0.70`。以下示例从公开 HPA 权重独立训练 Fold 1，直接运行这一配置只需要 Bestfitting 权重。

~~~bash
python train.py --model bestfitting_densenet121 --pretrained-weights load_checkpoint/bestfitting-densenet121.pth --fold 1 --epochs 20 --image-size 512 --batch-size 8 --num-workers 4 --device cuda --loss asl --augmentation basic --optimizer adam --learning-rate 0.00001 --seed 42 --output-dir outputs/bestfitting_final
~~~

训练结束后，将下列命令中的 `RUN_TIMESTAMP` 换成实际运行目录名。每折都会生成自己的 `final_model.pth`，推理时选择对应折的文件。

~~~bash
python inference.py --checkpoint outputs/bestfitting_final/cross_validation/fold_1/RUN_TIMESTAMP/final_model.pth --submission-dir outputs/bestfitting_final/submissions/fold_1 --thresholds 0.70 --device cuda --batch-size 8 --num-workers 4
~~~

`--fold` 接受 `1` 到 `5`。传入 `--fold all` 会顺序训练五折，在 `outputs/bestfitting_final/cross_validation/RUN_TIMESTAMP/` 下保存 `fold_1` 到 `fold_5` 以及 `cv_summary.json`。本地留出指标采用训练命令的阈值，默认 `0.50`。公开 HPA 权重的原训练集可能覆盖本地留出样本。要评估模型结构在本地五折上的表现，可把训练命令的初始化权重改为 DenseNet121 ImageNet 权重。

## 批量复现实验

准备好表中的四种权重与固定折清单后，先运行 Fold 1 实验脚本。它会生成后续续跑脚本需要的配置和权重记录。

~~~bash
python scripts/run_fold1_experiments.py
~~~

| 命令 | 用途 |
| --- | --- |
| `python scripts/export_asl_thresholds.py` | 从已训练的 Fold 1 ASL 模型补导出阈值对照，包括最终的 `0.70` 提交 |
| `python scripts/run_backbone_folds.py` | 续跑普通 DenseNet121 和 SE-ResNeXt50 的 Fold 2–5 |
| `python scripts/run_bestfitting_folds.py` | 续跑最终 ASL 配置的 Fold 2–5，每折导出 `submission_t0.70.csv` |

训练日志、权重、留出预测和待上传的 CSV 会写到 `outputs/`。批量训练脚本可通过 `--dry-run` 查看命令。中断后用相同命令重跑，可以复用已经完成的折。

## 实验说明

- [DenseNet 多尺度标签注意力与 GeM](docs/DENSENET_EXPERIMENT.md)
- [DenseNet 残差特征增强](docs/DENSENET_REFINE.md)
- [分类头与损失函数消融](docs/ABLATION.md)
- [Bestfitting 原始分类方案](https://github.com/CellProfiling/HPA-competition-solutions/tree/master/bestfitting) · [赛事分析论文](https://www.nature.com/articles/s41592-019-0658-6)
