# DenseNet 残差特征增强实验

实现位于 `models/densenet_refine.py`，模型名为 `densenet_refine`。
原 `densenet_attention.py`、`heads.py`、Dataset 和训练流程保持原实现。
新模型继续复用原标签空间注意力分类头和 GeM。

## 特征路径

以512输入为例：

```text
C2 = DenseBlock2： [B, 512,64,64]
C3 = DenseBlock3： [B,1024,32,32]
C4 = norm5/ReLU：  [B,1024,16,16]

L2/L3/L4 = 各自1x1投影到256通道 + GroupNorm + GELU

F3 = Refine(w2*Down(L2) + w3*L3 + w4*Up(L4))
F4 = Refine(v4*L4 + v3*Down(F3))

Refine = Depthwise3x3 -> Pointwise1x1 -> GroupNorm/GELU -> CBAM

增强C3 = ReLU(C3 + alpha * OutputProjection(F3))
增强C4 = ReLU(C4 + beta  * OutputProjection(F4))

现有 LabelAttentionHead(增强C3, 增强C4) -> 28 logits
```

`alpha`、`beta` 从0.1开始训练，保留原特征作为残差路径。
跨尺度权重经过Softmax，是全局可学习的尺度权重；CBAM的通道/空间权重则
根据每张图的特征计算。这里的特征通道不是原始四种荧光通道。
C2跳连补充较早的特征，上采样C4只用于语义指导，不会恢复丢失的原图细节。
下采样使用目标尺寸的自适应平均池化，上采样使用双线性插值。

新模型约9.64M参数，其中新增neck约1.34M。该结构是受
[FPN](https://arxiv.org/abs/1612.03144)和
[CBAM](https://arxiv.org/abs/1807.06521)启发的任务适配方案，非论文完整复现。
代码可训练不代表已经验证涨分。

## 同条件正式实验

默认与上一版一样，从本地ImageNet DenseNet121权重初始化，训练20轮。
先完成一轮 [DenseNet Attention 训练](DENSENET_EXPERIMENT.md)，将命令中的
`RUN_TIMESTAMP` 换成其运行目录名。复用该实验划分以及同一套 ASL、增强、优化器和学习率配置。

```powershell
python train.py `
  --model densenet_refine `
  --epochs 20 --image-size 512 --batch-size 8 --num-workers 4 `
  --loss asl --optimizer adamw --weight-decay 0.0001 `
  --scheduler warmup_cosine --warmup-epochs 2 --min-lr 0.000001 `
  --split-file outputs/densenet_attention/RUN_TIMESTAMP/split_manifest.json `
  --augmentation microscopy --learning-rate 0.0001 --seed 42 `
  --skip-submission
```

结果单独写到 `outputs/densenet_refine/<时间戳>/`。
先跳过自动提交：ASL模型不应直接用默认0.5生成最终提交。
完整训练后只在该模型的验证预测上重新选阈值，再用冻结阈值确认效果。
原模型的0.715不是新模型已知的最佳阈值；不能用Holdout或Kaggle分数反复搜索。
当前训练器的Checkpoint仍按命令行固定阈值的Val Macro-F1选取，本实验未改变
该选择规则；最终阈值校准不会追回此前未保存的其他epoch权重。

## 可选：加载当前已训练的模型

新文件也支持通过现有参数指定完整的 `densenet_attention` checkpoint：

```powershell
python train.py `
  --model densenet_refine `
  --pretrained-weights outputs/densenet_attention/RUN_TIMESTAMP/best_model.pth `
  --epochs 10 --learning-rate 0.00003 `
  --loss asl --optimizer adamw --scheduler warmup_cosine --warmup-epochs 1 `
  --split-file outputs/densenet_attention/RUN_TIMESTAMP/split_manifest.json `
  --augmentation microscopy --skip-submission
```

这会加载旧模型全部Encoder和分类头参数，新neck随机初始化。
加载前严格检查参数键和形状，只允许新neck的参数缺失。
也支持完整 `densenet_refine` checkpoint的严格权重初始化。
此功能只初始化权重，不恢复epoch、优化器或调度进度。
由于额外利用了旧模型的训练预算，应将其标记为微调实验，不能与从ImageNet
开始的20轮实验混为同预算比较。

## 检查

```powershell
python -m unittest discover -s tests -v
python train.py --model densenet_refine --smoke-test --image-size 512 --batch-size 8 --num-workers 0
```

验证覆盖三层特征和注意力的梯度、残差归零时的原模型等价性、非方形空间对齐、
初始化错误拒绝、Checkpoint往返和原模型checkpoint兼容性。
