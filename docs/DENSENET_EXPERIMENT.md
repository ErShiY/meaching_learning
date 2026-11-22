# DenseNet 多标签改进方案

## 模型：`densenet_attention`

保留 Baseline 的四通道卷积初始化和整个 ImageNet 预训练 DenseNet121，
所有 Encoder 参数继续训练。聚合和分类改成两条互补分支：

```text
四通道 [B,4,512,512]
  -> DenseNet121
     denseblock3：[B,1024,32,32] ----+
     norm5/ReLU：[B,1024,16,16] ----+-> 分别投影到256维 -> 多尺度空间token
                                   -> 28个可学习标签query的交叉注意力
                                   -> 每标签的MLP特征变换与独立读出

     norm5/ReLU -> 可学习GeM池化 -> LayerNorm/MLP/Dropout -> 全局logits

最终logits = 全局logits + 可学习系数 * 标签注意力logits
```

标签注意力对空间位置进行归一化，28个标签不互斥；模型不输出 Sigmoid/Softmax
类别概率。Loss 直接使用 `[B,28]` logits。没有硬编码标签共现关系。
GeM 的指数从3开始学习，并限制在1到6之间；它让模型学习整体平均与局部强响应
之间的取舍。新模型约8.30M参数，旧 `baseline` 与所有旧 ResNet Checkpoint 保留。

标签query池化借鉴 [Query2Label](https://arxiv.org/abs/2107.10834)，
GeM参考 [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)。
这里是任务适配的实验组合，不是论文完整复现，也不能预先保证涨分。

## Loss 和优化

- `--loss asl`：正标签 gamma=0、负标签 gamma=4、负概率 clip=0.05。
  减小容易负标签的影响；不等于类别频次加权，也不会增加稀有正样本。
- ASL 使用稳定的 log 概率计算，默认将 focal 权重 detach，取整个 batch/标签的均值。
  因此 ASL 的绝对 Loss 数值不能与旧 BCE Loss 大小比较。
- `--optimizer adamw`：默认 weight decay=1e-4；bias、归一化参数和标量参数不衰减。
- `--scheduler warmup_cosine`：每个优化步更新；默认前2个 epoch 从1e-5升到1e-4，
  后续余弦下降，最后一次更新达到 `--min-lr 1e-6`。
- 默认仍是 BCE、Adam、无调度，以便复现原始 Baseline。

ASL公式参考[作者实现](https://github.com/Alibaba-MIIL/ASL)。

## 数据与划分

`--augmentation microscopy` 保留原翻转，再加入四通道同步的90度旋转，
以及共享的轻度 gain/gamma（0.9到1.1）。不裁剪，不打乱荧光通道顺序，
不对验证、Holdout 或 Kaggle 测试图片应用随机增强。空间对应关系保持一致。
这属于普遍增强，不是专门过采样稀有类别。

`--split-method stratified` 使用稀有标签优先的迭代多标签分层，按每个类别的
期望配额，把样本分配给最需要该标签且仍有容量的集合；一个样本的所有标签
一起计入配额。保持精确的70/20/10总大小，标签比例尽量接近，不能保证稀有
或强共现类别完全等比例。它不使用生物样本分组元数据，不能替代 group split。

全量31,072个样本、seed42的稀有类别分布：

| 类别 | 旧 Train/Val/Holdout | 分层 Train/Val/Holdout |
| --- | --- | --- |
| 27 | 9/0/2 | 8/2/1 |
| 15 | 15/3/3 | 15/4/2 |
| 10 | 18/8/2 | 20/5/3 |

每次运行保存 `split_manifest.json`（各集合图片ID）、`label_distribution.csv`
和配置中的 split SHA256。后续用 `--split-file <manifest路径>` 精确复用；即使
CSV顺序变化，也按ID加载，并拒绝漏项、重叠、重复或未知ID。

改用分层后，验证集已改变，不能把新结果与旧 `0.631594` 直接作公平对比。
应在同一分层上重跑原始 DenseNet 对照。极稀有类只有1到4个验证样本，指标
仍会不稳定；增强、Loss和分层都不能凭空解决数据不足。

## 运行

完整新方案（项目已有DenseNet本地权重）：

```powershell
python train.py `
  --model densenet_attention `
  --epochs 20 --image-size 512 --batch-size 8 --num-workers 4 `
  --loss asl --optimizer adamw --weight-decay 0.0001 `
  --scheduler warmup_cosine --warmup-epochs 2 --min-lr 0.000001 `
  --split-method stratified --augmentation microscopy `
  --learning-rate 0.0001 --seed 42 --threshold 0.5
```

默认在训练结束生成 Kaggle submission；暂时只做本地评估时加 `--skip-submission`。
结果位于 `outputs/densenet_attention/<时间戳>/`。
ASL 的概率校准可能不同，先记录固定阈值0.5指标，再只在验证集上选阈值。
不要直接照搬旧模型的最佳阈值，也不要用 Holdout 反复调参。

同一新划分上的原始模型对照（用实际运行目录替换示例路径）：

```powershell
python train.py --model baseline --epochs 20 --image-size 512 --batch-size 8 --num-workers 4 --split-file outputs/densenet_attention/<时间戳>/split_manifest.json --output-dir outputs/baseline_stratified --skip-submission
```

如果要区分训练方案与模型结构的收益，可在原始 `baseline` 上使用同样的
ASL/AdamW/调度/增强选项，再与新模型比较；不需要把每个小模块单独跑一遍。

快速验证：

```powershell
python -m unittest discover -s tests -v
python train.py --model densenet_attention --smoke-test --epochs 20 --loss asl --optimizer adamw --scheduler warmup_cosine --split-method stratified --augmentation microscopy --num-workers 0
```

新 Checkpoint 保存训练配置，独立推理默认读取其中的图片尺寸和阈值；仍可通过
命令行覆盖。旧 Checkpoint 缺少配置时兼容旧默认值（256和0.5），应显式传入
其实际训练尺寸，如 `--image-size 512`。
