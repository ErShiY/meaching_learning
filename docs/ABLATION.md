# DenseNet 分类头与损失函数消融

这个入口调用原有 `train.py`，不修改 Baseline、Attention、数据集或训练默认逻辑。
四组只改变模型分类头与 Loss，其他设置从已完成实验复制。

| 实验名 | 模型 | Loss | 默认行为 |
| --- | --- | --- | --- |
| `baseline_bce` | 原始 DenseNet：GAP＋Linear | BCE | 新训练 |
| `baseline_asl` | 原始 DenseNet：GAP＋Linear | ASL | 新训练 |
| `attention_bce` | DenseNet：多尺度标签 Attention＋GeM | BCE | 新训练 |
| `attention_asl` | DenseNet：多尺度标签 Attention＋GeM | ASL | 复用已完成结果 |

参考目录来自一次完整的 [Attention＋ASL 训练](DENSENET_EXPERIMENT.md)。
原参考实验使用 ImageNet 初始化、512 输入、batch 8、20 epoch、seed 42、AdamW、
weight decay 1e-4、Warmup 2 epoch＋Cosine、学习率 1e-4→1e-6、
microscopy 增强以及固定分层划分。ASL 参数为 gamma_pos=0、gamma_neg=4、clip=0.05。
预训练 Encoder 全部参与训练，各组均从同一份 ImageNet 权重开始，
不会从已训练 Attention 权重继续训练。

原始 Baseline 的旧训练划分、优化器和增强不同，因此这里重新训练
`baseline_bce`，不拿旧 Kaggle 成绩替代公平对照。

## 使用

在项目根目录和已安装依赖的 Python 环境中执行。
首次运行时用 `--reference-run` 指向完成训练的 Attention＋ASL 目录，将
下面命令中的 `RUN_TIMESTAMP` 换成实际目录名。

先查看命令与设置，不创建目录、不启动训练：

```powershell
python ablation.py run --reference-run outputs/densenet_attention/RUN_TIMESTAMP --dry-run
```

完整四组矩阵：顺序训练三组，复用已有 Attention＋ASL。
单组保持原来的 tqdm batch 进度条，组与组之间自动切换；默认不生成 Kaggle submission。

```powershell
python ablation.py run --reference-run outputs/densenet_attention/RUN_TIMESTAMP --suite-dir outputs/ablation/head_loss_v1
```

如果只想先补两组，使用另一个目录：

```powershell
python ablation.py run --reference-run outputs/densenet_attention/RUN_TIMESTAMP --suite-dir outputs/ablation/head_loss_minimal --experiments baseline_asl attention_bce
```

若希望连 Attention＋ASL 也全部重新训练：

```powershell
python ablation.py run --reference-run outputs/densenet_attention/RUN_TIMESTAMP --suite-dir outputs/ablation/head_loss_all_new --experiments baseline_bce baseline_asl attention_bce attention_asl
```

可以只跑一个实验，例如 `--experiments baseline_asl`。
未选择的普通组会标为 `not_selected`；未选择重新训练的 `attention_asl` 会复用参考结果。
同一目录中的实验清单一旦建立就冻结；要更换组合，用新的 `--suite-dir`。

中断后继续执行同一目录：

```powershell
python ablation.py run --suite-dir outputs/ablation/head_loss_v1
```

已完成的组会被校验后跳过。未完成的组会在新时间戳目录从第 1 epoch 重新训练，
保留中断产物；这不是 epoch 断点续训，原 Checkpoint 没有保存优化器状态。

省略 `--suite-dir` 时自动使用 `outputs/ablation/<时间戳>/`。
使用其他完成的 Attention＋ASL 作为参考时，首次运行传入
`--reference-run <该运行目录>`；训练设置会整体跟随该参考，不提供单组超参数覆盖。
首次运行可用 `--device cuda:0` 指定设备；续跑直接读取已冻结设置。

## 结果与阈值

每完成一组，自动更新同一套验证集上的比较结果：

```text
outputs/ablation/head_loss_v1/
  suite.json                    # 冻结配置、命令、各组状态与实际运行目录
  split_manifest.json           # 固定的 Train / Val / Holdout 图片 ID
  summary.csv / .json / .md      # 汇总表；CSV/JSON 的 run_dir 指向实际模型目录
  baseline_bce/<时间戳>/         # 原 train.py 产物：模型、日志、验证预测等
  baseline_asl/<时间戳>/
  attention_bce/<时间戳>/
  reports/<实验名>/              # 阈值扫描、逐类结果、可选 Holdout 缓存
```

`attention_asl` 的原模型与结果保留在参考目录，汇总表直接引用它们。
每组保存独立路径，已有实验不会被覆盖。脚本检查完成轮数、checkpoint 选择记录、
训练配置和划分；续跑时还检查训练 CSV 和 ImageNet 权重的 SHA256。

只刷新比较表，不训练、不加载 GPU 模型：

```powershell
python ablation.py report --suite-dir outputs/ablation/head_loss_v1
```

验证集同时报告：

- 固定阈值 0.5 的 Macro-F1、Micro-F1。
- 每组各自在验证集上选择全局阈值后的 Macro-F1、Micro-F1。
- 最佳阈值、最佳 epoch，以及逐类 F1 和 support。

阈值搜索对所有组统一为 0.05～0.995、步长 0.005，并列时取较小阈值。
所有组仍按 **验证集阈值 0.5 的 Macro-F1** 选择 checkpoint，
随后只校准这一个 checkpoint；没有改变已有实验的选模规则。
本地评估沿用原 `engine.py`：直接按阈值二值化，不强制至少预测一个标签；
Kaggle 推理原有的无标签时 argmax 回退保留，因此二者解码规则存在这一差别。

默认汇总已保存的 Holdout@0.5；需要校准阈值下的 Holdout 成绩时运行：

```powershell
python ablation.py report --suite-dir outputs/ablation/head_loss_v1 --evaluate-holdout --device cuda --num-workers 4
```

这一步会读取各组最佳模型，对同一 Holdout 推理，并把概率缓存到实验套件目录，
复用参考组也一样。只应用验证集选好的阈值，**不在 Holdout 上搜索阈值**。
首次需要一次额外推理，之后可使用缓存。原运行目录和 checkpoint 不会被改写。

消融期间不必逐组生成 Kaggle submission。需要提交某组时，用 `suite.json`
或汇总表中的实际 `run_dir` 找到 `best_model.pth`，再执行原 `inference.py`，
显式设置该组验证集选出的阈值，而不是所有模型都使用 0.715。

## 结论边界

- 同一行比较 BCE／ASL，检验该分类头条件下损失函数的影响。
- 同一列比较原始头／新头，检验该损失条件下整个聚合与分类头组合的影响。
- 新头包含多尺度、标签 Attention、GeM 与全局分支，不能只把整体收益归因于 Attention。
- 这组实验不隔离通道融合方式、数据增强或分层划分的收益。
- 不直接比较 BCE 和 ASL 的 Loss 数值；重点比较相同协议下的 F1。
- 单 seed 和极少量稀有类样本仍有不确定性，结果无论提升或下降都应保留。
