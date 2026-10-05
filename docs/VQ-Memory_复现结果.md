# VQ-Memory / RuleSafe 复现结果

论文：[Beyond Short-Horizon: VQ-Memory for Robust Long-Horizon Manipulation in Non-Markovian Simulation Benchmarks](https://arxiv.org/abs/2603.09513)（本地依据 arXiv 2603.09513v2）。本页整理截至 2026-08-29 的新服务器重建记录，不代表 2026-10-05 重新运行。

## 问题与方法

RuleSafe 使用多阶段保险柜任务测试机器人是否记得先前操作。单帧视觉可能无法区分“外观相同、任务阶段不同”的状态。VQ-Memory 对本体状态（关节等机器人内部状态）的历史窗口训练 VQ-VAE（向量量化变分自编码器），再把离散码本聚类为记忆 token，供 DP3 或 VLA 策略在后训练和推理时使用。

## 当前工程链路

1. 按论文附录 Table 5 整理 20 条规则，并生成每条规则 50 条、共 1,000 条演示。
2. 将演示整理为 220,000 个时间步、每步 1,024 × 6 点云与 13 维状态/动作的 DP3 Zarr 数据；并包含 rule、memory、plan、next-plan 条件字段。
3. 在 GPU7 短训 VQ tokenizer：window 50、stride 20、256 项码本、K-means 4 类、记忆长度 40。当前仅 50 步。
4. 用官方 DP3 读取链路加载数据，在 GPU7 做 50 步短训，保存普通及 EMA 检查点；8 个 batch 的评估均值 loss 约 0.01938。
5. 验证 HumanoidGen 公共保险柜资产在 SAPIEN 中加载、推进和关节读数；另构建三关节桥接件进行资产驱动冒烟。全量回归记录为 61 passed、9 skipped。

代码快照位于 `vq-memory/`。规则定义、数据生成、VQ tokenizer 和训练入口是**根据论文与开源上游独立重建**，并非作者发布的官方 VQ-Memory 仓库。

## 复现边界

- 当前成功率 1.0 对应**规则生成或演示冒烟**，不是机器人策略在论文 RuleSafe 基准的正式闭环成功率。
- DP3 的 50 步与 horizon 4 仅验证数据、前向、反向、EMA 和保存链路；论文 DP3 配置为 batch 256、200k 步、50 步动作块。
- 尚未获得作者同源的 RuleSafe 十类保险柜映射、H1-2 + Inspire hands + OMPL 专家轨迹，以及官方数据、模型与检查点。
- 尚未完成按论文规模训练 DP3，或对 RDT、CogACT、pi0 做同口径微调与闭环评测；因此不报告论文 Table 1–4 的同口径 SR（成功率）或 PS（过程分）。
- 旧服务器历史实验和当前新服务器短训应分开看；本文只按新服务器重建记录判断完成度。

下一步应先核实资产与专家轨迹可获得性，恢复真实接触演示和闭环评测，再做同条件的 DP3 无记忆 / 原始历史 / VQ-Memory 对照。未经这些步骤，训练 loss 不能代替策略成功率。
