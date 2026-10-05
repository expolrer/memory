# DySta / SD-VLA 复现结果

论文：[Static-Dynamic Disentanglement for Efficient Multi-Frame Vision-Language-Action Models](https://arxiv.org/abs/2602.03983)（本地依据 arXiv 2602.03983v3）。本页整理截至 2026-09-02 的实验记录，不代表 2026-10-05 重新运行。

## 方法与实现

DySta 将视觉 token 拆为两级静态区域和动态区域：每帧 256 个 token 中，L1 静态 133、L2 静态 107、动态 16。两帧输入保留一份静态 token 和两份动态 token，视觉上下文为 `133 + 107 + 2 × 16 = 272`。静态 token 使用 InfoNCE 对比损失约束时间稳定性；门控网络决定是否刷新静态区域，L1 刷新时级联刷新 L2。目标是在策略训练后复用静态前缀的 KV cache，减少动作查询时的重复计算。

本地代码在 `dysta/src/dysta/`；训练和评测入口在 `dysta/scripts/`；论文参数及实验记录在 `dysta/configs/paper.yaml`。OpenVLA-OFT、LIBERO 和基础模型均为外部依赖，不在本仓库分发。

## 已完成

- 使用 OpenVLA-7B 基础权重与四套官方 LIBERO RLDS（去除空动作）训练 Spatial、Object、Goal、Long。
- 每套训练 15,000 个**优化器更新步**，两张 H100 80GB；每卡 micro-batch 4、梯度累积 8、全局 batch 64。
- LoRA rank 32、BF16、AdamW、学习率 5e-4、seed 42；训练时保存 LoRA、动作头、DySta adapter、处理器和数据统计。
- 完成单元测试、真实双卡前向/反向、RLDS 历史索引验证、LIBERO 无头渲染冒烟和正式闭环评测。
- 四套任务均为 10 个任务 × 每任务 50 回合；严格汇总器检查任务 ID、回合数和任务套件。

| 套件 | 物理训练 GPU | 优化器步 | 成功 / 回合 | 成功率 | Wilson 95% 区间 |
| --- | --- | ---: | ---: | ---: | --- |
| Spatial | 6, 7 | 15,000 | 484/500 | 96.8% | 94.87%–98.02% |
| Object | 6, 7 | 15,000 | 480/500 | 96.0% | 93.90%–97.40% |
| Goal | 0, 7 | 15,000 | 477/500 | 95.4% | 93.19%–96.92% |
| Long (r2) | 6, 7 | 15,000 | 339/500 | 67.8% | 63.58%–71.75% |
| **合计** | | | **1780/2000** | **89.0%** | |

Long 的第一次运行 r1 在第 4,175 步遭外部 SIGKILL。由于检查点没有保存 AdamW、调度器和数据迭代器状态，r2 从 OpenVLA-7B 基础权重重新训练完整 15,000 步。Long r2 已训练并评测完毕，不能把从其 15,000 步权重继续训练称为论文配置内的续训。

## 资源记录

Spatial 15,000 步约 11.42 小时墙钟、22.84 H100 GPU 小时。单卡 PyTorch 峰值 allocated 约 31.7 GiB；实际进程显存还包括框架预留与其他开销。四套训练均为类似双卡规模，具体环境和数据加载会影响耗时。

## 已知差距

1. Long 仅 67.8%，与论文总体 LIBERO 结果存在明显差距；目前不能宣称论文总体成功率已对齐。
2. 尚无**相同起点、数据、训练预算**的无 DySta 对照，因此成功率不能单独证明 DySta 带来提升。
3. 缓存裁剪/重算核心已实现，但跨控制时刻的完整 KV cache 复用未在策略 rollout 中验证；未复现论文 FLOPs、437 ms 延迟或 1.70× 加速。
4. CogACT/SimplerEnv、LIBERO-Memory 和真实机器人实验未完成。
5. 论文没有公开可发现的完整 DySta 实现、检查点或 LIBERO-Memory 资产；两帧历史、InfoNCE temperature 0.07、gate 先验 lambda 0.1 和固定 token 区间等是重建假设。

复查时应以每套 500 回合的逐任务日志、严格汇总输出、最终检查点及配置为准。大型原始证据仍在实验服务器；本仓库仅保存文本结果与代码快照。
