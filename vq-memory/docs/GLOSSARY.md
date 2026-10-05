# Glossary

| Term | Chinese explanation |
| --- | --- |
| VQ-Memory | 向量量化记忆模块，把历史关节状态编码成离散记忆 token。 |
| RuleSafe | 规则保险柜基准，论文提出的长程非马尔可夫机器人操作任务集合。 |
| Benchmark | 基准测试，用统一任务和指标评估模型。 |
| Non-Markovian | 非马尔可夫，当前观察不足以决定动作，必须依赖历史信息。 |
| Long-horizon manipulation | 长程操作，需要多个连续步骤才能完成的机器人任务。 |
| VLA | 视觉-语言-动作模型，输入图像和语言指令，输出机器人动作。 |
| Proprioceptive state | 本体感知状态，机器人自身关节角、位姿、手爪状态等内部状态。 |
| Joint state | 关节状态，机器人每个关节的位置或角度。 |
| VQ-VAE | 向量量化变分自编码器，把连续序列压缩为离散码并重建。 |
| Codebook | 码本，VQ-VAE 中可学习的一组向量。 |
| Token | 离散符号，模型可处理的编号或嵌入。 |
| K-means | K 均值聚类，把相近向量合并成少量类别。 |
| Memory length | 记忆长度，策略输入中包含多少个历史记忆 token。 |
| Action chunk | 动作块，模型一次预测未来多步动作。 |
| Diffusion policy | 扩散策略，用扩散生成过程产生动作序列。 |
| Flow matching | 流匹配，一类生成模型训练方法，pi0 使用类似范式。 |
| Success Rate | 成功率，完整完成任务的比例。 |
| Process Score | 过程分数，中间步骤正确完成的比例。 |
| SAPIEN | 机器人物理仿真平台。 |
| OMPL | 开源运动规划库，用于规划机械臂路径。 |
| HumanoidGen | LLM 辅助的人形/双臂机器人数据生成框架。 |
