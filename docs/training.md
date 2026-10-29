# 训练配置与分布式工程记录

## 分支和阶段

| 分支 | 保存的配置 | 完成证据 |
| :-- | :-- | :-- |
| SmolVLA selected66 JC-parity | 8进程、每进程batch8、300epoch/109200step | 配置和审计；DINO热启动审计记录了基础检查点109200及哈希，本轮没有完整权重/日志供复跑 |
| DINO阶段1 | 冻结基础模型与DINO，训练适配器/交叉注意力，50epoch | 历史审计记录018200检查点及哈希、有限梯度与3155760个可训练参数 |
| DINO阶段2 | 动作专家+稠密视觉模块联合微调100epoch | 配置与一次真实forward/backward记录，未提供阶段2最终训练完成证据 |
| DexGraspVLA位姿–力 | 8进程BF16、400epoch配置、25D与5D分离 | 实现/配置存在；没有对应完整训练曲线、最终权重或任务对照结果 |

selected66共有23283帧，global batch64，每epoch364个DDP更新；为均匀batch每epoch存在13个重复样本位置。数据迭代、padding和loss的有效元素计数需要一起审查，不能仅比较配置中的总loss。

## DDP有效计数

对于不同rank的局部分子S和有效计数N，DDP会平均梯度。原实现只all-reduce detached计数，并按`world_size / sum(N)`缩放局部可微分子；这样经过DDP平均后的梯度对应全局有效元素均值。

`reference/training_math.py`提供公式参考，测试包含不相等计数的反例。当前参考测试不启动torch.distributed，不能视为多GPU梯度数值一致性验收。

## DataLoader退出问题

历史八卡A800排查记录对比0/2/8 workers、persistent开关和训练长度。一次旧运行在输出End of training后停留超过2分钟；跟踪后的运行表明各rank能够通过barrier与end_training，问题与persistent fork/PyAV worker清理有关，不能简单归因为缺失分布式barrier。

选择`num_workers=8, persistent_workers=false`后，记录中的330step运行跨过325step epoch边界，并在End of training后不足5秒自然退出。epoch边界重新创建workers有短暂开销；这一记录只支持退出可靠性排查结论，不作为训练加速百分比。

该历史记录未写诊断checkpoint；本轮整理未启动GPU训练。完整机器路径、端口和设备地址不纳入公开材料。

## 依赖边界

历史SmolVLA训练环境报告Python3.12.13、PyTorch2.11.0+cu130、Accelerate1.14.0、Transformers5.5.4；不要把它与DexGraspVLA的另一套环境直接混装。

本仓库的轻量参考只依赖NumPy与SciPy。PyTorch模块和历史模型测试需要匹配的完整上游环境、模型权重和示教数据；这些不包含在公开项目中。
