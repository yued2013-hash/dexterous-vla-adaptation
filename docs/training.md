# 训练配置与分布式工程

## 训练阶段与配置

| 分支 | 参数设置 | 关联记录 |
| :-- | :-- | :-- |
| SmolVLA selected66 JC-parity | 8进程，每进程batch8，global batch64，300epoch / 109200step配置 | 数据与训练审计、基础检查点109200及哈希 |
| DINO阶段1 | 冻结基础模型与DINO，训练适配器和交叉注意力；50epoch配置 | 18,200步checkpoint、3,155,760个可训练参数、有限梯度检查 |
| DINO阶段2 | 动作专家与稠密视觉模块联合微调；100epoch配置 | 配置文件与真实forward/backward检查记录 |
| DexGraspVLA位姿–力 | 8进程BF16、400epoch配置；25D控制与5D力分别编码 | 训练实现、分组归一化与分布式配置 |

selected66包含23,283帧，global batch64，每epoch364个DDP更新。均匀batch组织每epoch产生13个重复样本位置，数据padding与损失有效计数在同一训练接口处理。

## DDP有效计数归约

按xyz、rotation6D与hand分组组织损失，每组累计局部分子S与有效标量数N。跨rank对detached计数all-reduce，并将局部可微分子乘 `world_size / sum(N)`；经过DDP梯度平均后，得到全局有效元素均值对应的梯度。

[`reference/training_math.py`](../reference/training_math.py)给出公式参考，测试覆盖各rank有效计数不相等的情况。

## DataLoader与训练退出

在八卡A800排查中，对比0/2/8 workers、persistent开关及训练长度，跟踪rank的barrier、end_training和PyAV worker清理路径。最终采用 `num_workers=8, persistent_workers=false`：330步运行跨过325步epoch边界，End of training后不足5秒自然退出。

该配置通过epoch边界重建worker，完成数据加载生命周期与分布式训练退出的衔接。

## 环境组织

SmolVLA训练环境记录Python3.12.13、PyTorch2.11.0+cu130、Accelerate1.14.0、Transformers5.5.4。SmolVLA和DexGraspVLA分别保留环境配置与基础模型依赖。

接口参考环境依赖NumPy/SciPy。模型模块使用对应上游框架、权重和示教数据运行；配置中的 `${LOCAL_ARTIFACTS}` 指向本地资源目录。
