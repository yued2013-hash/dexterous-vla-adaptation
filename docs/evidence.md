# 实现验证与实验记录

## 数据与模型检查

| 检查项 | 实测记录 | 对应材料 |
| :-- | :-- | :-- |
| 数据规模 | selected66：66段、23,283帧、10Hz；全量66段训练配置 | 数据审计与DINO审计 |
| 状态/动作来源 | feedback与teleoperation command分离，数值回映差约3.58e-7，相同行比例0 | 来源审计 |
| RGB时间对齐 | P95约16.17ms；136帧超过50ms，20帧超过100ms | 时间对齐分布 |
| DINO零门控 | 固定噪声forward输出最大差0.0 | 模型热启动审计 |
| 模型检查 | 历史模型审计记录32项测试通过、forward/backward有限 | DINO审计文件 |
| 阶段1检查点 | 18,200步checkpoint路径、哈希及反向记录 | 训练记录 |
| 参考实现检查 | 固定锚点、旋转转换、有效计数归约与位姿/力接口测试 | `reference_validation.json` |
| 补丁重建 | 5份SmolVLA补丁实际应用检查；换行规范化后逐文件重建目标源码 | `patch_validation.json` |
| 实机展示 | 约26秒机械臂与灵巧手桌面物体操作片段 | `docs/media/demo.mp4` |

## 记录组织

`evidence/`保留模型和数据的审计数值，`source_provenance.json`记录原文件SHA，`reference_validation.json`记录接口参考测试结果，`patch_validation.json`记录补丁对固定基础快照的重建检查。

selected66的数据与训练配置按同一分支组织；其它数据规模的配置保留各自来源标记。公开记录通过文件哈希、检查点步数和配置路径关联，便于从结果追踪到具体实现。
