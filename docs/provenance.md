# 代码来源与固定版本

- LeRobot/SmolVLA基础来源：https://github.com/huggingface/lerobot 。
- DexGraspVLA基础来源：https://github.com/Psi-Robot/DexGraspVLA 。
- DINOv2基础来源：https://github.com/facebookresearch/dinov2 。

提供的LeRobot基础快照含版本标记`07285677a3ef2f287a05c271f4b668f20e2ce57f`；DINO分支审计标记其本地基础为`736299d607882176391b97be24edd616d53864cb`；DexGraspVLA适配的本地工作树基础记录为`4be75b80b92119b8d5b0e59d67ae5f5e5e5f8a53`。这些版本分别对应基础快照与适配工作树，同时通过实际文件SHA和基础快照差异清单追踪实现。

JC-parity接口、社区基础模型以及selected66、DINO和位姿/力适配分别保留来源记录，公开材料使用脱敏路径与文件哈希进行关联。

LeRobot相关模块/补丁保留Apache-2.0版权与授权文件，补丁以提供的`lerobot-source`文件快照为基准，而非任意最新版。DexGraspVLA本地包未附授权文件，公开上游仓库元数据也未声明许可证，因此本仓库只整理其接口事实、来源哈希及独立数学参考，不重新打包该训练框架。

未打包原始示教数据、机器人SDK、设备地址、模型权重或完整训练输出。实机操作视频经过浏览器兼容转码并移除音轨，保留完整操作顺序。
