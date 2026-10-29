# 代码来源与固定版本

- LeRobot/SmolVLA基础来源：https://github.com/huggingface/lerobot 。
- DexGraspVLA基础来源：https://github.com/Psi-Robot/DexGraspVLA 。
- DINOv2基础来源：https://github.com/facebookresearch/dinov2 。

提供的LeRobot基础快照含版本标记`07285677a3ef2f287a05c271f4b668f20e2ce57f`；DINO分支审计标记其本地基础为`736299d607882176391b97be24edd616d53864cb`；DexGraspVLA适配的本地工作树基础记录为`4be75b80b92119b8d5b0e59d67ae5f5e5e5f8a53`。这些不是同一个版本，且不保证都是公开上游可检出的commit；因此另外保存本轮实际文件SHA及对提供的基础快照的差异清单。

既有JC-parity接口、社区基础模型与后续selected66、DINO、位姿/力适配分开记录；本仓库不根据文件存在推断每一行代码的个人作者身份。私有工作树路径、同事目录与机器地址不公开。

LeRobot相关模块/补丁保留Apache-2.0版权与授权文件，补丁以提供的`lerobot-source`文件快照为基准，而非任意最新版。DexGraspVLA本地包未附授权文件，公开上游仓库元数据也未声明许可证，因此本仓库只整理其接口事实、来源哈希及独立数学参考，不重新打包该训练框架。

未打包原始示教数据、机器人SDK、设备地址、模型权重或完整训练输出。原始视频仅做兼容性转码及去除音轨，保留完整操作顺序；视频本身不绑定某一模型版本或checkpoint。
