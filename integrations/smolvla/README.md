# SmolVLA integration snapshot

`modules/`保留提供材料中的新增relative_pose与dense_vision模块。`patches/`保存相对提供的lerobot-source快照的修改，覆盖模型、配置、数据工厂和训练入口。

这是可审查的实现快照，不是完整可安装LeRobot分发。先匹配`evidence/source_provenance.json`和文件哈希，审查diff后再集成；不要直接覆盖任意新版依赖。所有原版权头保留，授权文本见本目录LICENSE。

`reference/`中的轻量测试不导入这些PyTorch模块；原模型32项测试仅以历史审计记录呈现。

`tests/`保存材料中的两份PyTorch回归测试源码，需要匹配的完整LeRobot环境；根目录的12项参考测试不会导入或代替它们。
