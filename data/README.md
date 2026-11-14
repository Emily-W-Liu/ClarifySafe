# ClarifySafe v2 数据

[加密数据包](clarifysafe-v2.bundle)包含 48 条正式测试任务（12 个场景组）、24 条开发任务、18 条风险门控消融任务及 9 条只读对照任务。[校验清单](manifest.json)记录每个数据文件的哈希。风险消融的 18 条任务由正式集的 9 个场景派生。正式测试结果以 48 条测试任务为分母。

数据来自 AppWorld 0.2.0，固定源码提交为 `42b5bcf3cd334fee33f0c37c02070a9f5807add5`。根据 [AppWorld 的许可说明](https://github.com/StonyBrookNLP/appworld#lock_with_ink_pen-license)，受保护部分及其衍生数据的公开分发采用加密格式。此包使用 AppWorld 的 bundle 格式和工具，内附上游关于加密分发的说明与 [Apache 2.0 许可证全文](LICENSE-AppWorld)。解包与复现步骤见[实验指南](../docs/experiments.md)。
