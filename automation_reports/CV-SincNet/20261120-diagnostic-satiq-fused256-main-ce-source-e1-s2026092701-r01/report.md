# SatIQ 融合 256 的仅主 CE 完整首轮对照

[登记](experiment.json) · [事件](events.jsonl) · [单包任务口径](../../../docs/PAMF_SATIQ_SINGLE_PACKET_BENCHMARK_20261120.md) · [完整损失组](../20261119-phase1-satiq-ccs2023-fused256-sup100-s2026092701-r01/report.md)

## 改动

保留 8,000 点公共同步字段、原生五分支、160 维身份表示及其 96 维投影、3,103 类分类头、Adam 学习率 0.0003 和无合成增强。模型从零初始化。将同星分类 CE、分支 CE、分支 KL 和五分支原型 CE 的损失权重置零。训练代码保留其中部分辅助项的日志，梯度检查确认它们没有进入更新。

原始发布与当前发布使用实际 L 数据核对。两边初始状态、首批分类分数以及首轮完整有序组批一致。首轮 28,225 批，每批 32 包，共 903,200 个互异物理包。完整损失的对照来自原正式模型的第 1 轮，GPU6；本候选在 GPU0，两卡均为 RTX 3090。

## 固定读数

第 5,000、20,000、28,225 步记录源验证主 CE 和 3,103 类单包 Top-1。两组到 E1 后，用各自冻结检查点分别重算整个 L 和 V 的主 CE 与 Top-1。只用真实源包，独立 T 留出。K16 作为补充，闭集性能判断以全类单包为主。

完整损失组首轮源 V 已复核。主 CE 7.5360，单包 Top-1 0.2509%（125 / 49,814）。该基准来自实际 E1，候选还在运行。

## 实时执行

**VERIFIED 已启动。** 2026-11-20T02:42:22+08:00 已读回短检查退出码 0、控制进程 2244627、训练 PID 2244837、物理 GPU0 和第 100 步的训练日志。控制进程已接上 E1 后的两组全 L/V 评分。四组正式 200 轮实验独立运行。

- 训练日志 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-fused256-main-ce-source-e1-s2026092701-r01/train.log`
- 配置 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-fused256-main-ce-source-e1-s2026092701-r01/PAMF_SUP/resolved_config.json`
- 配对评分 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-fused256-main-ce-source-e1-s2026092701-r01/source_accuracy_e1.json`
