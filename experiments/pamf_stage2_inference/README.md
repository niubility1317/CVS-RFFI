# DURA v1.0

2026-10-21 用户指定的默认恢复器，部署标识 `dura_v1`。

固定 R69 `stage2` 源 E20/4940 现有权重，完整 266951 参数；两 prior、四 CG、27 次 FFT/IFFT。CUDA Graph、device 有效性累计与残差复用默认同时开启，FP32/TF32off，无新增训练或权重更新。已验证执行 commit 为 `47c264286ca689f060a3f36aafb763fe209a6f06`。

- [版本定义](frozen_version.json)：唯一权重、执行选项、验证 run 与指标。
- [完整机制与调用说明](../../docs/DURA_V1.md)：粗恢复、接收作用估计、两阶段先验/CG、源域蒸馏、执行加速与证据边界。
- [完整七景与同卡时延](../../automation_reports/CV-SincNet/20261021-phase1-restoration-stage2-graph-s2026092701-r01/report.md)：原 F50/defaulthead，168000×7，独立 truth-last；B1 G+F 41.220936→4.531972 ms。

默认入口 `InferenceEngine(model, reader=None, reuse=True, graph=True, native=False)`。原 R17、R65 `prune_after10` 和两阶段 native 保留为历史/执行对照。后续修改另记版本，不覆盖本版权重或实验产物。
