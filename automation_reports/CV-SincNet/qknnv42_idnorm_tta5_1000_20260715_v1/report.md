# qKNNV42严格ADV3B02+289,685参数适配器+TTA5正式矩阵

本文件是根目录正式报告`E:\type10-7\automation_reports\CV-SincNet\qknnv42_idnorm_tta5_1000_20260715_v1\report.md`的Git承载镜像。实验固定1000任务：125次单qKNN+单视图FFT96基线，以及7个epoch档×125次任务独立`289,685`参数`id_norm_late_feature`适配；适配臂使用固定5-view TTA+FFT96。

训练只允许当前任务的目标receiver LEO support，禁止clean、source、proxy和query；query禁dense图、角色Oracle和类别配额。该FP16 delta理论状态为579,370字节且推理需要5次骨干前向，因此仅作为非极轻量资源诊断，不得作为星上轻量部署成功证据。

完整方法、损失、验证、N607命令、运行状态、最终表和artifact索引以根目录正式报告为工作面；每次状态更新后同步本文件。

当前证据：Git提交`e1e6711`；N607直连、GPU/磁盘/进程预检PASS；5个运行文件SHA256一致并通过远端编译；实际checkpoint严格加载0/0/0，`id_norm_late_feature`精确为`289,685`参数、42个参数张量；三份母缓存未发现clean view。

运行状态：端到端E1/K1 smoke训练与评测PASS；1000任务manifest已生成，8个shard于2026-07-15 13:24CST启动，PID为`1371812–1371819`，正在运行。

## 最终摘要

- 状态：1000/1000评测、875/875训练、16,000/16,000行loss全部完成并通过审计；0失败、0日志异常。
- 最佳：30epoch，old=73.133%、new=63.793%、H=66.762%，相对单视图基线配对ΔH=+3.594±1.116pp。
- K依赖：K1适配有害；K5/K10/K20在30epoch分别提升H约4.028pp、6.334pp、9.221pp。
- 资源：289,685参数、579,370B FP16 delta、5次骨干前向；属于非极轻量资源诊断。
- 完整报告与artifact索引见根目录正式报告和`E:\type10-7\local_artifacts\qknnv42_idnorm_tta5_1000_20260715_v1_summary`。
