# qKNNV42严格ADV3B02+289,685参数适配器+TTA5正式矩阵

本文件是根目录正式报告`E:\type10-7\automation_reports\CV-SincNet\qknnv42_idnorm_tta5_1000_20260715_v1\report.md`的Git承载镜像。实验固定1000任务：125次单qKNN+单视图FFT96基线，以及7个epoch档×125次任务独立`289,685`参数`id_norm_late_feature`适配；适配臂使用固定5-view TTA+FFT96。

训练只允许当前任务的目标receiver LEO support，禁止clean、source、proxy和query；query禁dense图、角色Oracle和类别配额。该FP16 delta理论状态为579,370字节且推理需要5次骨干前向，因此仅作为非极轻量资源诊断，不得作为星上轻量部署成功证据。

完整方法、损失、验证、N607命令、运行状态、最终表和artifact索引以根目录正式报告为工作面；每次状态更新后同步本文件。
