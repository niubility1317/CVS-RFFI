# Phase 2 clean 已完成预测健康清点

对保留的原生、PAMF、混合方法 pilot、正式接续及两代 MoPC 队列，逐个检查有完成标记的任务：索引身份、数据口径、预期阶段/K/N、预测文件、query 索引补集、预测类别范围，以及现有 fit JSON 记录中数值有限。不会读取 truth 或对成绩评分，不改变任何实验输出。未保存 fit 记录的任务只标记无法据此检查损失；数值有限并不单独证明准确率或方法正确性。

独占 `runs/20261115-phase2-clean-health-audit-cpu4-r01`，最多 4 个 CPU 进程。输入仅为 `20261113-phase2-clean-native-pilot-r01`、`20261113-phase2-clean-hybrid-pilot-r01`、`20261114-phase2-clean-pamf-pilot-r01`、`20261114-phase2-clean-non-issl-full-m4-r02`、`20261114-phase2-clean-scale7-full-m4-r03`、`20261114-phase2-mopc-public-clean-pilot-m1-r01`、`20261114-phase2-mopc-public-clean-full-m4-r01`、`20261115-phase2-mopc-clean-affine-resume-m21-r01` 的 scheduler.json；不把未完成或部分预测计为完整。输出状态、各方法有效计数、记录存在性和具体异常，失败时保留证据并检查根因。
