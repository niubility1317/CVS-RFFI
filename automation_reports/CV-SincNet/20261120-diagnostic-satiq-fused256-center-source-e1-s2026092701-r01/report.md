# SatIQ 融合 256 训练期中心化首轮配对

[配置](experiment.json) · [事件](events.jsonl) · [融合 256 正式对照](../20261119-phase1-satiq-ccs2023-fused256-sup100-s2026092701-r01/report.md) · [E10 中心化分解](../20261120-diagnostic-satiq-e10-center-whiten-s2026111907-r01/report.md)

第 10 轮的融合 256 冻结表征从原始跨日同星 AUC 0.5862 提升到源均值中心化的 0.6716。为检验在训练中扣均值的收益，原有单位化 256 维融合向量进入原分类头之前先扣除一个训练源均值缓冲，并重新单位化。缓冲从零开始，每个训练批用本批已单位化向量的 detached 均值作指数移动更新，更新系数 0.05。验证和独立评估只能读取缓冲。检查点保存该状态。没有新增训练参数、身份特征或波形输入。

与融合 256 使用相同 8,000 点输入、标签、原五分支、原版合成增强关闭、损失配方和模型／采样种子，从零训练一个实际 epoch，28,225 步。固定 5,000、20,000、28,225 步比较 3,103 类验证 Top-1、主 CE 和固定同星 K16。对照组的完整首轮跨日同星 AUC 已核实为原始 0.5765、仅扣源 L 固定均值 0.6507、部分白化 0.6656。训练期中心化评估同时保留普通原始对照和原模型离线中心化读出，避免把已有后处理的收益归给训练改动。

N607 物理 GPU0，PID 2197179，2026-11-20T01:29:13+08:00 已读回生效配置、CWD、GPU UUID 和第 400 步训练日志。**VERIFIED 已启动**。验证结果待出。正式四组继续，独立 T 保持留出。

- 生效配置、训练日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-fused256-center-source-e1-s2026092701-r01/PAMF_SUP/resolved_config.json`、`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-fused256-center-source-e1-s2026092701-r01/PAMF_SUP/steps.jsonl`
- 过程日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261120-diagnostic-satiq-fused256-center-source-e1-s2026092701-r01/train.log`
