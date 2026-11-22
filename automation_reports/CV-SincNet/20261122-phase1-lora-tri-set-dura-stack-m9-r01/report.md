# LoRa Tri 候选集合与 DURA 的固定联合对照

使用 `20261122-phase1-lora-tri-label-set-m3-r01` 的三个同种子冻结识别器，固定比较直通、原固定 DURA、原 DURA 加推理共享形状保护、训练中保护共享形状的 DURA。两级冻结模型只使用同一 LoRa 合同，不重训或使用目标改方法。三种 DURA 路线均与本行冻结 Tri 直通逐包配对。全部目标 30000 包，15 个未见接收机，clean 与六景，84 个评分单元。源审计仅检查波形。

N607 普通账户、GPU0–7，按实时容量调度每卡本次最多一进程。命令为 `python -m experiments.pamf_lora_main.full_dura_publish --stack`，传入对齐数据、原 G、保护重训 G 和新版 Tri 的 release receipt。CWD 为 `CV-SincNet/releases/20261122-phase1-lora-tri-set-dura-stack-m9-r01-release`，输出为 `CV-SincNet/runs/20261122-phase1-lora-tri-set-dura-stack-m9-r01`。真实冻结 G 的无目标源 smoke 后接续。来源、非有限值、clean/旁路、盲预测计数异常为技术停止条件；预测全部固定后独立评分。该矩阵也提供下一实验的原样冻结对照。
