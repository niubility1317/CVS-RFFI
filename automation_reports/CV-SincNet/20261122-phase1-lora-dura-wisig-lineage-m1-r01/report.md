# LoRa 按 WiSig 原八级训练 DURA

运行名 `20261122-phase1-lora-dura-wisig-lineage-m1-r01`。N607 普通账户，既有 PyTorch CUDA 环境，工作目录为该 run 的独立 release。发布以 Git 提交记录为唯一身份，目标数据为已验证的全量 LoRa 对齐缓存。

## 数据、候选与资源

- 种子 2026092702；LoRa 源训练 L 2,800 包、U 25,200 包，按物理 ID 配对，只使用 IQ。源 V 中固定 600 包用于波形诊断。G 无 F、TX 真值和目标输入。
- 粗恢复先混合对齐算子代理、full 配对和 clean。作用监督只使用精确回放的源 residual 算子代理和 clean 单位作用。原版 R17、R38、R44、R46、R65、R69 蒸馏训练用 LoRa 现有 full 配对与 clean。各阶段 20×247 步、批量 256，保留 R65 E10 剪枝和原 R69 两展开阶段。
- 先串行训练一条完整 lineage。所有 GPU 可用，按发布时的训练进程与显存判断，单卡总上限 4 个进程，本运行同一时刻一个 GPU worker。建源作用缓存最多 12 CPU worker。
- 冻结同种子 Tri-set 负责正式读出。目标为固定 15 台未见接收机 × 2,000 个物理包 × 七景，登记八个既有对照和两个新读出。完整 DURA 输出与旧门控读出使用同一 DURA 波形。目标 clean 进入与 practical 一致的推理入口。
- 技术停止规则为数据契约或固定对照不符、连续链中出现非有限梯度、权重无法严格重载、源作用监督失败、预测矩阵不完整。保留原产物。源诊断分数不改变正式矩阵或冻结权重。

## 入口与产物

`python -m experiments.pamf_lora_main.full_dura_lineage_publish --repo REPO --output NEW_RELEASE --context LORA_CONTEXT --ssh-config N607_CONFIG --data-submit DATA --draw-submit DRAW --trace-submit TRACE --tri-submit TRI_SET`

远端入口为 `python -u -m experiments.pamf_lora_main.full_dura_lineage dispatch runtime.json`，自动完成源 smoke、作用代理缓存、训练、源 V 诊断和盲测预测。记录 `smoke.json`、`source_action/complete.json`、`G/lineage/{stage}/training.jsonl`、源权重与冻结状态、`source_wave` 及 1,050 个预测文件。全部预测冻结后，本机 `python -m experiments.pamf_lora_main.full_dura_score --submit SUBMIT --ssh-config N607_CONFIG --dataset LORA_DATA --output READOUT`，独立评分 70 个单元并报告逐景、逐接收机、逐 TX 和配对正确数。

首轮单种子用于检验完整训练链和实际增量。同一方案最终评估使用三种子，六景均值与 clean 相差至多 3 个百分点，另外对齐 WiSig 收回直通缺口的比例并报告绝对 clean。详细证据与可检验原因见 [LoRa DURA 诊断](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r01/diagnosis.md)。
