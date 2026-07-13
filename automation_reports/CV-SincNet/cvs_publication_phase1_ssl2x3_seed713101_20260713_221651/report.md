# Phase1半监督2×3正式重跑报告

- 实验ID：`cvs_publication_phase1_ssl2x3_seed713101_20260713`
- 报告时间：2026-07-13 22:16:51 HKT
- 操作者：Codex
- 状态：本地实现与验证完成，待N607同步和启动
- 目标：在CVCNN-CE、RIEI-FD、DRIFT上分别比较伪标签与增强一致性两条独立无标签路线；两条路线均默认启用星地信道训练增强。

## 假设与比较对象

|路线|方法数|无标签机制|关键假设|
|---|---:|---|---|
|`pseudo_label`|3|高置信硬伪标签CE|置信度门控可利用源域无标签样本扩展TX监督|
|`augmentation_consistency`|3|clean→LEO强视图soft KL|不生成硬伪标签也能通过预测一致性提升星地鲁棒性|

两条路线互斥，不构造“伪标签+一致性”联合候选。主要比较同一方法、同一seed、同一数据划分和同一星地增强下的路线差异。

## 固定数据协议

|字段|值|
|---|---|
|source labeled|`0.1`|
|source unlabeled|`0.6`|
|source validation|`0.3`|
|train days|`0,1`|
|test days|`2,3`|
|train receivers|`0,1,2,3,4,5,6`|
|test receivers|`7,8,9,10,11`|
|epochs|`200`|
|seed|`713101`|

三部分源域数据互斥；目标接收机域不参与训练、无标签损失、阈值或checkpoint选择。

## 星地信道增强

两条路线均固定启用`--use_sat_channel_view_aug`，有标签训练batch使用clean+satellite双视图。训练与正式测试场景统一为：

- `leo_clear_weak`
- `leo_low_elev_weak`
- `leo_rain_weak`

训练增强参数：`sat_view_prob=1.0`、`sat_view_seed=2027`。

## 无标签机制参数

- 伪标签：`start_epoch=1`、`threshold=0.95`、`margin=0.0`、`lambda=1.0`。
- 增强一致性：clean预测经stop-gradient作为soft target，LEO强增强视图作为student，使用KL；`start_epoch=1`、`temperature=1.0`、`lambda=1.0`。
- 增强一致性路线不产生硬伪标签；伪标签路线不计算soft KL一致性损失。

## checkpoint选择

正式checkpoint固定为`best_by_val.pt`，仅当未取整source validation TX accuracy严格提高时覆盖。test、LEO test、伪标签指标、一致性指标和最终epoch不得参与选择。完成后逐方法验证checkpoint epoch等于全量`metrics.json`中的最高验证准确率epoch。

## 本地变更

- `baselines/common/consistency.py`：新增soft增强一致性损失及训练step。
- `baselines/common/cvs_trainer.py`：分别记录伪标签与一致性指标，不改变验证门控checkpoint语义。
- `baselines/{cvcnn_ce,riei_fd,drift}/train_cvs.py`：接入互斥的两条无标签路线，并强制无标签路线使用source SSL split。
- `scripts/launchers/run_cvs_baseline_queue.sh`：新增SSL模式、0.1/0.6/0.3划分和路线参数。
- `scripts/launchers/run_phase1_ssl_baseline_matrix.sh`：新增2×3专用launcher，默认星地增强。
- `tests/test_baseline_consistency.py`：新增损失、CLI和6命令矩阵测试。
- `docs/PHASE1_SSL_BASELINE_PROTOCOL_20260713.md`：Git-backed协议镜像。
- 根目录`项目.md`：先行更新5.1节，锁定两条互斥路线和正式checkpoint规则。

## 本地验证

- `conda run -n ssr-gpu python -m pytest -q tests/test_baseline_consistency.py tests/test_baseline_pseudo_labels.py tests/test_baseline_training_behaviors.py tests/test_cvs_rffi_launcher.py`：32通过、1跳过。
- `bash -n scripts/launchers/run_cvs_baseline_queue.sh`：通过。
- `bash -n scripts/launchers/run_phase1_ssl_baseline_matrix.sh`：通过。
- dry-run确认生成6条命令；每种方法各出现2次；伪标签和一致性开关不在同一命令中；6条命令均携带星地增强与0.1/0.6/0.3划分。

## N607计划

- cwd：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU映射：GPU0→CVCNN-CE两条路线；GPU1→RIEI-FD两条路线；GPU2→DRIFT两条路线。
- 并发：每GPU2条训练进程，不超过项目上限。
- run root：`paper_reproduction/runs/cvs_publication_phase1_ssl2x3_seed713101_20260713/{pseudo_label,augmentation_consistency}`
- log root：`paper_reproduction/logs/cvs_publication_phase1_ssl2x3_seed713101_20260713/{pseudo_label,augmentation_consistency}`
- launcher：`scripts/launchers/run_phase1_ssl_baseline_matrix.sh`
- exact command、scheduler/worker PID、文件SHA256和启动健康状态待同步后填写。

## 成功条件

- 6条实验均进入训练并完成200/200epoch；
- 伪标签线日志出现`[PSEUDO-METRICS]`且不出现`[CONSISTENCY-METRICS]`；
- 一致性线日志出现`[CONSISTENCY-METRICS]`且不出现`[PSEUDO-METRICS]`；
- 两条路线均确认`sat_view_aug=1`和三个`leo_*_weak`训练场景；
- 每个正式checkpoint均通过最高source validation accuracy复算；
- 对6个正式checkpoint执行同样本、同场景详细评估，并保留同一run的联合指标行。

## 风险

- 每GPU同时运行同一方法的两条路线；启动后必须核验显存，若OOM只允许降低batch size并记录有效优化步差异，不得关闭星地增强或无标签机制。
- 单seed结果仅支持受控配对比较，不支持显著性主张。
- 伪标签precision使用隐藏真值仅作审计，不参与训练或checkpoint选择。

