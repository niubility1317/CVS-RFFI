# CORE90 GAME V2矩阵发布

用户2026-09-11授权：优化修复问题，然后发布实验矩阵。

## 预登记

- 矩阵：V2_A-F×392005/392006/392007共18行，加392005的3个ordinary source LR候选，共21行。A/B为ordinary，C/D为head-only lookahead D1，E/F为完整EG；每对lambda_adv=0/.35。全部E200、FP32、确定性计算、固定课程、no-audit、control=off；这些关闭项是F1对照设计。strong候选LR=1e-4/2e-4/4e-4，完成后按预定四source场景平均macro-F1选一次，同分取低LR；本次不自动追加确认实验。
- 权限：source-only开发。固定split_seed=392002，沿用既有数据划分和物理角色，不改变数据schema。不访问target prediction/truth；训练末尾仅生成source四场景预测与评分。固定final E200，不按target或历史分数选checkpoint。
- 初始化：21行全部from_scratch=true，baseline_ckpt和game_resume为空。无teacher/EMA外部继承；EMA由本run fresh model复制。启动smoke仅对当场随机初始化状态save/load，不加载历史checkpoint。
- 本地基线修复：7b1b64cd；实际发布commit为包含本报告和code/configs/core90_game_v2_20260911_r3的Git提交，部署时记录完整OID。
- 发布修复：smoke在CUDA初始化前配置确定性；使用实际双参数组AdamW，验证E1/E80/E131及ordinary/D1/EG九次有限更新，E131读取U_s；调度CLI拒绝每卡>2，损坏completion归入技术失败。smoke只验证执行兼容性，不证明训练效果或自然伪标签非零。
- 本地验证：21个持久化JSON重新parse通过；16项矩阵/容量聚焦测试通过。独立P0/P1发布审查通过，限定这21行及每卡2进程；先前76项复审回归、B8的18个长轨迹端点证据继续有效。
- 环境：N607/dell-DSS8440，用户szu2070436088，Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，PyTorch2.1.0+cu121。2026-09-11直连preflight VERIFIED；/home剩余7.1T。
- 数据：/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl，已读回存在。
- release/CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/core90_game_v2_20260911_r3。
- run root：/home/szu2070436088/2510044040/CV-SincNet/runs/core90_game_v2_20260911_r3。
- logs/status：/home/szu2070436088/2510044040/CV-SincNet/logs/core90_game_v2_20260911_r3/queue_status.json及同目录各row.stdout.log。
- 唯一launch owner：本任务/root。全服务器GPU现存计算PID计入容量，每卡最多2个训练进程；健康任务保留。
- 精确命令（由本地Git归档落地后在release执行）：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/dispatch_core90_game.py --matrix code/configs/core90_game_v2_20260911_r3/matrix.json --release /home/szu2070436088/2510044040/CV-SincNet/releases/core90_game_v2_20260911_r3 --status /home/szu2070436088/2510044040/CV-SincNet/logs/core90_game_v2_20260911_r3/queue_status.json --max-processes-per-gpu 2`。
- 技术停止：两个pre-prediction失败停止新派发，保留pending和产物，等待健康active完成；无自动重试，无性能停机。协议/路径/输出碰撞或启动层故障立即停止派发并核实；不干预其他run。
- 预期artifact：resolved_config.json、backend_configuration.json、全epoch日志、final checkpoint、source_final_eval/source_prediction_manifest.json、四source场景scores及completion.json。仅exit0且SOURCE_ARTIFACTS_COMPLETE判该row完成。
- 科学判定：比较配对seed的solver×adv效应，同时报告实际field/head次数及wall time；3个source LR候选尚未构成已确认强基线。F2/F3需要新source donor证据，本次不扩展。现有probe TRANSFER_FAILURE保持失败事实，不将关闭的自适应控制标成激活。

## 当前状态

RUNNING，启动后态VERIFIED；尚未完成E200或最终评分。

Recovery: CPU prefix-sum fallback only for exact deterministic CUDA cumsum error. Same sampled increments, RNG stream and channel formula; deterministic flag retained. Local reproduced RED then 2 regression and 4 capacity tests PASS. All 21 configs differ from r1 only in output and config paths. No historical checkpoint used.

Second-site investigation: r2 smoke omitted the normal augmentor. Audited F1 call chain and covered all three prefix-sum sites (satellite phase, normal phase, complex slew) with shared torch_compat.deterministic_cumsum. Smoke now builds and stage-configures the real augmentor. Forced-old-CUDA local regression passes actual augmentation, complex prefix sum and all nine stage/solver smoke updates; RNG and unrelated error checks also pass. No determinism relaxation, no blind restart or reuse of failed roots. F1 audit-disabled capability path is outside this release coverage.

## 实际发布与后态

- 采集时间：2026-09-11T20:21:09.038010+08:00。发布commit：`5bd1665631b15b1ed97fae0f6b0ed57f25b68ec9`；本地/远端归档SHA256一致，远端编译PASS。
- r3真实L_s/U_s九项smoke PASS，实际增强器已接入，确定性保持开启；首批5行运行，16行排队，0失败。
- 调度PID=1024212，PPID=1；5个worker的PID/PPID/CWD/argv/GPU UUID与配置逐项读回一致，全服务器每卡2个计算进程。两次快照动作记录持续增长，全部已记录更新accepted且loss有限。
- V2_A_seed392005：PID=1024350，GPU=0，已接受54步，当前E2。
- V2_A_seed392006：PID=1024351，GPU=1，已接受54步，当前E2。
- V2_A_seed392007：PID=1024352，GPU=3，已接受54步，当前E2。
- V2_B_seed392005：PID=1024353，GPU=5，已接受52步，当前E2。
- V2_B_seed392006：PID=1024354，GPU=7，已接受49步，当前E1。
- r1首次smoke失败，r2普通增强分支失败，均保留；r2的6个失败worker及调度均已退出。修复源码及发布审查已完成，未复用旧root。
- 证据：[启动快照](initial_poststate.json)、[增长快照](verified_poststate.json)、[交付核验](delivery_verification.json)。当前仅启动验证，不作完整训练效果结论；16个待排队行尚无实际训练激活证据。

## 2026-09-12已完成行全量测试汇总

截至07:04:53，7/21行完成E200及source四场景预测评分，14行仍在训练或评分。完整解析3357条epoch和164780条动作记录；已完成行覆盖1400个epoch、68600次接受更新，独立复算756000条预测，与保存评分完全一致。

ordinary的adv=0与adv=0.35三seed平均LEO Accuracy分别为85.414%和85.911%，配对平均变化+0.497pp，逐seed方向并不一致。B8-D1、adv=0、seed392005与同seed ordinary的全部预测及置信度相同。全部数据仅属source V，尚无本批target测试或完整solver×adv结论。

详细数据：[逐实验、场景、TX/RX/day、混淆矩阵及训练轨迹报告](results_20260912_0700/report.md)。当前状态为PARTIAL_MATRIX_ANALYSIS / COMPLETED_ROWS_VERIFIED；健康实验继续运行，本次仅只读分析。

## 2026-09-12 23:47完整矩阵结果

21/21行均完成E200及四source场景预测评分，无失败。全量4200个epoch、205800次更新和2268000条预测已核查，独立复算与保存评分误差为0。全部更新接受，未发现非有限loss或stdout异常。

固定lr=2e-4时，A/B/C/D/E/F的三seed平均LEO Accuracy依次为85.414%/85.911%/85.414%/84.470%/87.682%/86.786%。E−A为+2.268pp，D−B为−1.441pp，F−E为−0.896pp。C与A三个seed的全部预测及置信度一致。source学习率候选中lr=4e-4按预登记四场景Macro-F1均值胜出，但只有单seed，未进行多seed确认。

[完整21行详细数据、逐TX/RX/day与计算量](results_20260912_2347/report.md)。状态：FULL_MATRIX_SOURCE_ANALYZED。本批没有target评估，也不作目标域泛化或科学晋级声明；本次未新增训练或修改远端状态。
