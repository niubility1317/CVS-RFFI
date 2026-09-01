# MARC-OT Target5/K10 r6并行pilot最小预登记

- 状态：`LOCAL_VERIFIED`
- run ID：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r6`
- Git commit：`9b0bbe44a5c6b3e1fa3a725099e8cfd57461b257`
- Phase1输入：复用r3已完成且严格回读的`marc_ot_weight_bundle.pt`；不重训、不覆盖
- 矩阵：`R0/R1/R2/R4/R6/R8`，K=`10`，Target5=`rx_3_19__seed_713102__k_10__new_5`，三种LEO弱场景
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r6/checkout`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r6`
- bundle：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt`
- GPU：物理GPU0～7；每卡最多2个训练单元
- 预期artifact：18份`support_frozen_state.pt`与`support_state_receipt.json`、`support_collection.json`、18份prediction与回执、`pilot_result.json`、独立`score/score_collection.json`
- 本地验证：分片CLI42项、MARC-OT相邻链路119项通过；定点P0/P1复审`APPROVED`
- 技术停止规则：协议/query越界、错误split/receiver/seed/K/scene、输出冲突、错误checkout、无prediction闭合或确定性执行故障；低性能不得停止

## support-only并行调度

| GPU | 单元 |
|---:|---|
| 0 | `leo_clear_weak/R8`、`leo_clear_weak/R1` |
| 1 | `leo_low_elev_weak/R8`、`leo_low_elev_weak/R1` |
| 2 | `leo_rain_weak/R8`、`leo_rain_weak/R1` |
| 3 | `leo_clear_weak/R6`、`leo_clear_weak/R2` |
| 4 | `leo_low_elev_weak/R6`、`leo_low_elev_weak/R2` |
| 5 | `leo_rain_weak/R6`、`leo_rain_weak/R2` |
| 6 | `leo_clear_weak/R4`、`leo_low_elev_weak/R4` |
| 7 | `leo_rain_weak/R4`；3个R0单元先顺序完成 |

每个训练进程只执行：

```text
CUDA_VISIBLE_DEVICES=<physical_gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py adapt-unit --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r6/pilot --device cuda:0 --batch-size 128 --scenario <scenario> --arm <arm>
```

只有18份冻结态全部通过`freeze-collection`后，`predict-unit`才可打开query；prediction完整前不得连接truth。

## LANDED与smoke

- release归档SHA256：`4a846308aad6f53c85dd23d448ff7ed570257fb9e25f5732392e7bdbb66c1dd3`，本地/远端一致
- r6 checkout远端编译通过
- 真实checkpoint no-query smoke：`PASS`
- smoke边界：`query_opened=false`、`query_rows_used=0`、`source_iq_rows_used=0`
- 正式启动前GPU0～7均恢复为空闲
- 状态：`LANDED / SMOKE_PASS`

## 正式启动与首次回读

- 启动时间：2026-09-01 23:56～23:57 CST
- 状态：`RUNNING`
- `R0`已在三个场景顺序完成，共3份`support_frozen_state.pt`和3份`support_state_receipt.json`；三个回执均为`SUPPORT_STATE_FROZEN`，且`query_opened=false`、`query_rows_used=0`
- 其余15个`adapt-unit`均已启动；PID与物理GPU映射已逐项回读，GPU0～6各2个训练单元，GPU7为1个训练单元，未超过每卡2个训练实验的上限
- 15个Python进程的CWD均为r6不可变release checkout；GPU显存占用约820～912MiB/进程，首次回读未见`Traceback`、`RuntimeError`、CUDA OOM或协议异常指纹
- 启动用本地SSH通道已全部退出；远端训练继续运行。后续保持低频只读监控，不因中间性能停止，不在18份冻结态齐备前执行`freeze-collection`或打开query

## 2026-09-02 00:44技术异常回读

- 状态：`PARTIAL_RUNNING_WITH_ONE_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- `leo_low_elev_weak/R2`在support-only阶段失败：`support-bank transport marginals failed to converge: row_error=0.00010051 column_error=5.58794e-09`。
- 根因：R2的transport/statistics权重均为0，但r6旧实现仍无条件计算Sinkhorn并执行收敛检查；与R2目标无关的零权重分支中断了该单元。
- 影响：3个R0已完成，14个Python适配进程仍健康运行约49分钟，失败单元已退出；18/18冻结态无法闭合，因此不得打开query、不得连接truth、不得评分。
- 处置：保留r6全部产物，不覆盖、不原地重启、不停止其余14个健康单元。本地修复采用零权重lazy-skip并已有回归测试；完成定点P1修复、验证、Git发布和新不可变release后，只能以新run ID执行替代实验。
- ETA：当前r6本身不能合法完成，故无有效完成时间；替代run的ETA需在修复release真实smoke和首个完整适配单元后重新估算。
