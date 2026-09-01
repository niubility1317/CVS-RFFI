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

## 2026-09-02 01:30只读进度

- 直接N607 preflight：`PASS`；项目根目录与8张GPU可见。
- r6冻结态由3份增至6份：三个R0和三个R1均已完成；`query_opened=false`边界不变。
- 精确r6 Python适配进程由14个降至11个；当前仍只有`leo_low_elev_weak/R2`一个异常日志，未出现第二种故障指纹。
- r6仍保持`PARTIAL_RUNNING_WITH_ONE_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得执行freeze-collection或打开query。
- 当前GPU仍承载r6剩余任务及其他既有工作负载，不满足安全发布完整替代矩阵的容量条件；本次不启动、不终止、不覆盖任何任务，继续等待。

## 2026-09-02 03:02只读进度

- r6新增`leo_low_elev_weak/R6`冻结态，当前共7份合法support回执，精确适配进程降至10个。
- 异常日志仍仅为已知`leo_low_elev_weak/R2`，未出现新故障指纹；query仍保持关闭。
- N607共有25个GPU计算进程，完整替代矩阵仍无安全容量；不启动替代run，继续只读等待。

## 2026-09-02 06:06第二个同指纹技术失败

- `leo_rain_weak/R2`已退出并出现与首个失败相同的零权重OT收敛异常：`row_error=0.000105031 column_error=9.31323e-09`。
- 当前r6为7份冻结态、9个精确适配进程、2个同指纹R2失败；这确认故障是旧R2路径的确定性系统问题，而非单场景偶发性能现象。
- 本地lazy-skip修复尚未部署到r6，因此不属于“修复后复现”；继续保留全部partial artifacts，不热补丁、不原地重启，也不停止仍健康的9个单元。
- GPU仍有24个计算进程，替代矩阵容量不足；query继续关闭，r6保持`NO_PERFORMANCE_RESULT`。

## 2026-09-02 06:38只读进度

- 新增`leo_clear_weak/R4`、`leo_rain_weak/R4`和`leo_rain_weak/R6`冻结态；r6现有10份合法support回执、6个精确适配进程。
- 异常仍固定为两个已知R2同指纹失败，未出现其他arm或新错误类型；query保持关闭。
- N607共有21个GPU计算进程，完整替代矩阵仍无安全容量；继续保留产物并等待。
