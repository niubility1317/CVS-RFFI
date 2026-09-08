# MARC-OT K10 Target5替代实验r9

- 状态：`LOCAL_VERIFIED`
- run ID：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r9`
- 候选/矩阵：MARC-OT优化版；`rx_3_19/seed713102/K10/new5`，3个LEO弱场景×`R0,R1,R2,R4,R6,R8`，共18个support-only单元；R0顺序运行，其余15个单元按GPU容量并行。
- 协议：`p2_min_v1`、`VALIDATED_ONCE`；`capsule_id=d92-e0-full-target125:5910674066e8bbf93684fddd6af6fd2cef7e8f208d64e403ac7e58030a2a8cc5`；`split_id=d92-e0-full-target125:rx_3_19__seed_713102__k_10__new_5`。
- 冻结代码/config commit：`96df4b46fd3d02e6c6330f45308c3c8f3ce4d1df`；核心实现提交：`c816eb0b4acad8c2a59f7ffa5590d90da8b07b53`；r7 autograd修复提交：`393449e4ac2ef5fb84ffe09b3ad1e670a89734db`；r8旧版NumPy桥接修复提交：`96df4b46fd3d02e6c6330f45308c3c8f3ce4d1df`。
- 命令：`CUDA_VISIBLE_DEVICES=<physical_gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py adapt-unit --config configs/marc_ot_k10_optimized_20260902.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r9/pilot --device cuda:0 --batch-size 128 --scenario <scenario> --arm <arm>`。
- 环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r9/checkout`。
- 输入：冻结matrix manifest、ADV3B02 checkpoint和r3 Phase1权重bundle；不重训Phase1。
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r9`，不可覆盖。
- GPU：遵守每卡最多2个训练实验；发布前重新核实容量，不干预既有任务。
- 停止规则：仅协议/query越界、错误split/receiver/seed/K/scene、输出冲突、错误checkout、无prediction闭合或确定性系统故障；低性能不得停止。
- 预期artifact：18份`support_frozen_state.pt`和回执、`support_collection.json`、18份prediction和回执、`pilot_result.json`、独立`score/score_collection.json`。
- truth-last：18份support冻结态全部完成后才能`freeze-collection`；prediction全部固定前不得连接truth。

## 2026-09-08 00:27运行状态

- release归档本地/远端SHA256一致：`f6d4d238741077bb9e58d55d4832826510a521077fcb2f00ce92ad693f0d11fe`；远端编译通过。
- 真实checkpoint、`leo_clear_weak/R8`无query smoke通过：`status=PASS`、`query_opened=false`、`query_rows_used=0`；训练审计显示24个optimizer step，四阶段参数均实际到达。
- 容量核验时N607有4个既有计算进程；在每GPU最多2个训练实验的限制内，已启动12/15个正式适配单元：3个场景的R1/R2，以及`leo_clear_weak`和`leo_low_elev_weak`的R4/R6/R8。其余`leo_rain_weak/R4,R6,R8`等待至少3个合规槽位。
- 一次启动命令曾使用无效场景名`leo_rician_weak`和`leo_shadowed_weak`，7个进程由argparse在正式输出创建前退出；失败日志已保留。随后使用预注册场景`leo_low_elev_weak`和`leo_rain_weak`启动对应单元，未覆盖失败日志。
- 当前状态：`RUNNING_SUPPORT_ONLY`；尚未freeze collection、未打开query、未生成prediction、未连接truth、未评分。

## 2026-09-08 10:47用户暂停终态

- 用户明确要求“任务先停止”；仅向仍存活的12个r9 `adapt-unit`进程发送`TERM`，随后以完整r9命令指纹独立核验存活进程为0。
- 已保留3份完成态：`leo_clear_weak/R1`、`leo_low_elev_weak/R1`、`leo_rain_weak/R1`；其`support_state_receipt.json`均已存在。其余12个单元未形成完成回执，不宣称完成。
- release、日志、3份冻结态和所有中间产物均保留，未删除、未覆盖；未干预其他任务。
- query始终未打开；尚未freeze collection、未生成prediction、未连接truth、未评分，没有可报告的最终性能结果。
- r9当前状态：`PAUSED_BY_USER / PARTIAL_SUPPORT_ARTIFACTS / NO_FINAL_PERFORMANCE_RESULT`；自动续跑已暂停。只有用户再次明确授权恢复后，才可先核实产物和容量并制定不覆盖既有产物的恢复方案。
