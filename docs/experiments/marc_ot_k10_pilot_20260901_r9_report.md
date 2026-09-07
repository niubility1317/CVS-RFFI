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

