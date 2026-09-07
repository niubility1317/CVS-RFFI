# MARC-OT K10 Target5替代实验r8

- 状态：`LOCAL_VERIFIED`
- run ID：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r8`
- 候选/矩阵：MARC-OT优化版；`rx_3_19/seed713102/K10/new5`，3个LEO弱场景×`R0,R1,R2,R4,R6,R8`，共18个support-only单元；R0顺序运行，其余15个单元按GPU容量并行。
- 协议：`p2_min_v1`、`VALIDATED_ONCE`；`capsule_id=d92-e0-full-target125:5910674066e8bbf93684fddd6af6fd2cef7e8f208d64e403ac7e58030a2a8cc5`；`split_id=d92-e0-full-target125:rx_3_19__seed_713102__k_10__new_5`。
- 冻结代码/config commit：`393449e4ac2ef5fb84ffe09b3ad1e670a89734db`；核心实现提交：`c816eb0b4acad8c2a59f7ffa5590d90da8b07b53`；r7 autograd技术修复提交：`393449e4ac2ef5fb84ffe09b3ad1e670a89734db`。
- 命令：`CUDA_VISIBLE_DEVICES=<physical_gpu> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py adapt-unit --config configs/marc_ot_k10_optimized_20260902.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r8/pilot --device cuda:0 --batch-size 128 --scenario <scenario> --arm <arm>`。
- 环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r8/checkout`。
- 输入：冻结matrix manifest、ADV3B02 checkpoint和r3 Phase1权重bundle；不重训Phase1。
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r8`，不可覆盖。
- GPU：遵守每卡最多2个训练实验；发布前重新核实容量，不干预既有任务。
- 停止规则：仅协议/query越界、错误split/receiver/seed/K/scene、输出冲突、错误checkout、无prediction闭合或确定性系统故障；低性能不得停止。
- 预期artifact：18份`support_frozen_state.pt`和回执、`support_collection.json`、18份prediction和回执、`pilot_result.json`、独立`score/score_collection.json`。
- truth-last：18份support冻结态全部完成后才能`freeze-collection`；prediction全部固定前不得连接truth。

## 2026-09-08 00:16远端smoke终态

- release归档本地/远端SHA256一致：`4ec209c2dc1e186d674e56c33b66a97656a1e592ceb7589040d11dc1d9a747f2`；远端编译通过。
- r7的autograd断链已消失；真实checkpoint、无query smoke继续运行到D92评分张量桥接时失败：`RuntimeError: Could not infer dtype of numpy.float32`，位置为`stage2_marc_ot_runner.py:463 torch.as_tensor(...)`。
- formal adapt-unit为`0/15`，未创建support collection、未打开query、未生成prediction、未连接truth、未评分；没有性能结果。
- r8终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。保留release和smoke产物，不覆盖、不原地修补或重启；修复旧版PyTorch/NumPy数组桥接后使用新提交、新release和新run ID。
