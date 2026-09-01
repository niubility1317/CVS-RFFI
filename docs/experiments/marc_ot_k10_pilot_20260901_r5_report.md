# MARC-OT Target5/K10 r5最小预登记

- 状态：`LOCAL_VERIFIED`
- run ID：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r5`
- 修复commit：`43778637f8338f3468a7b4366fa858aeccc8ae75`
- 修复范围：support-bank OT计划与transport loss统一使用每特征维均方距离；严格边际容差、`epsilon=0.1`和80轮不变
- Phase1输入：复用r3已完成且严格回读的`marc_ot_weight_bundle.pt`；不重训、不覆盖
- 矩阵：`R0/R1/R2/R4/R6/R8`，K=`10`，Target5=`rx_3_19__seed_713102__k_10__new_5`，三种LEO弱场景
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r5/checkout`
- bundle：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt`
- run输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r5/{smoke,pilot,score}`
- GPU：物理GPU0，启动前重新核对
- 本地验证：正式几何RED→GREEN；相关107项通过；独立P0/P1审查`APPROVED`
- 技术停止规则：协议/query越界、错误split/receiver/seed/K/scene、输出冲突、错误checkout、无prediction闭合或确定性执行故障；低性能不得停止

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py smoke --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r5/smoke --device cuda:0 --batch-size 128 --arm R8 --scenario leo_clear_weak

CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py pilot --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r5/pilot --device cuda:0 --batch-size 128
```

smoke必须`query_opened=false/query_rows_used=0`；prediction完整前不得连接truth。

## LANDED

- release归档SHA256：`6cc3f1b64c2dc830802e09b5c9d419791a42f5f594e19d94a762df08c0b170ba`，本地/远端一致
- r5 checkout远端编译通过；发布前GPU0空闲
- 状态：`LANDED`

## no-query smoke

- 状态：`PASS`
- artifact：`smoke/smoke_result.json`，`99,248`字节
- 边界：`query_opened=false`、`query_rows_used=0`、`source_iq_rows_used=0`
- 绑定：`capsule_id`、`split_id`、Target5 outer key与预登记一致
- 执行：R8四阶段共24个优化步，5折held-out support证据有效，完整support路径完成
- 资源：训练`15.1459s`，CUDA峰值`424,442,368`字节
- 性能声明：smoke不连接truth，不产生正式性能结论
