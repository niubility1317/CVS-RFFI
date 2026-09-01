# MARC-OT Target5/K10 r2最小预登记

`r1`在真实Phase1训练前审计阶段因空`query_guard`发生`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`，其release与产物保持不变。本次只修复生产sampler漏传的部分类覆盖参数，冻结科学矩阵不变。

- run ID：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2`
- 修复commit：`8bf3d9abeea56846b8d96c47b9380400e500b5aa`
- 修复验证：正式seed的55个cell均含3～4个adapt类和2～3个guard类；关联69项通过；定点P0/P1审查`APPROVED`
- 矩阵：`R0/R1/R2/R4/R6/R8`；K=`10`；Target5=`rx_3_19__seed_713102__k_10__new_5`；三种LEO弱场景
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2/checkout`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- Phase1输出：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2/inputs`
- run输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2/{smoke,pilot,score}`
- GPU：物理GPU0，启动前重新确认

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_marc_ot_phase1_bundle.py --config configs/marc_ot_phase1_bundle_20260901.json --output-root /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2/inputs --device cuda:0

CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py smoke --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2/smoke --device cuda:0 --batch-size 128 --arm R8 --scenario leo_clear_weak

CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py pilot --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r2/pilot --device cuda:0 --batch-size 128
```

直接技术停止规则与r1一致；smoke必须`query_opened=false/query_rows_used=0`；prediction闭合前不得打开truth；低性能只触发`NO_PROMOTION_TO_TARGET25`。
