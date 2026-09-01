# MARC-OT Target5/K10 r4最小预登记

r3 Phase1 bundle已真实完成；r3 no-query smoke在打开query前因CUDA资源统计初始化顺序发生技术失败。本次只修复`set_device -> reset_peak_memory_stats`顺序，复用r3不可变bundle，科学矩阵不变。

- run ID：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r4`
- 修复commit：`db71994f748c7c23f9b5defeacfe74c701d3078f`
- 矩阵：`R0/R1/R2/R4/R6/R8`；K=`10`；Target5=`rx_3_19__seed_713102__k_10__new_5`；三种LEO弱场景
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r4/checkout`
- bundle：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt`
- smoke/pilot/score：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r4/`
- GPU：物理GPU0，启动前重新核对

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py smoke --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r4/smoke --device cuda:0 --batch-size 128 --arm R8 --scenario leo_clear_weak

CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py pilot --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r4/pilot --device cuda:0 --batch-size 128
```

smoke必须`query_opened=false/query_rows_used=0`；prediction闭合前不得打开truth；低性能只触发`NO_PROMOTION_TO_TARGET25`。
