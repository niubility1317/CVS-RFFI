# MARC-OT Target5/K10 r3最小预登记

r1与r2分别因空`query_guard`和真实ADV3B02共享Sinc的functional state冲突发生系统技术失败，均为`NO_PERFORMANCE_RESULT`并完整保留。本次只加入共享Sinc兼容修复，科学矩阵、seed、训练预算和选择规则不变。

- run ID：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3`
- 修复commit：`d6dd6c678edf8d63a92e0b42037eae8edfa5276e`
- 矩阵：`R0/R1/R2/R4/R6/R8`；K=`10`；Target5=`rx_3_19__seed_713102__k_10__new_5`；`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/checkout`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- Phase1输出：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs`
- run输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/{smoke,pilot,score}`
- GPU：物理GPU0，启动前重新核对

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_marc_ot_phase1_bundle.py --config configs/marc_ot_phase1_bundle_20260901.json --output-root /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs --device cuda:0

CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py smoke --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/smoke --device cuda:0 --batch-size 128 --arm R8 --scenario leo_clear_weak

CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py pilot --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r3/pilot --device cuda:0 --batch-size 128
```

smoke必须保持`query_opened=false/query_rows_used=0`；prediction闭合前不得打开truth；低性能只触发`NO_PROMOTION_TO_TARGET25`。
