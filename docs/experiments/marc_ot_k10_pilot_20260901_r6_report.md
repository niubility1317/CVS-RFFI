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
