# MARC-OT Target5/K10 pilot最小预登记

## 状态与冻结范围

- 新不可覆盖run ID：`marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1`。2026-09-01再次检查本地`automation_reports/CV-SincNet/<run-id>`、根目录`runs/<run-id>`、worktree`runs/<run-id>`及`docs/experiments/<run-id>_report.md`四个run-specific冲突路径，均为`ABSENT`。
- 冻结代码commit：`452967f1bb38f133afb69d0f7e5c027d97bcc92a`。
- 当前状态：`LOCAL_VERIFIED / FINAL_P0P1_REVIEW_APPROVED / N607_NOT_RUN`。最终独立P0/P1审查及唯一一次定点复审均已通过；真实Phase1 MARC-OT bundle、N607 preflight/release/compile、真实checkpoint无query smoke、pilot和评分仍待执行；`pilot_executed=false`，不得推导训练覆盖或性能结论。
- 冻结矩阵：`R0/R1/R2/R4/R6/R8`；Target5 pilot outer=`rx_3_19__seed_713102__k_10__new_5`；K=`10`；场景=`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

## Environment、CWD与GPU

- 本地验证environment：`ssr-gpu`；N607执行environment：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/checkout`。
- GPU：待N607 preflight后分配；命令中的`<GPU_AFTER_PREFLIGHT>`只允许替换为当次分配的物理GPU，进程内设备固定为`cuda:0`。

## 输入与输出路径

- config：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/checkout/configs/marc_ot_k10_pilot_20260901.json`。
- manifest：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json`。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- MARC-OT bundle：`/home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/inputs/marc_ot_weight_bundle.pt`。本地生产入口已具备执行bank step和严格保存/回读bundle的能力，但该真实训练输入当前未生成；不得以合成测试bundle或旧Meta-adapter bundle替代。
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/smoke`。
- pilot输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/pilot`。
- 独立score输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/score`。

## 冻结命令

```text
CUDA_VISIBLE_DEVICES=<GPU_AFTER_PREFLIGHT> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_marc_ot_phase1_bundle.py --config configs/marc_ot_phase1_bundle_20260901.json --output-root /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/inputs --device cuda:0

CUDA_VISIBLE_DEVICES=<GPU_AFTER_PREFLIGHT> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py smoke --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/smoke --device cuda:0 --batch-size 128 --arm R8 --scenario leo_clear_weak

CUDA_VISIBLE_DEVICES=<GPU_AFTER_PREFLIGHT> /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_marc_ot_pilot.py pilot --config configs/marc_ot_k10_pilot_20260901.json --manifest /home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --bundle /home/szu2070436088/2510044040/CV-SincNet/releases/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/inputs/marc_ot_weight_bundle.pt --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/marc_ot_k10_target5_r0r1r2r4r6r8_20260901_r1/pilot --device cuda:0 --batch-size 128
```

## 直接技术停止规则

只允许因协议/query越权、错误receiver/seed/K/scene/split、错误checkout/CWD、output root已存在、checkpoint或bundle绑定不符、非有限state/loss/gradient、确定性重复异常、无法产生完整prediction或独立scorer连接错误停止。不得因低性能、负收益或不晋级停止；低性能只进入分析和`NO_PROMOTION_TO_TARGET25`判断。

## 预期artifact

- Phase1：`config_snapshot.json`、`summary.json`和`marc_ot_weight_bundle.pt`；summary必须记录`EXACT_TASK_RECEIVER_COLUMNS`与`M_MINUS_D_FOR_EVERY_TRAINING_EPISODE`。
- smoke：`smoke_result.json`，其中`query_opened=false`、`query_rows_used=0`。
- pilot：18份`support_frozen_state.pt`、18份`predictions.npz`和`pilot_result.json`。
- prediction闭合后由独立scorer连接同一manifest绑定的truth sidecar，输出18份score及`score_collection.json`；本预登记不包含任何性能数值或结论。
