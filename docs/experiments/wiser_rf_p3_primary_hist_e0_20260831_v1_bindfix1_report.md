# WISER-RF P3-Primary历史绑定修复pilot预登记

## 状态与版本

- run ID：`wiser_rf_p3_primary_hist_e0_20260831_v1_bindfix1`；当前状态：`LOCAL_VERIFIED`。
- Git提交：`ba704748d78d2ea37307c4a5aed2f582866bbfa6`。该提交只修复历史D92 E0 pilot的真实`capsule_id/split_id`绑定并增加回归测试；不改变P3方法、训练预算、arm、scene或科学门槛。
- 前一run`wiser_rf_p3_primary_hist_e0_20260831_v1`在无query smoke阶段因旧绑定占位值退出，未进入pilot，永久保留为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；本run为用户已授权的新提交、新release和新run ID修复重发。
- 本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。聚焦回归先复现失败，修复后pilot与Target25相关30项测试及`py_compile`通过。
- 2026-08-31只读preflight曾确认8张GPU均无compute-app；本run仍冻结物理GPU0，启动前重新盘点且不超过用户授权的每GPU3个训练实验上限。

## 冻结输入、矩阵与协议

- outer=`rx_3_19__seed_713102__k_10__new_5`，receiver=`3-19`，seed=`713102`，K=`10`，new-count=`5`；arm=`N0,N1,N2,N3,N4,N5,N6`；scene=`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 真实历史绑定：`capsule_id=d92-e0-full-target125:5910674066e8bbf93684fddd6af6fd2cef7e8f208d64e403ac7e58030a2a8cc5`；`split_id=d92-e0-full-target125:rx_3_19__seed_713102__k_10__new_5`；`protocol_schema=p2_min_v1`；`phase2_data_status=VALIDATED_ONCE`。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- manifest：`/home/szu2070436088/2510044040/CV-SincNet/runs/bisage_d92_hist_e0_target125_20260830_v1_techfix1/matrix_manifest.json`。
- source summary：`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component/int8_domain_class_prototypes.npz`；source binding=`configs/wiser_rf_adv3b02_source_binding.json`；P3 config=`configs/wiser_rf_p3_primary_20260831.json`。
- support-only完成训练、阶段选择和插值，固定`query_rows_used=0`。21个support状态全部冻结后才能只读打开query；prediction完整后才由独立scorer连接truth。

## release、路径与命令

- release归档：`wiser_rf_p3_primary_hist_e0_20260831_v1_bindfix1_ba704748.tar.gz`；远端release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/wiser_rf_p3_primary_hist_e0_20260831_v1_bindfix1_ba704748`。仅允许一次归档本地/远端SHA比对和一次远端编译。
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/wiser_rf_p3_primary_hist_e0_20260831_v1_bindfix1`；log：`/home/szu2070436088/2510044040/CV-SincNet/logs/wiser_rf_p3_primary_hist_e0_20260831_v1_bindfix1/pilot.out`；score root：`<run-root>/score`。四个新目标均已只读确认不存在。
- smoke与pilot由同一后台owner串联，前者非零即不执行后者：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-smoke --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root <run-root>/smoke --device cuda:0 --runtime-commit ba704748d78d2ea37307c4a5aed2f582866bbfa6 --arm N6 --scenario leo_clear_weak
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_stage2_wiser_pilot.py p3-pilot --manifest <manifest> --pilot-outer-key rx_3_19__seed_713102__k_10__new_5 --p3-config configs/wiser_rf_p3_primary_20260831.json --checkpoint <checkpoint> --source-summary <source-summary> --source-binding configs/wiser_rf_adv3b02_source_binding.json --output-root <run-root>/pilot --device cuda:0 --runtime-commit ba704748d78d2ea37307c4a5aed2f582866bbfa6 --arms N0 N1 N2 N3 N4 N5 N6
```

## 停止规则、artifact与晋级

- 仅因协议/query/truth泄漏、错误split/receiver/seed/K/scene、输出冲突、可微与精确D92不同构、非有限loss/gradient、prediction不完整、scorer绑定错误、进程归属不清或确定性重复异常停止；不得因低性能停止。
- 预期artifact：smoke结果、21个support audit与prediction/receipt、pilot completion marker、独立detailed score、资源记录和`pilot_auto_result.json`。
- pilot门槛保持不变：P3 BA三scene中位提升≥3pp、最差scene≥-0.5pp、P3 floor中位及low-elev floor不下降、P1/P2每scene≥-2pp、zero-id=0、条件数≤基线2倍、至少2/3scene净help为正；N1不得成为冠军。
- 只有三个scene全部闭合且`full_target25_authorized=true`才发布历史Target25的25outer/75scene；否则记录科学未晋级。Target25通过才授权K10的`new5/new10/new20`扩展，Stage B仍不在本run自动执行范围内。
