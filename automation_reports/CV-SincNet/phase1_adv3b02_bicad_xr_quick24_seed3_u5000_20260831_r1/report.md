# Phase1 ADV3B02-BiCAD-XR Quick24正式实验报告

## 预登记

- 状态：`LOCAL_VERIFIED`，待N607 release与启动。
- run ID：`phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r1`。
- Git代码提交：`acc610f1c8f0d947b189f9b00dd8d1b9a1e78275`。
- 方法：`D0`、`D5`、`E1`、`ADV3B02-BiCAD-XDC-V1`。
- 矩阵：4候选×fold1/fold8×seed392001/392002/392003，共24行。
- 训练：day1/2/3，5000 optimizer updates，200epochs上限，source-only，`concat_sat_ce_only+leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- fold1 source receivers：`3,4,6,8`；fold8 source receivers：`1,3,4,6`。
- 调度：GPU0–7，每卡3个本run训练进程；启动前若出现无关占用，则不得让总训练进程数超过该卡实时安全容量，也不得影响无关进程。
- 本地release归档：`E:\type10-7\local_artifacts\phase1_bicad_xr_quick24_20260831_r1.tar.gz`。
- N607 release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_bicad_xr_quick24_20260831_r1`。
- N607 run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r1`。
- N607 dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r1.dispatcher.log`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，远端已验证`torch2.1.0+cu121`且CUDA可用。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，仅Phase1 source数据。
- 启动命令：`/bin/bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_bicad_xr_quick24_20260831_r1/code/scripts/launch_phase1_bicad_xr_quick24_n607_20260831.sh`。
- 启动CWD：release根；环境：普通账户`szu2070436088`，禁止管理员账户。
- 预期每行artifact：`bicad_xr_final.pth`、checkpoint runtime、训练日志、clean JSON、`leo_clear_weak` JSON、`leo_low_elev_weak` JSON、`leo_rain_weak` JSON、严格重建结果和`ARTIFACTS_COMPLETE.json`。
- 允许停止：数据越权、错误candidate/fold/receiver/day/seed/update、输出冲突、错误release/CWD、命令无法运行、同一确定性pre-prediction异常至少重复2行、无final checkpoint/四场景闭合或进程归属不清。
- 禁止停止：中间或最终性能低、单seed差、缺少额外形式化receipt/hash/seal或报告字段。
- Phase2、target receiver、support、query、truth：全部禁止访问；不得以目标结果反馈选种、调参、重训或选择性重跑。

## 本地验证

- BiCAD-XR测试：240项通过，其中包含两项固定N607启动脚本测试；3条既有AMP弃用警告。
- 相邻HCF-DG/ADV3B03回归：159项通过。
- 真实ADV3B02历史checkpoint无query smoke：`PASS`；严格重建195个状态张量，反向控制、一次optimizer step及clean+三种LEO_WEAK前向完成。
- 独立P0/P1审查：原始receiver/day全局编号越界问题已修复；定点复审`CLEAN`。
- 24行干跑：24个组合唯一，GPU0–7各3行，train days均为day1/2/3，updates均为5000。

## 发布与运行记录

待release SHA、远端编译/smoke、dispatcher PID、worker绑定和最终结果返回后追加。低性能属于科学结果，不属于技术失败。
