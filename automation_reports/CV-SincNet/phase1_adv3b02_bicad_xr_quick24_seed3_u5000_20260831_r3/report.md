# Phase1 ADV3B02-BiCAD-XR Quick24正式实验报告（r3）

## 预登记

- 状态：`LOCAL_VERIFIED`，待N607 release与启动。
- run ID：`phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r3`。
- Git代码提交：`0284b80288418ba6eb342a42741bc66e9e6a08de`。
- 新run理由：r2中D0为6/6闭合；D5的6行因多元素训练头张量序列化错误在最终评估闭合阶段技术失败；E1与BiCAD-XDC的12行因合法XDC矩阵NaN占位符不能写入严格JSON而技术失败。r2全部checkpoint、日志、metrics和failure marker保持不可变，不重启、不覆盖。
- r3定点修复：XDC矩阵非有限占位符转为JSON`null`；多元素训练头张量转为嵌套列表；真实非有限`train_loss`显式抛出`FloatingPointError`，不得被null转换掩盖。
- 方法：`D0`、`D5`、`E1`、`ADV3B02-BiCAD-XDC-V1`。
- 矩阵：4候选×fold1/fold8×seed392001/392002/392003，共24行。
- 训练：day1/2/3，5000 optimizer updates，200epochs上限，source-only，`concat_sat_ce_only+leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- fold1 source receivers：`3,4,6,8`；fold8 source receivers：`1,3,4,6`。
- 调度：GPU0–7，每卡3个本run训练进程；启动前若出现无关占用，不得超过实时安全容量或影响无关进程。
- 本地release归档：`E:\type10-7\local_artifacts\phase1_bicad_xr_quick24_20260831_r3.tar.gz`。
- N607 release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_bicad_xr_quick24_20260831_r3`。
- N607 run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r3`。
- N607 dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r3.dispatcher.log`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，仅Phase1 source数据。
- 启动命令：`/bin/bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_bicad_xr_quick24_20260831_r3/code/scripts/launch_phase1_bicad_xr_quick24_n607_20260831_r3.sh`。
- 启动CWD：release根；环境：普通账户`szu2070436088`，禁止管理员账户。
- 预期每行artifact：`bicad_xr_final.pth`、checkpoint runtime、训练日志、clean JSON、三种LEO_WEAK JSON、严格重建结果和`ARTIFACTS_COMPLETE.json`。
- 允许停止：数据越权、错误candidate/fold/receiver/day/seed/update、输出冲突、错误release/CWD、命令无法运行、同一确定性pre-prediction异常至少重复2行、无final checkpoint/四场景闭合或进程归属不清。
- 禁止停止：中间或最终性能低、单seed差、缺少额外形式化receipt/hash/seal或报告字段。
- Phase2、target receiver、support、query、truth：全部禁止访问。

## 本地验证

- 两条r2根因测试均按TDD先红后绿。
- 新r3不可覆盖启动脚本测试按TDD先红后绿。
- BiCAD-XR：245项通过；仅3条既有AMP弃用警告。
- 相邻HCF-DG/ADV3B03：159项通过。
- `py_compile`：通过。
- `git diff --check`：通过。
- Git远端OID独立读回：`0284b80288418ba6eb342a42741bc66e9e6a08de`，与本地HEAD一致。
- 本地release SHA256：`2bfc6b832dee578234c7a3845ecaad890fe59bcdcb9f0cf733f0531bea121408`；归档已重新打开并核对包含r3启动脚本及两个定点修复文件。
- 独立P0/P1定点复审：进行中；仅其直接P0/P1结果可阻止release。
