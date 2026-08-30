# Phase1 ADV3B02-BiCAD-XR Quick24正式实验报告（r3）

## 预登记

- 状态：`RUNNING`。
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
- 独立P0/P1定点复审：`CLEAN`；38/38项聚焦测试通过，6个审查范围文件与代码提交`0284b80288418ba6eb342a42741bc66e9e6a08de`一致，无直接导致真实run崩溃、非法JSON、非有限train loss被掩盖、输出覆盖、target/Phase2越权或artifact无法闭合的问题。

## 发布与启动记录

- N607普通账户只读preflight：`PASS`；用户`szu2070436088`，项目根可见，启动前GPU0–7均无compute process。
- 不可覆盖核查：r3的release根、run根、dispatcher日志、PID文件和远端归档均不存在；未触碰r2或无关任务。
- release归档：本地与远端SHA256均为`2bfc6b832dee578234c7a3845ecaad890fe59bcdcb9f0cf733f0531bea121408`；只执行这一次归档SHA比较。
- 远端编译：`PASS`；两个修复模块、矩阵launcher及r3启动脚本语法均通过。
- 真实checkpoint无query smoke：`PASS`；使用r2的`D0-F1-S392002/bicad_xr_final.pth`，GPU0严格重建，missing/unexpected/shape mismatch均为空，optimizer step、有限loss/梯度及clean和三种`LEO_WEAK`前向全部通过；`target/Phase2/support/query/truth`访问均为`false`。
- 启动时间：N607服务器时间2026-08-31；dispatcher PID`2583716`，PPID`1`，CWD严格绑定r3 release根，cmdline严格绑定r3 run/release、formal quick24矩阵和`--max-jobs-per-gpu 3`。
- 直属主训练进程：24个；候选`D0/D5/E1/ADV3B02-BiCAD-XDC-V1`各6行，fold1/fold8各12行，seed392001/392002/392003各8行，24/24均绑定day1/2/3和5000 updates。
- GPU装箱：GPU0–7各3个本run训练进程；启动后利用率97%–99%，显存约2818–3368MiB，未发现无关compute process。
- 初始artifact：24个row目录、24个`train.log`已创建；`ARTIFACTS_COMPLETE=0`、`TECHNICAL_FAILURE=0`，未检出确定性异常指纹。训练健康运行，不因中间性能或日志缓冲停止。
- 当前边界：保持source-only；24行闭合并仅按source证据冻结候选/seed之前，禁止目标接收机day1–4测试及任何Phase2/query/truth访问。
