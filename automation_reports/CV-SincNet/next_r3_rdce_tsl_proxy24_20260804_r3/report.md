# NEXT-R3 RDCE×TSL-160 Proxy24 r3最终实验报告

## 1. 基本信息

|字段|值|
|---|---|
|run ID|`next_r3_rdce_tsl_proxy24_20260804_r3`|
|状态|`PRE_REGISTERED / FINAL_NUMERIC_REPAIR`|
|日期|2026-08-04|
|主agent|Codex主agent（Sol/high）|
|唯一N607 runner|Luna/max|
|科学实现/独立审查|Terra/max；runner与floor修复均`P0=0、P1=0、READY`|
|协议/证据|`p2_min_v1`；`SOURCE_HELD_PROXY`；`formal_new_registration_claim=false`|

## 2. 目标、矩阵与指标

唯一候选仍为`R3-RDCE160×TSL-160`。r3不改方法公式、floor系数、nu0、rho、seed、receiver、class、K或比较臂；只修复TSL prior的INT8-log/FP16 wire在解码端把已夹紧正端点轻微量化到同一floor下的问题。

冻结矩阵为held receiver`1-1、18-2`×6类×`K1/K5`=`24 row`，输出96 state、288 state-arm prediction。结果统一按四态命名：

|状态|中文名称|new/H|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|`N/A`|
|`DA1_REG0`|域适应后/新类注册前|`N/A`|
|`DA0_REG1`|域适应前/新类注册后|报告held-proxy new/H|
|`DA1_REG1`|域适应后/新类注册后|报告held-proxy new/H|

主因果量固定为`R1Q-R0Q`、`R0L-R0Q`和`[R1L-R1Q]-[R0L-R0Q]`。不得选择性删row、换seed或按中间性能停止。

## 3. 前两次技术退出与本次唯一修复

|run|状态|事实|
|---|---|---|
|r1|`LANDED_NOT_STARTED_WRONG_ENVIRONMENT / NO_PERFORMANCE_RESULT`|N607无`ssr-gpu`，在py_compile前停止|
|r2|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|CVS-RFFI编译/import通过；prepare因`v_post`轻微低于既有floor退出；无prepare artifact/smoke/prediction|

commit`640ed75c1e21985a6d063dce4920e73eb9adaeaf`只把有限正`v_post`夹回原有同一floor；非有限或`<=0`仍fail-closed。`64×float32 eps`、nu0、rho和全部科学路径未改。测试真实经过`_quantize_logv0`的INT8-log/FP16编码/解码复现下穿，不是mock。独立复核`P0=0、P1=0、READY`；主agent串行回归`43 passed`并通过py_compile。

r3是该候选最后一次发布；若再次出现任何阻止完整结果的技术失败，直接关闭TSL路线，不创建r4。

## 4. Git、release与环境

|字段|值|
|---|---|
|runtime commit|`640ed75c1e21985a6d063dce4920e73eb9adaeaf`|
|runtime archive|`E:\type10-7\code\snapshots\d92_lite125_20260804_wt\automation_reports\CV-SincNet\next_r3_rdce_tsl_proxy24_20260804_r1\release\next_r3_runtime_640ed75c.tar.gz`|
|archive SHA/size|`b1fdb3e31ca5b38a93672cf3bc38c3e23e30a1445fe5c722260aa7d921f10e3a`；6,483,227B|
|remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r3_rdce_tsl_proxy24_20260804_r3`；创建前必须`ABSENT`|
|remote archive|`<root>/input/next_r3_runtime_640ed75c.tar.gz`|
|CWD|`<root>/source`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`（Python3.10.19）|
|GPU|物理GPU0；`CUDA_VISIBLE_DEVICES=0`，进程内`cuda:0`|

## 5. 真实输入

|资产|绝对路径|SHA256|
|---|---|---|
|received-IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|IQ receipt|同目录`d106_ls_received_iq.receipt.json`|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|strict tap/cells|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|tap receipt|同目录`d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|RDCE wire|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`|`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|

authority固定为`capsule_id=e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`、`split_id=6d57b511e95382b0c1ebfad91b2a0de4be5c08d5896e0625aa3a4dfa110b5134`、`validator_receipt_sha256=2282942170a2bbd03aba904fe88d9e33840873c481d20be406ef54b50aa4fbfc`。

## 6. 精确发布与健康控制

精确命令使用r1报告§7.1的全部参数和SHA，仅逐字替换：run ID/root为r3；Python为上述CVS-RFFI解释器；同步archive为`next_r3_runtime_640ed75c.tar.gz`及本报告SHA。执行顺序固定为archive hash/解包→py_compile/import→foreground prepare→foreground单行`SMOKE_NO_TRUTH`→detached完整24行predict→确认24/96/288且truth未打开→score读取本次`prepare/truth.json`。

日志固定为`<root>/logs/{compile,prepare,smoke,run.out,run.err}`，PID为`<root>/logs/main.pid`；输出为`<root>/prepare`、`<root>/smoke`、`<root>/output`。只在P0协议/安全违规，或至少两个不同row在prediction前出现相同确定性exception fingerprint时停止；不得按accuracy、H或floor早停。fresh-run retry未授权。

## 7. 结果模板

|状态/比较|K|receiver|old/retained BA|held-proxy BA|H|old floor|all floor|总正确数|资源/时延|结论|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
|域适应前/新类注册前|—|—|PENDING|N/A|N/A|PENDING|N/A|PENDING|PENDING|PENDING|
|域适应后/新类注册前|—|—|PENDING|N/A|N/A|PENDING|N/A|PENDING|PENDING|PENDING|
|域适应前/新类注册后|—|—|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
|域适应后/新类注册后|—|—|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|

完成后必须补24行same-row表、K/receiver/class分层、资源、异常、PID/GPU/SSH断开证据和是否关闭/继续；不得拼接边际最大值。

## 8. r3最终执行结果（2026-08-04）

|阶段|状态|证据|
|---|---|---|
|远端落盘|`LANDED`|root重新从`ABSENT`创建；archive 6,483,227B，SHA=`b1fdb3e31ca5b38a93672cf3bc38c3e23e30a1445fe5c722260aa7d921f10e3a`；独立连接复核source与archive仍存在|
|compile/import|`PASSED`|Python3.10.19；`PY_COMPILE_AND_IMPORT_OK`；`compile.log` SHA=`6638d1659b2dd718474795a1bb70e197e460ed232be9861ba2b84852741e8098`|
|prepare|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|`PREPARE_EXIT=1`；`NextR3TSL160Error: physical LOO fold has no correctly classified reference margin`；位置`stage2_next_r3_tsl160.py:755`，发生于`_physical_loo_radius`，无prediction|
|smoke/full predict|`NOT_STARTED`|因prepare未产出package，未执行smoke、24-row predict或score；无r4、无fresh-run retry|

prepare日志固定在远端`<root>/logs/prepare.log`，大小1,996B，SHA=`5e198c17a92b2af68a094f3bb53e50791b8503ccaeb2afc28e6bdaeab11aa645`。远端top-level仅`input/`、`source/`、`logs/`；`prepare/predictor_package.json`、`prepare/truth.json`、`prepare/prepare_receipt.json`、`smoke/`、`output/`、`logs/main.pid`、`logs/run.out`、`logs/run.err`均不存在。prepare结束后匹配进程为空；GPU0为`0% util, 1MiB/24576MiB`；本地`ssh.exe`与到`172.31.111.215:22`的ESTABLISHED连接均为0。

两个远端日志已只读拉回`release/remote_evidence/{compile.log,prepare.log}`，本地SHA分别与远端`6638d1659b2dd718474795a1bb70e197e460ed232be9861ba2b84852741e8098`和`5e198c17a92b2af68a094f3bb53e50791b8503ccaeb2afc28e6bdaeab11aa645`一致。

**最终判定：**r3为该候选最后一次发布。由于TSL物理LOO参考margin异常在prepare阶段阻断，TSL路线关闭；本run不产生任何性能、注册、DA或promotable结论。保留远端root、archive、源码和日志，不删除任何既有r1/r2或输入资产。
