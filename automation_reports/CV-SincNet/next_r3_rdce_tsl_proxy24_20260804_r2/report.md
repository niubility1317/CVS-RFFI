# NEXT-R3 RDCE×TSL-160 Proxy24 r2实验报告

## 1. 基本信息

|字段|值|
|---|---|
|run ID|`next_r3_rdce_tsl_proxy24_20260804_r2`|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|
|日期|2026-08-04|
|主agent|Codex主agent（Sol/high）|
|唯一N607 runner|Luna/max|
|科学实现/独立审查|Terra/max；`P0=0、P1=0、READY`|
|协议/证据|`p2_min_v1`；`SOURCE_HELD_PROXY`；`formal_new_registration_claim=false`|

## 2. 目标与冻结矩阵

r2只修复r1预注册环境名与N607现场不一致的问题；方法、数据、seed、receiver、held class、K、四态、比较臂和停止规则全部不变。唯一候选为`R3-RDCE160×TSL-160`。矩阵为held receiver`1-1、18-2`×6个held class×`K1/K5`=`24 row`；每row生成`DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1`四态，共96 state、288 state-arm prediction。指标统一称为域适应前/后、新类注册前/后；REG0的new/H为`N/A`。

主因果量固定为`R1Q-R0Q`、`R0L-R0Q`和`[R1L-R1Q]-[R0L-R0Q]`；`R0L-R0F、R1L-R1F`仅是全管线替换比较。不得选择性删row、换seed或按中间性能停止。

## 3. r1处置与r2唯一改动

r1已完成archive落地和解包，但N607不存在`/home/szu2070436088/.conda/envs/ssr-gpu/bin/python`，故在`py_compile`前停止，状态为`LANDED_NOT_STARTED_WRONG_ENVIRONMENT / NO_PERFORMANCE_RESULT`。r1 root永久保留，不覆盖、不恢复。

r2使用N607现有、此前D106/D122/D130实验已实际使用的`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`（Python3.10.19）。首次运行只做`py_compile`和必要import smoke；通过后立即prepare和真实单行smoke，不增加其它gate。

## 4. 本地版本与验证

|字段|值|
|---|---|
|runtime source commit|`f35c3cdf8737068f5aca2e9b2ddcb164a7579bf2`|
|report lineage commit|`34d05e6f`及本报告后续commit|
|runtime archive|`E:\type10-7\code\snapshots\d92_lite125_20260804_wt\automation_reports\CV-SincNet\next_r3_rdce_tsl_proxy24_20260804_r1\release\next_r3_runtime_f35c3cdf.tar.gz`|
|archive SHA/size|`151c0f1d76ad8b0d373e318fdef38149d412980ae1bbca5500662bc4bfc01abf`；6,475,647B|
|本地验证|`ssr-gpu`下`py_compile`通过；runner、NEXT-R3核心、D129 head回归共`40 passed`|
|独立复核|predict package无truth/全量标签/双query/role字段；共同query、checkpoint bridge、延后score闭合；`P0=0、P1=0`|

## 5. 真实资产

|资产|路径|SHA256|
|---|---|---|
|received-IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|IQ receipt|同目录`d106_ls_received_iq.receipt.json`|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|strict tap/cells|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/strict_tap/d106_ls_strict_tap.npz`|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|tap receipt|同目录`d106_ls_strict_tap.receipt.json`|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|RDCE wire|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`|`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|

authority固定为`capsule_id=e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`、`split_id=6d57b511e95382b0c1ebfad91b2a0de4be5c08d5896e0625aa3a4dfa110b5134`、`validator_receipt_sha256=2282942170a2bbd03aba904fe88d9e33840873c481d20be406ef54b50aa4fbfc`。

## 6. N607精确发布

|字段|值|
|---|---|
|remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r3_rdce_tsl_proxy24_20260804_r2`；创建前必须`ABSENT`|
|archive mapping|上述本地archive→`<root>/input/next_r3_runtime_f35c3cdf.tar.gz`|
|CWD|`<root>/source`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU|物理GPU0；`CUDA_VISIBLE_DEVICES=0`，进程内`cuda:0`|
|日志|`<root>/logs/{compile,prepare,smoke,run.out,run.err}`；PID=`<root>/logs/main.pid`|
|输出|`<root>/prepare`、`<root>/smoke`、`<root>/output`|

精确命令使用r1报告§7.1的全部参数和SHA，仅作以下两项逐字替换：

1. `ROOT=.../next_r3_rdce_tsl_proxy24_20260804_r1`→`ROOT=.../next_r3_rdce_tsl_proxy24_20260804_r2`；
2. `PY=/home/szu2070436088/.conda/envs/ssr-gpu/bin/python`→`PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。

执行顺序固定为：远端archive hash/解包→`py_compile`和必要import→foreground prepare→foreground单行`SMOKE_NO_TRUTH`→detached完整24行predict→确认24/24、96/96、288/288且truth未打开→score读取本次`prepare/truth.json`。fresh-run retry未授权。

## 7. 停止与成功条件

只在P0协议/安全违规，或至少两个不同row在产出prediction前出现相同确定性exception fingerprint时停止本run进程树；不得按accuracy、H、floor或任何性能值早停。成功条件为prepare三文件完整、真实checkpoint单行smoke通过、24 row/96 state/288 arm完整、query fit/update/selection全为0、score最后才打开truth。

## 8. 完成后结果表

|状态/比较|K|receiver|old/retained BA|held-proxy BA|H|old floor|all floor|总正确数|资源/时延|结论|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
|域适应前/新类注册前|—|—|PENDING|N/A|N/A|PENDING|N/A|PENDING|PENDING|PENDING|
|域适应后/新类注册前|—|—|PENDING|N/A|N/A|PENDING|N/A|PENDING|PENDING|PENDING|
|域适应前/新类注册后|—|—|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
|域适应后/新类注册后|—|—|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|

完成后补充24行same-row表、K/receiver/class分层、资源、异常、PID/GPU/SSH断开证据和是否继续该方法；不得用边际最大值拼接性能结论。

## 9. r2实际执行结果

preflight、r2 root`ABSENT`、GPU0资源检查、archive同步/hash/size、CVS-RFFI Python3.10.19下`py_compile`和必要import均通过。foreground prepare随后在首个TSL prior的physical-LOO几何构造中以exit=1退出，唯一异常指纹为`NextR3TSL160Error: empirical-Bayes diagonal variance fell below its frozen floor`，调用链为`_build_prior_from_cells→build_tsl160_phase1_prior→_geometry`。没有生成prepare三JSON，未执行smoke、完整predict或score，truth未打开，无性能结果。

代码审计显示`_prior_from_cells`在封存前已按同一规则将`v0`夹到正floor，但`int8 logv0＋FP16 scale/offset`wire解码可能使最小项轻微下穿；`_geometry`把这一可恢复的量化边界当成非法输入直接退出。当前只允许一次非调参修复：对有限正`v_post`按原有同一floor夹紧，保留0/NaN/Inf拒绝；不改floor系数、nu0、rho、方法、矩阵或阈值。修复须经独立复核后使用全新不可覆盖run；若下一次仍出现技术失败，关闭TSL路线，不再追加发布轮次。

r2 partial root保留`input/source/logs/{compile,prepare}.log`；无runner PID，GPU0=`0%/1MiB`，SSH收尾为`ssh.exe=0`、N607:22连接=0。
