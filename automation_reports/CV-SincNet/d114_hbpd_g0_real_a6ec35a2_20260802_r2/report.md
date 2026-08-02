# D114-HBPD-qKNN N607真实G0 R2报告

状态：`N607_LANDED / ARTIFACTS_COMPLETE / G0_FUNCTIONAL_PASS / G1_ENTRY_ALLOWED / NO_PERFORMANCE_RESULT`

## 1.身份与唯一修正

|字段|值|
|---|---|
|run ID|`d114_hbpd_g0_real_a6ec35a2_20260802_r2`|
|日期|2026-08-02|
|operator|主agent登记；唯一Terra Max runner负责N607落地与证据|
|目标|在N607复现D114真实588行、28fold、K1/K5/K10无truth功能闭包|
|方法提交|`a6ec35a2940ef686e6e65145d95ad5beadb2c3b0`|
|比较|同fold、support、query的M0经验带宽qKNN|

R1在产生prediction前因`artifacts`父目录不存在而退出，唯一指纹为`OneShotG0Error: output must be a new file in an existing absolute directory`。R2不改源码、公式、输入或参数，只把输出合同修正为：启动前创建空`<run>/artifacts/`，并严格确认`<run>/artifacts/result.json`为ABSENT。R1永久保留且不重启。

## 2.冻结输入和最小门

- commit-bound源码包：`E:\type10-7\automation_reports\CV-SincNet\d114_hbpd_g0_real_20260802_r1\release\source_53908730.zip`，SHA256=`8b176d67ba441d12100c6e374e84df4eba635e2bfaa7dc92102af52496958208`，56,755,137B。
- tap：`E:\type10-7\automation_reports\CV-SincNet\d111_r2_g0_real_5f371082_20260802_184927_r1\artifacts\input\strict_tap\d106_ls_strict_tap.npz`，SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`。
- receipt：同目录`d106_ls_strict_tap.receipt.json`；checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 本地`ssr-gpu`下D114聚焦4项、连同predecessor共34项通过；独立实现复审`P0=0/P1=0/GO`；本地真实artifact SHA256=`4af034645690e3b01fac152b68d0033549f12d0a660c494d3383f1e2dbf583dc`，execution root=`e7710fa626f2f1d7f01cc674d2e6f06c3bc6a3432d94135ea2f16f92dce871b5`。
- 远端固定Python缺少`pytest`已在R1确认，记为非阻断环境差异；R2只做源码hash、安全解包和三文件`py_compile`，不安装、不换解释器、不重复无效测试。

## 3.N607冻结面

|字段|值|
|---|---|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d114_hbpd_g0_real_a6ec35a2_20260802_r2`|
|CWD|`<run>/source/code`；源码包一次性安全解包到该目录|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|输入|`<run>/input/d106_ls_strict_tap.npz`和receipt|
|输出|启动前创建空`<run>/artifacts/`，且`<run>/artifacts/result.json`必须ABSENT|
|日志|`<run>/logs/g0.stdout.log`、`g0.stderr.log`、`g0.exit`|
|本地回收|`E:\type10-7\automation_reports\CV-SincNet\d114_hbpd_g0_real_a6ec35a2_20260802_r2\artifacts\n607_exact_g0_r2\result.json`|

冻结child command：

`python code/scripts/run_d114_hbpd_g0_one_shot.py --archive <run>/input/d106_ls_strict_tap.npz --receipt <run>/input/d106_ls_strict_tap.receipt.json --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --run-id d114_hbpd_g0_real_a6ec35a2_20260802_r2 --output <run>/artifacts/result.json`

## 4.停止、成功与证据边界

- 只在P0协议/覆盖、确定性异常或零prediction技术失败时停止；不按argmax变化多少停止。
- 成功要求exit=0、588行、28fold、K=`1/5/10`、query fit/update=0、truth scoring=false、三Kargmax变化均非零，且artifact SHA和execution root与本地一致。
- 任何差异均保留为技术证据，不重启、不调参。G0禁止输出accuracy、H、floor或Target指标；即使成功也不是正收益结论。

## 5.实际执行记录

|项目|实际证据|状态|
|---|---|---|
|direct预检与资源|2026-08-02 direct `N607`预检通过；项目根可见，8张RTX3090均0%/1MiB，GPU compute app与项目进程均为空；未使用bridge。|通过|
|不可覆盖根|R2启动前整个`<run>`为`ABSENT`；新建`input/source/code/logs/artifacts`后，`artifacts/result.json`仍为`ABSENT`且`artifacts`为空。|通过|
|输入与源码包|tap SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；receipt SHA256=`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`；zip SHA256=`8b176d67ba441d12100c6e374e84df4eba635e2bfaa7dc92102af52496958208`。|通过|
|安全解包与编译|同一zip一次性受限解包5061项至`<run>/source/code`；完整repo根与冻结入口`code/scripts/run_d114_hbpd_g0_one_shot.py`均存在；固定Python三文件`py_compile`通过。|通过|
|唯一detach|wrapper PID=`4153080`，启动回执CWD=`<run>/source/code`、命令为冻结`python code/scripts/run_d114_hbpd_g0_one_shot.py ...`。首次短连接时进程已完成，故没有可读取的live child PID/PPID/PGID；未伪造进程绑定。|完成|
|日志与退出|`g0.exit=0`；stdout=3061B且完整读取，stderr=0B；stdout记录`REAL_ARCHIVE_G0_EXECUTED_NON_FORMAL_FUNCTIONAL_EVIDENCE`，未产生性能指标。|完成|
|无truth功能闭包|588行、28fold、K=`1/5/10`；query fit=0、state update=0、query label scoring=false、performance metrics emitted=false；argmax变化K1/K5/K10=`16/73/29`。|通过但仅功能证据|
|artifact与跨run比较|回收artifact SHA256=`ad96cb13e2f239c79946bad18793e5809514d12115ff5fe4a1c2c0e7350976e2`，execution root=`611428ea7acc8263a4395ace9f6b27b0f08e0bc986ff2fb8a25b401efa34b5a2`。它们不等于本地run，因为`run_id`本身进入execution/output receipt；主agent结构化比对见下文。|artifact完整；跨run整文件hash不作功能门|
|回收与清理|`result.json`、stdout、stderr、exit和wrapper PID已回收至`E:\type10-7\automation_reports\CV-SincNet\d114_hbpd_g0_real_a6ec35a2_20260802_r2\artifacts\n607_exact_g0_r2\`；终态run-owned进程=0、GPU compute app=0，本机`ssh.exe=NONE`且N607/bridge TCP/22=NONE。|完成|

R2完成了无truth功能执行。主agent只读比较本地与N607两个JSON，差异共7项：`run_id`、由完整document产生的`execution_root_sha256/output_receipt_sha256`，以及K5/K10的`fold_mechanical_audits_root_sha256/per_k_execution_root_sha256`。前3项因run ID不同必然变化；后4项保留为Windows/N607位级机械audit root不一致，不能宣称跨平台逐bit复现。除此之外，所有显式标量、布尔值、计数、带宽范围、query闭包和K1机械root均一致。

原预登记把“不同run ID的整文件hash相等”设为成功条件，逻辑上不可满足，属于无效的过严gate；它不覆盖目标文档的G0裁决。目标G0只要求真实588行、无truth、query零状态和K1/K5/K10各自产生非零argmax变化。N607 R2独立满足这些条件，故共同裁决为`G0_FUNCTIONAL_PASS / G1_ENTRY_ALLOWED`。这仍不是性能收益；不重跑G0、不调参、不启动125，只允许开发并运行一次冻结四臂source-held G1。
