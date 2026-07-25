# D103-R2正式Phase1-held预注册交接

状态：`PREREGISTERED / INPUTS_BOUND / N607_GPU_STACK_BLOCKED / N607_LAUNCH_NO_GO / TARGET25_NO_GO`

实验ID：`d103_r2_rxid_phase1held_20260725_r1`

完整本地报告：`E:\type10-7\automation_reports\CV-SincNet\d103_r2_rxid_phase1held_20260725_r1\report.md`

## Release证据

- candidate：`D103-R2-RXID-CROSSRECEIVER-MB4`
- 核心实现commit：`59978e44`
- release hardening增量commit/当前HEAD：`80f58ce5`
- 独立复审index SHA256：`30c8c98ff8fcdf2915f4c2e797c605cecc98d138e95ad3b0bc6e542faf9fdc9b`
- 首轮release复审：`REVIEW_GO / P0=0 / P1=0 / P2=2`
- 增量release复审index SHA256：`298dd85d3f8258f0942c7a09e1fb842a51bff1c2ab817a9a43833fd124a89cad`
- 增量release复审：`GO / P0=0 / P1=0 / P2=0`
- 本地验证：67项D103定向测试通过；36个Python文件编译解析通过；真实tap/dual 400step无query-truth smoke通过。
- 无Git push或远程Git上传。

## 冻结实验

- `protocol_schema=p2_min_v1`
- source split：8400→588/5292/2520；42个receiver×TX组各`L_s=14`；leave-day后10–12。
- fit：1 final+49 outer+196 leave-day=246fit、98,400step。
- held：63性能行、49稳定性行，每个稳定性行4个实际160维shift。
- matched M0/D102/D103；D102保持`DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE`。
- 每GPU最多2个worker；总GPU时≤30、显存≤4GiB/fit、run-root≤20GiB。
- 不读取Target；held接受前`TARGET25_NO_GO`。

## 首次N607只读交接结果

- route：direct N607成功；bridge未使用；
- `source_train/cache_set.json`SHA256=`d719808ceaed07c13f6c8d8053acf910a61904243ec1d47c28cc4e4b679cffd2`；
- `cache_scope=source_train`、`roles=[source]`，3个`leo_*_weak`成员存在、非symlink且实际SHA与manifest全部一致；
- selection salt、base runtime、dual export receipt、base parity receipt和checkpoint五项实际SHA全部等于冻结值；
- 首次检查时18个远端release目标全部`ABSENT`；增量commit`80f58ce5`改变两个脚本SHA，GPU恢复后必须重新检查全部18路径；
- `/home`可用8,138,337,734,656B；
- 未发现GPU设备使用者或目标训练进程，但不能替代NVML进程表；
- 最终`ssh.exe=0`，N607和bridge的ESTABLISHED连接均为0；
- 未sync、mkdir、创建run-root、启动或停止任何进程。

完整逐路径表在根目录正式报告中。输入绑定与release hardening均已进入本地Git提交；当前禁止sync和启动的独立原因是N607 GPU栈阻塞。根目录报告在2026-07-25对这项状态说明作了文字校正，当前SHA256=`925470bf1fa63086ecf65d28737fefb7ece5f35f1b38a0490bdb5ac900cb8bd0`；内容未改变候选、矩阵、输入SHA、停止规则或启动门。

## 当前启动阻塞

N607内核驱动=`535.309.01`，`/usr/bin/nvidia-smi`运行时解析NVML=`580.173.02`，返回`Driver/library version mismatch`、exit18。清除`LD_LIBRARY_PATH/CUDA_HOME`后仍失败，系统目录没有535.309.01匹配库。逐卡GPU利用率、显存和计算进程表无法形成可信证据，因此保持`N607_LAUNCH_NO_GO`。

本轮不修改symlink、包、内核模块、服务，不重启。GPU栈由服务器维护面恢复后，唯一运行代理必须重新执行完整preflight和占用检查；在此之前禁止sync和启动。

## 预注册停止规则

不依据性能停止。仅在P0协议/安全错误，或两个不同fit在prediction前产生同一规范化确定性异常指纹时，停止dispatch并终止三重绑定的本run PID。保留全部partial artifacts，不自动重试，不覆盖原run。
