# d92_e0_full_ccoc_g0_k10_20260816_v6 Runner Handoff

## STATUS

ARTIFACTS_COMPLETE / G0_MECHANISM_RESOURCE_PASS / NO_PERFORMANCE_RESULT。

Luna/max为普通N607账户的sole runner。唯一detached launch=1，fresh-run retry=false；SSH启动通道未返回可采信exit，记为SSH exit=UNKNOWN，但远端持久化marker、driver与output/log artifact均闭合。未launch、未retry、未远端写入/删除、未触碰v1-v5。

## PRECHECK

- 直连普通N607 preflight=VERIFIED；项目根、固定Python、GPU0可见。
- 启动前同run PID=0；source、archive、launch、output、logs、launcher driver out/err和local retrieval根均ABSENT。
- archive=215970 bytes，SHA256=03721e4e082592dca6d8faf9716d2f2f70e9b6c14ad48bfef7c1ebd1bd699a38，39 members/35 files。
- launch=9322 bytes，SHA256=fe7a8a366540d1b992fb793e8982ac7b6e5273f3875d3f06cf01e1544e3194b4。
- tar safety、required entries、bash -n、Python3.10.19、Torch2.1.0+cu121、CUDA8卡、四actual seals均通过；after_apply完整SHA为afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a。

## SYNC与COMMAND

按archive→launch顺序SCP，远端size/SHA复核通过。唯一命令：

~~~bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_ccoc_g0_launch_2a77d164_20260816_v6.sh >./d92_ccoc_g0_launch_2a77d164_20260816_v6.out 2>./d92_ccoc_g0_launch_2a77d164_20260816_v6.err </dev/null &
~~~

远端driver约于2026-08-16 20:29:46 CST创建；SSH exit=UNKNOWN。独立artifact closure确认命令已执行并完成，未重试。

## PID_GPU

完成态精确pgrep：run/source均无活进程。GPU0=0%/1MiB，compute apps=0。

## G0_TECHNICAL_ARTIFACTS

- persisted status与validation.marker均为D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS。
- validation.pass=true；三scene键完整，scene_gates全部true。
- scene wall（ms）：clear=70.463259、low_elev=69.751364、rain=70.339019；scene ratio：1.126558、1.129636、1.142053；p90 wall=70.463259ms、p90 ratio=1.142053。
- canonical candidate after fit_audit的registration_incremental_peak_working_set_bytes：clear=729088B、low_elev=98304B、rain=81920B，max=729088B；512KiB target在clear scene未通过，1MiB hard在三scene均通过，G0技术状态按hard resource gate成立。
- 各scene active/fallback、rho、state、state_bytes、margin/quantum、query/state/MAC/resource门均true。
- query decision policy=per_sample_all_registered_classes；batch/global/quota/role-oracle/fit/selection/update/query-graph/truth-present/truth-used访问均false；query extra MAC=0，ground update=false。
- g0_driver.out=6881B、g0_driver.err=0B、launcher.out=41B、launcher.err=0B。
- 本runner未读取或解释accuracy、H、BA、floor、forgetting、truth内容或scorer结果。

## RETRIEVAL

冻结local retrieval根：E:/type10-7/local_artifacts/d92_e0_full_ccoc_g0_k10_20260816_v6。canonical remote/local manifest、count、bytes、tree hash闭合：

|组件|files|bytes|tree SHA256|
|---|---:|---:|---|
|source|69|1956312|89d5c6b606b754bc3cb18a332e933e7bc60f2e2dde5312edb0a9df5f02a86877|
|output|21|1451638|a058caa3167e129d5cd8a64fbac7b656ce7b513a5ba0d3d799530b9f48271e44|
|logs|8|8657|1d0e9492a934fb6f95f31c350f59d1cd76f152a0a688a52b69216626a71f2126|
|drivers|2|41|7aa8c9fdc385a03b4bc73c6de02e6bef58855eb7b3843d4f9b88ac9b6453d8f7|

source逐行manifest diff=0；output/logs/drivers tree hash直接相等。首次同basename递归SCP产生的合并副本保留但不纳入canonical manifest，未删除任何artifact；远端全部artifact保留。

## SSH_CLEANUP

每次SSH/SCP后均检查本地ssh.exe与TCP22；最终无ssh.exe、无N607/bridge的ESTABLISHED TCP22，仅可能存在TIME_WAIT。

## NEXT_ACTION

本run可交由主代理进行技术artifact分析，但不得改写为性能结果、不得重试或重新launch。NO_PERFORMANCE_RESULT保持不变。
