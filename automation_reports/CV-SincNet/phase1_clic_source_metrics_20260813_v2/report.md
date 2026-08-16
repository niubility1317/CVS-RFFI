# Phase1 CLIC源域指标补全v2运行报告

## 预注册与边界

- 实验ID：`phase1_clic_source_metrics_20260813_v2`。
- 角色：Luna/max唯一N607 runner；本报告只记录冻结发布、smoke、formal技术证据及同一run的结果。
- 冻结Git full hash：`549a4b3a95d96556d849c6324fbde8601359379a`；archive不得包含dirty/HEAD漂移。
- 目标：在不修改方法、阈值、fold、指标、GPU矩阵或任何target/query边界的前提下，发布并运行source-only metrics v2。
- 性能读取边界：运行与技术QA阶段不读取target/query/truth结果；仅在全部技术artifact闭合后读取source clean与三LEO source known指标及6-fold C/G配对字段。
- `SMOKE_INVOCATION=1`、`FORMAL_INVOCATION=1`各最多一次；`retry=NO`。任何路径存在、authority不闭合或远端状态不确定时停止并上报，不覆盖、不重试。

## 冻结路径

| 项目 | 冻结路径 |
|---|---|
| release | `/home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260813_v2_549a4b3a` |
| formal run | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_metrics_20260813_v2` |
| formal log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_source_metrics_20260813_v2` |
| formal outer | `/home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_metrics_20260813_v2_outer.out` |
| smoke run | `/home/szu2070436088/2510044040/CV-SincNet/runs/.smoke_phase1_clic_source_metrics_20260813_v2_F1` |
| smoke log | `/home/szu2070436088/2510044040/CV-SincNet/logs/.smoke_phase1_clic_source_metrics_20260813_v2_F1` |
| smoke outer | `/home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_metrics_20260813_v2_smoke_outer.out` |

## 运行合同

- smoke：仅F1、共享cache、`F1C→F1G`串行forward；不score、不读性能。
- formal：`6 shared cache → 12 forward → 6 pair score → 1 aggregate`；dry-run应为25行。
- GPU：formal cache/forward固定GPU0..5；每GPU最多两个forward；score/aggregate仅CPU；smoke仅GPU0且C/G串行。
- 技术停止条件：仅协议/访问/哈希/覆盖风险、错误checkout、launcher-wide确定性技术异常、或至少两个不同fold在无prediction前复现同一确定性异常；不得因性能停止。
- 预期artifact：6 shared-cache NPZ+receipt、12 feature NPZ+binding、6 pair metrics、1 aggregate；需满足source-only、target=query=0、fit=0、selection=0、finite、物理ID/scene闭合、C/G共享cache、生产reopen。

## 发布与验证记录

| 阶段 | 状态 | 证据 |
|---|---|---|
| local | `LOCAL_VERIFIED` | 待记录冻结archive及本地Git/Bash验证 |
| landing | `PENDING` | 待记录单次SCP、远端tar bytes/SHA、stage解包与atomic rename |
| smoke | `PENDING` | `SMOKE_INVOCATION=0`；待记录唯一PID/CWD/cmdline/GPU/日志与receipt |
| formal | `PENDING` | `FORMAL_INVOCATION=0`；仅在ADV正式训练自然完成且GPU0..5清空后启动 |
| artifacts | `PENDING` | 待记录技术QA；未闭合前不读性能 |
| analysis | `PENDING` | 待记录source-only同row结果；不含target/query/truth结果 |

### 2026-08-16本地预检与archive

- Git Bash首检：`MSYSTEM=MINGW64`；所有本地命令均使用`C:\\Program Files\\Git\\bin\\bash.exe`、`login=false`。
- Git承载面：`E:/type10-7/github_publish/CVS-RFFI-repo`；该树HEAD为后续`b2e0f30c`且存在他人未跟踪文件，本runner未stage或修改它们。发布源严格为full hash`549a4b3a95d96556d849c6324fbde8601359379a`。
- 本地N607直连只读预检：`VERIFIED`。远端`dell-DSS8440`，项目根可见；ADV v2正式主PID`801059`仍活跃，GPU0..5各1个训练进程（PID`801089,801092,801095,801100,801103,801106`），GPU6..7空闲；因此smoke/formal均未调用。
- 预期release、formal run/log/outer及smoke run/log/outer均为`ABSENT`；未预建任何run/log路径。
- 冻结Git canonical SHA已逐项闭合：builder=`c9e9cbef3d51537c0946f4ec280e68c4e0f6c7bcdf6fc5495eb279a61eeb4169`；exporter=`2b1367a2508a503795d4c193264615e67e348eedfc46c4bd731803ccd18341b7`；scorer=`e4eb7ca70e2d0dc1656a733f92359e21d41e4e60d52deb554ff3843e1683c3f9`；formal launcher=`a7182ca1835a3caa87a4397d5b552c795796ded0e97f1e3d920c0f49ae3ca05f`；smoke launcher=`0470ffb627bd7ee059d4a9f722a770346c6c35624ad1a9373205f59ee8f93b6c`；冻结报告blob=`9c6a17e718f931407362bb1aebf88f18ac8c6ffcffa396597b88e29cb709da37`。
- 唯一archive：`E:/type10-7/code/runner_release_phase1_clic_source_metrics_20260813_v2_549a4b3a.tar`；未本地解包；SHA=`3f6c7862529e1f18ebd62741cac967dbe339f10a753ce674b541099a4afd5f23`；大小`268257280` bytes；目录条目`5107`。archive中未发现`Dataset_WigSig`、`ManySig.pkl`、`runs/`、`source_clean_proxy.npz`或真实checkpoint路径；两个约16MiB历史JSON为冻结commit既有tracked报告artifact，未改动。
- 本阶段状态：`LOCAL_VERIFIED`；`SMOKE_INVOCATION=0`；`FORMAL_INVOCATION=0`；`retry=NO`；`NO_PERFORMANCE_RESULT`。
- 远端SCP前只读门：`VERIFIED`；`releases`父目录存在，唯一incoming archive目标和最终release均`ABSENT`。已授权且仅执行一次SCP，目标为`/home/szu2070436088/2510044040/releases/.phase1_clic_source_metrics_20260813_v2_549a4b3a.tar.incoming`。

### 远端landing与静态门

- 唯一SCP：`VERIFIED`，返回码0；本地未留下ssh/scp进程或到N607的`ESTABLISHED`连接。
- 远端incoming bytes=`268257280`，SHA=`3f6c7862529e1f18ebd62741cac967dbe339f10a753ce674b541099a4afd5f23`，与本地archive完全一致。
- 远端release：从incoming解包到独占stage后atomic rename，最终路径为`/home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260813_v2_549a4b3a`；未创建formal或smoke run/log根。
- 远端提取文件raw/LF-normalized SHA：builder raw=`eb26e2061e45dc6cc63454477e1a4019b38ec687b2adaab1b90a2c90b3ff1806`/LF=`c9e9cbef3d51537c0946f4ec280e68c4e0f6c7bcdf6fc5495eb279a61eeb4169`；exporter raw=`033f5d409f9077d55c441393f2fbb206b1b405d9de4fe47f1bfb9a57dfe44b21`/LF=`2b1367a2508a503795d4c193264615e67e348eedfc46c4bd731803ccd18341b7`；scorer raw=`d60ba165a511be8e881e8c8f0f2835a037a0cf7c199fbcd88e30e654fff464f7`/LF=`e4eb7ca70e2d0dc1656a733f92359e21d41e4e60d52deb554ff3843e1683c3f9`；formal/smoke launcher raw与LF分别均等于冻结SHA；冻结报告raw=`829f8c0b0653fa90deee467eb63a6c7c128119119e0785293695d29aabdaa7db`/LF=`9c6a17e718f931407362bb1aebf88f18ac8c6ffcffa396597b88e29cb709da37`。CRLF差异已显式记录，未误判。
- 远端Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；3个Python入口`py_compile`通过；builder/exporter/scorer三个`--help`均成功；两份launcher`bash -n`通过；formal/smoke dry-run行数为`25/3`；`--target/--query/--truth/--prediction/--package`禁用flag匹配为`0`。
- 当前阶段：`LANDED / STATIC_VERIFIED`；`SMOKE_INVOCATION=0`；`FORMAL_INVOCATION=0`；`retry=NO`；`NO_PERFORMANCE_RESULT`。远端静态临时目录为`/tmp/phase1_clic_source_metrics_v2_static.pNxk1K`，不在release树中。

## 运行证据

待runner追加：UTC/本地时间、远端hostname/date、ADV占用、release/archive和文件哈希、exact命令、PID/CWD/cmdline、GPU映射、stage屏障、完整artifact路径/SHA/row/finite/物理ID/scene、异常及最终判定。

## 版本边界

本runner只拥有本报告；不得stage/commit`conversation_index/`或其他agent文件。所有方法、代码、脚本、阈值和矩阵均取冻结full hash，不在本runner阶段改动。
