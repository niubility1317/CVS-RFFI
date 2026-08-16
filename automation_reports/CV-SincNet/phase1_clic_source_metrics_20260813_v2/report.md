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
| local | `LOCAL_VERIFIED` | 冻结full hash、canonical SHA、archive bytes和Git Bash静态门已闭合 |
| landing | `LANDED` | 唯一SCP、remote tar bytes/SHA、stage解包和atomic rename已闭合 |
| smoke | `SMOKE_STOPPED_TECHNICAL_FAILURE` | observed PID`835435`已退出；cache阶段确定性技术异常；`RETRY=NO` |
| formal | `FORMAL_INVOCATION=0` | formal run/log/outer均`ABSENT`，未启动 |
| artifacts | `INCOMPLETE / NO_PERFORMANCE_RESULT` | 预期6个smoke artifact均缺失，未读取性能 |
| analysis | `NOT_PERFORMED` | 不存在可分析的完整同row结果；未读取target/query/truth |

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

### Smoke预启动门

- 正式ADV v2仍为外部活跃训练，runner只做monitor-only；按冻结资源规则，F1 smoke仅使用GPU0且C/G串行，允许在GPU0已有1个ADV进程时总进程数为2。
- 唯一预注册命令：`nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260813_v2_549a4b3a/code/scripts/smoke_phase1_clic_source_metrics_f1_v2_20260816.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_metrics_20260813_v2_smoke_outer.out 2>&1 &`。
- 启动前硬门：smoke run/log/outer均必须`ABSENT`；`SMOKE_INVOCATION`从0变1且`retry=NO`；不读取性能，不启动formal。

## 运行证据

待runner追加：UTC/本地时间、远端hostname/date、ADV占用、release/archive和文件哈希、exact命令、PID/CWD/cmdline、GPU映射、stage屏障、完整artifact路径/SHA/row/finite/物理ID/scene、异常及最终判定。

## 版本边界

本runner只拥有本报告；不得stage/commit`conversation_index/`或其他agent文件。所有方法、代码、脚本、阈值和矩阵均取冻结full hash，不在本runner阶段改动。

## 最终封存结论

- 最终状态：`SMOKE_STOPPED_TECHNICAL_FAILURE / FORMAL_INVOCATION=0 / NO_PERFORMANCE_RESULT / RETRY=NO`。本报告封存阶段未访问N607、未改代码、未删除或覆盖任何run artifact。
- 冻结release一致性：本地archive SHA=`3f6c7862529e1f18ebd62741cac967dbe339f10a753ce674b541099a4afd5f23`、bytes=`268257280`；远端incoming同bytes/SHA。release owner=`szu2070436088:szu2070436088`，mtime=`2026-08-16 16:19:57.089679173 +0800`；incoming owner相同，mtime=`2026-08-16 16:17:50.734670226 +0800`；stage路径已不存在。incoming tar member与release关键builder/exporter/scorer/formal launcher/smoke launcher/report的raw SHA逐项`match=1`，且LF-normalized SHA分别匹配冻结任务值，故可证明release来自同一冻结`549a4b3a95d96556d849c6324fbde8601359379a`。
- Smoke路径证据：smoke base mtime=`2026-08-16 16:21:40.818686518 +0800`；source root mtime=`16:21:40.819686518 +0800`；log root mtime=`16:21:40.823686518 +0800`；outer owner为`szu2070436088:szu2070436088`、size=`0`、mtime=`2026-08-16 16:21:35.270686125 +0800`。maxdepth3树仅有F1共享、F1C、F1G三个空叶目录及其父目录，无任何NPZ、receipt或binding文件。
- Smoke调用证据：唯一观测到的启动回执为`SMOKE_LAUNCH_PID=835435`；首个独立核验时该PID已退出且无source-metrics进程。由于没有持久化调用计数器，persistent unique-invocation proof=`UNKNOWN`；但路径由脚本独占创建，未发现第二个PID、第二份outer或第二次路径创建证据。不能把该事实升级为完整唯一调用证明。
- 预期artifact闭合：`0/6`，以下路径均`ABSENT`，无bytes/SHA可记录：F1 shared cache NPZ、shared receipt、F1C feature NPZ/binding、F1G feature NPZ/binding；因此没有pair metrics、aggregate或性能表。
- Smoke outer全文为空。唯一cache日志`F1_source_v_cache.out`为`1007` bytes，完整技术栈如下：

```text
Traceback (most recent call last):
  File "/home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260813_v2_549a4b3a/code/build_phase1_clic_source_v_leo_iq.py", line 1316, in <module>
    raise SystemExit(main())
  File "/home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260813_v2_549a4b3a/code/build_phase1_clic_source_v_leo_iq.py", line 1296, in main
    result = build_source_v_received_iq(build_parser().parse_args(argv))
  File "/home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260813_v2_549a4b3a/code/build_phase1_clic_source_v_leo_iq.py", line 943, in build_source_v_received_iq
    c_clean = _read_clean_validation_binding(
  File "/home/szu2070436088/2510044040/releases/phase1_clic_source_metrics_20260813_v2_549a4b3a/code/build_phase1_clic_source_v_leo_iq.py", line 635, in _read_clean_validation_binding
    raise CLICSourceVLeoCacheError(f"{arm} clean-v4 V day axis drifted")
__main__.CLICSourceVLeoCacheError: C clean-v4 V day axis drifted
```

- 生产读取器QA：release Python模块import成功；`read_source_v_cache_snapshot`可加载。实际reopen返回`CLICSourceVFeatureExportError: source-V cache or receipt is missing`；C/G feature及binding均不存在，无法进行feature reopen或finite/physical-binding QA。该失败仅表示smoke artifact不完整，不是性能结果。
- Formal封存：formal run、formal log、formal outer均`ABSENT`；无formal launcher进程，故`FORMAL_INVOCATION=0`可确认。未启动formal，不存在formal性能结论。
- 活动资源封存：ADV wrapper`801059`及六个child`801089,801092,801095,801100,801103,801106`仍活跃，GPU0..5各1个ADV训练进程，GPU6..7无进程；source-metrics/smoke进程不存在。未干预ADV。
- SSH封存：本地无ssh/scp进程、无到N607的`ESTABLISHED`连接；仅保留TCP`TIME_WAIT`记录及本机22监听，不影响远端run判断。
- 最终判定：smoke是技术失败并已自然退出；保留所有部分run/log目录，不重跑、不启动formal、不读取任何性能或target/query/truth结果。根镜像报告路径不存在，未创建额外镜像；本次Git变更仅限本owned report。
