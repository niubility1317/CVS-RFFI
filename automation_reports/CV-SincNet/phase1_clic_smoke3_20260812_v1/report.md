# Phase1 CLIC真实训练入口三批技术烟测报告

## 1. 状态与目标

- 实验ID：`phase1_clic_smoke3_20260812_v1`
- 当前状态：`SMOKE_TECHNICAL_PASS / NO_PERFORMANCE_RESULT / FORMAL_LAUNCH=0`
- 操作者：主控Codex；N607唯一runner：`Luna/max`
- 目标：在N607真实ManySig、真实GeoSat-C final checkpoint和真实训练器入口上，分别对C、G执行恰好3个optimizer batch；三个batch依次使用`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，验证数据元信息、单份received-IQ弱星地信道、CLIC前向、共同L_base、VJP、AMP、资源回执和图释放的完整闭环。
- 本烟测只判断技术健康，不读取或比较准确率、loss优劣等性能，不产生checkpoint、terminal、prediction或score，也不用于选模。

## 2. 根因与修复

- v4正式矩阵12/12在首batch同一错误`P1-CLIC source-L batch metadata is absent`退出；原因是训练器把真实`move_batch`返回的`extra=(domain, metadata)`错误当成直接metadata映射。
- 修复后通过既有`_meta_from_extra`读取第二项metadata，严格提取128行`base_index/sig_i/local_label`；仅接受整数，拒绝bool、浮点、字符串、非有限值、负base/sig和越界标签。
- 新增参数`--phase1_clic_technical_smoke_batches 3`。它复用正式训练完整batch生命周期，在三场景全部审计、AMP和图释放闭合后写技术receipt并退出；默认`0`保持正式40epoch路径不变。
- 发布核心commit：`f43f313e`（`fix: bind CLIC trainer batch metadata`）。

## 3. 冻结烟测矩阵与配置

| 行 | 折 | 臂 | operator | GPU | batch | 场景顺序 |
|---|---:|---|---|---:|---:|---|
| `F1C_CLIC12` | F1 | C | `raw_phase_control` | 0 | 3 | clear→low_elev→rain |
| `F1G_CLIC12` | F1 | G | `complex_local_invariant_curvature` | 1 | 3 | clear→low_elev→rain |

- 其余配置完全继承冻结launcher：seed=`7281164`、batch=`128`、AdamW、`clean CE+0.10×KL(clean-stopgrad→single-LEO)`、4个source-L训练TX、1个known-validation TX、1个proxy-unknown TX；held/proxy/query/target训练访问均为0。
- G只从同一份received_i提取lag=`{1,2,4,8}`的多尺度三点复曲率token；C/G除operator外配置相同。
- 唯一新增命令参数为`--phase1_clic_technical_smoke_batches 3`；run/log/output均使用本烟测不可覆盖ID。

## 4. 本地证据

- TDD RED：15组非法物理绑定中14组按预期失败；旧实现会静默截断bool/浮点/字符串或放行负值与越界标签。
- GREEN：物理绑定定向19项通过。
- `ssr-gpu`：`py_compile train_ssdg.py/test_phase1_clic.py`通过；`test_phase1_tx_partition.py + test_phase1_clic.py`共195项通过，仅既有AMP弃用warning。
- `git diff --check`通过，仅工作树LF/CRLF提示。
- 独立复审：`SPEC=PASS / QUALITY=PASS / P0=0 / P1=0`。
- 核心文件SHA256：`train_ssdg.py=33371D4504F08037BF1345A245826107615391A706DD05F04E6DD1E126B10A08`；`test_phase1_clic.py=8DDFCFC7F2C0C025B5F92C89B0A24AA80757888C8B908772B278D8693DF8B697`。

## 5. N607发布与停止合同

- 发布源必须是Git commit `f43f313e`的干净archive，不得携带未提交Task7文件。
- 预定release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_smoke3_20260812_v1_f43f313e`。
- 预定run/log：`runs/phase1_clic_smoke3_20260812_v1`、`logs/phase1_clic_smoke3_20260812_v1`；落地前必须均不存在。
- 先执行普通账户直连preflight、GPU/进程检查、archive与远端文件hash、`py_compile`、trainer help、launcher dry-run；随后仅启动上述C/G两行。
- 成功条件：两份`phase1_clic_technical_smoke_receipt.json`均`completed=true`、`batches=3`、三场景各1、VJP/AMP/resource/graph release闭合，且proxy/query/target/selection均0；无`final_ssdg.pth`、无terminal receipt。
- 失败条件：任一配置/协议/执行异常立即封存烟测为技术失败，不启动正式v5；不得在同一run ID重试。任何性能值都不得作为停止或启动依据。

## 6. 运行回填

- 运行时间：2026-08-12（N607直连短连接；普通账户；未使用admin）。
- preflight/resource：通过。8张GPU在启动前无compute训练进程；ManySig与F1 GeoSat-C final checkpoint均存在；smoke3的release/run/log/outer均为absent。
- clean archive：仅由Git commit `f43f313e`生成，未携带Task7或其他dirty/untracked文件；本地archive=`E:\type10-7\code\runner_tmp_phase1_clic_smoke3_20260812_v1\phase1_clic_smoke3_20260812_v1_f43f313e.tar`，SHA256=`4F96E4203830809BA807750F03921EE21F26D79EDEE5C401BD179D5C87B3A03F`，bytes=`266874880`，Task7成员数=0。
- SCP/landing：SCP恰1次；远端archive=`/home/szu2070436088/2510044040/CV-SincNet/phase1_clic_smoke3_20260812_v1_f43f313e.tar`，远端SHA256=`4f96e4203830809ba807750f03921ee21f26d79edee5c401bd179d5c87b3a03f`、bytes=`266874880`与本地闭合。随后原子staging解包至release=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_smoke3_20260812_v1_f43f313e`，release不可覆盖。
- 远端静态门：`py_compile`（train_ssdg.py、phase1_clic.py、model.py、model_dual_cvsincnet.py）、trainer`--help`（含`phase1_clic_technical_smoke_batches`）、launcher`bash -n`、`RUN_ID=phase1_clic_smoke3_20260812_v1 --dry-run`均通过；release内生成的`__pycache__`已清理并复核为空。archive解出LF文件SHA：`train_ssdg.py=1BFC6409E36D30BD4C230B002CD09A48557FD8071F25206714885D459FA4077C`、`phase1_clic.py=36FFDE23244D80AD15647F72252C70004D1E33549D7BF3D6E510E99E56461335`、`model.py=8F6391500D456EAD76B50D171167E931E2665B03371A001C27ABC531629AF26A`、`model_dual_cvsincnet.py=29A953D1A2B075D3B08AF9687DCD8E0BE903818E0950B9EB4C340EA221CC39F0`、`launcher=41AABBBA91CD0CA33ACC366D2B32F1531F1C37262AB9C99D5FC90B83E8BB30B1`，远端逐项匹配。
- smoke入口：未调用完整12臂launcher；从release dry-run精确取得F1C/F1G两行，仅追加`--phase1_clic_technical_smoke_batches 3`，分别绑定GPU0/GPU1，独立日志与PID记录。smoke启动调用恰1次（两子进程），正式12臂launch=`0`、retry=`NO`。首次回传阶段因PowerShell CRLF导致的`cat`路径包装错误未触发重试；两子进程均已实际完成并退出。

| 臂 | GPU | PID | exit | 日志 | 输出目录 |
|---|---:|---:|---:|---|---|
| `F1C_CLIC12`（C） | 0 | 2437209 | 0 | `logs/phase1_clic_smoke3_20260812_v1/F1C_CLIC12_smoke.out` | `runs/phase1_clic_smoke3_20260812_v1/F1C_CLIC12` |
| `F1G_CLIC12`（G） | 1 | 2437210 | 0 | `logs/phase1_clic_smoke3_20260812_v1/F1G_CLIC12_smoke.out` | `runs/phase1_clic_smoke3_20260812_v1/F1G_CLIC12` |

- F1C技术receipt=`runs/phase1_clic_smoke3_20260812_v1/F1C_CLIC12/phase1_clic_technical_smoke_receipt.json`：schema=`cvs.phase1.clic_technical_smoke.v1`、`completed=true`、`arm=C`、`batches=3`、formal scenes=`leo_clear_weak`/`leo_low_elev_weak`/`leo_rain_weak`各1、`scene_audits_complete=true`、`amp_attempts=3`、`graph_release_count=3`、`proxy_rows_loaded=0`、`query_rows_opened=0`、`target_rows_opened=0`、`selection_feedback_count=0`。
- F1G技术receipt=`runs/phase1_clic_smoke3_20260812_v1/F1G_CLIC12/phase1_clic_technical_smoke_receipt.json`：字段同上，`arm=G`；G的冻结lag仍为`{1,2,4,8}`，receipt的source/class/physical顺序哈希与C一致。
- 两份日志均含`[P1-CLIC-SMOKE] arm=... batches=3 scenes=3 proxy_rows_loaded=0 query_rows_opened=0`；未发现Traceback或error。两份config receipt均显示`source_l_only=true`、`use_held=false`、`use_proxy=false`、`use_target=false`、`query_truth_access=false`、`query_role_access=false`。
- 输出计数：`phase1_clic_technical_smoke_receipt.json=2`、`phase1_clic_config_receipt.json=2`；`final_ssdg.pth=0`、terminal receipt=`0`、prediction=`0`、score=`0`、failure receipt=`0`。
- 清理：完成后`train_ssdg`进程为空，GPU compute应用为空；本地`ssh.exe=0`，N607及bridge的TCP22连接均为0。数据、checkpoint、release、日志和receipt均保留，未删除或覆盖既有v1-v4运行。
- 最终结论：三batch真实训练入口技术闭环通过，可将本报告及上述receipt交由主控决定是否另建不可覆盖的正式v5；本烟测不提供任何性能结果，也不构成正式矩阵启动授权。
