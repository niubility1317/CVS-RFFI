# ADV3B02-MUSE-SSDG Phase1最小预登记

## 2026-08-20正式N607运行预登记

- 运行ID：`phase1_adv3b02_muse_ssdg_20260820_e5b321b`；输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_muse_ssdg_20260820_e5b321b`，禁止覆盖。
- 固定实现：`work/cvs-active`合并提交`e5b321bdf95dc8a696a12b6d64f0fbc9405da603`；该提交的两个父提交为主线`ed890015ddb9968663609727c28fdb4d749d4334`与最终复审通过的MUSE提交`e767bad0082b3564f01c1d765b543a1780aa03d6`。
- 候选与资源：单seed`392002`、200epoch；GPU0顺序运行`M0,M1`，GPU1顺序运行`M2,M3`。同一主Agent为唯一launch owner，不并行重复启动任何候选。
- 数据与边界：`ManySig.pkl`；`tx_rx_day_1_7_2`；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；训练期零目标接收机访问；checkpoint选择固定为`final_only`。
- 必需评测：每个完成训练的候选必须由同一最终checkpoint执行一次联合评测，并分别保留`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`日志与metrics。任一场景失败不得写`ARTIFACTS_COMPLETE`。
- N607只读预检：2026-08-20T09:54:06+08:00直连普通账户成功；项目根、`Dataset_WigSig/ManySig.pkl`和`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`可见；8块RTX3090均为1MiB/0%且无compute app或既有项目训练任务。
- 停止规则：仅在协议/路径/checkout/output冲突、训练或评测执行错误、缺失final checkpoint、缺失prediction/metrics闭合、OOM/NaN或相同确定性启动前异常时停止对应运行；不得因性能高低停止。
- 实际release提交：`29316416cd4fed806fe1030562c0204448f09681`；归档提交包含预登记报告，实际训练代码仍由`e5b321bdf95dc8a696a12b6d64f0fbc9405da603`固定。
- release映射：本地`E:/type10-7/local_artifacts/releases/phase1_adv3b02_muse_ssdg_20260820_e5b321b.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_muse_ssdg_20260820_e5b321b.tar.gz`→解压根`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_muse_ssdg_20260820_e5b321b`。
- 唯一release归档SHA-256：本地与远端均为`03f0585ec8184dab6a28d79ddebdbca07d021f8b4c0df6f2ee498995ee3bf505`；远端Python编译、launcher语法检查和M3 dry-run均通过，dry-run恰好产生1条训练命令与1条联合评测命令，且未创建run root。
- 实际启动命令框架：GPU0使用`--only=M0,M1`，GPU1使用`--only=M2,M3`；共同设置`ROOT=<release根>`、`RUN_ID=phase1_adv3b02_muse_ssdg_20260820_e5b321b`、`RUNS_ROOT=<正式输出根>`、`WISIG_PKL=<主项目ManySig.pkl>`和固定远端Python。

### 启动健康检查与M2定点修复

- 启动PID：GPU0队列`3680151`，GPU1队列`3680152`；两者CWD均为固定release根，启动日志均非空。M0与M2候选目录分别创建，未发生输出覆盖。
- M0健康：GPU0队列持续运行；检查时已进入E003/200，前三个epoch约86–89秒，显存约2.2GB，日志持续增长且未发现Traceback/OOM/NaN。
- M2技术失败：首batch在`train_ssdg.py`统一遥测字典读取未赋值的`domain_pass`，触发`UnboundLocalError`；候选状态为`TRAIN_FAILED`，原始训练日志和全部失败产物保留。GPU1 launcher按预登记规则退出，M3未启动。
- 根因：MUSE分支已生成`domain_mask/temporal_mask/strong_mask`，但遗漏了legacy分支已有的三个通过率标量计算；该问题不涉及数据权限、loss、路由或性能。
- 本地TDD：新增`test_muse_pseudo_gate_pass_rates_are_defined_for_first_batch_telemetry`；生产修复前以缺少统一pass-rate入口产生预期RED，随后MUSE与legacy共同调用同一计算函数，单测及MUSE聚焦回归转GREEN。
- 重启边界：不覆盖或重启仍健康的M0/M1队列；仅以新run ID`phase1_adv3b02_muse_ssdg_20260820_m23_r1`在GPU1重启缺失的`M2,M3`，保留原M2失败目录作为技术证据。
- 修复release提交：`4816dbbdc08235718174cefc2b8f13375ab1f635`；本地`E:/type10-7/local_artifacts/releases/phase1_adv3b02_muse_ssdg_20260820_m23_r1.tar.gz`→远端同名`releases`归档→解压根`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_muse_ssdg_20260820_m23_r1`。
- 修复release SHA-256：本地与远端均为`12cfd49b5a23b36db240b54a093a3c271c8da391f667a38c8e5d3f9a6e70d9ba`；远端编译、launcher语法与`--only=M2,M3`dry-run通过，恰好产生2条训练和2条严格联合评测命令，且未创建新run root。
- 修复队列启动：GPU1 launcher PID`3684782`，CWD为修复release根，命令固定`--only=M2,M3`；M2新候选目录与非空启动/训练日志已创建，原失败目录未覆盖。
- 修复后健康检查：M2越过原57秒故障窗口并完成E001/200，首epoch用时133.6秒；GPU1显存约2.35GB、利用率21%、功耗约138W，训练日志由6272字节增长到11910字节，未发现`UnboundLocalError`、Traceback、OOM或NaN。状态为`RUNNING`，不是`ARTIFACTS_COMPLETE`。
- 当前多点矩阵：原队列M0运行、M1等待；修复队列M2运行、M3等待。四点均保持同seed、同split、同epoch预算与独立候选目录；不在单seed结果前复制多seed矩阵。

## 2026-08-20六卡同seed单因素消融预登记

- 运行ID：`phase1_adv3b02_muse_ssdg_ablate6_s392002_20260820`；输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_muse_ssdg_ablate6_s392002_20260820`，禁止覆盖。
- 固定实现：`work/cvs-active`提交`f1db4fd48a315e3cad637d79710f8e7c39fdb072`；新增只能与`--only=M3`绑定的六个命名消融臂，原M0–M3队列不变。
- 固定种子与资源：六臂全部`seed=392002`、200epoch；GPU2=`NO_PRIOR`、GPU3=`NO_PROTO`、GPU4=`NO_TEMPORAL`、GPU5=`NO_SATELLITE`、GPU6=`NO_CROSSRX`、GPU7=`NO_NUISANCE`。GPU0/1保留已在运行的M0–M3队列，不重启、不改参、不覆盖。
- 因果对照：每臂仅从M3去除一类机制，其他数据划分、source角色比例`0.07/0.63/0.15/0.15`、`tx_rx_day_1_7_2`、PAIC、epoch预算、checkpoint规则与训练后评测保持一致。
- 消融定义：`NO_PRIOR`移除source-domain prior alignment；`NO_PROTO`移除classification prototype融合、prototype reliability及未标注prototype更新；`NO_TEMPORAL`移除temporal-stability reliability证据；`NO_SATELLITE`移除satellite identity student和MUSE satellite loss，但保留共同ADV3B02卫星增强；`NO_CROSSRX`移除cross-receiver loss；`NO_NUISANCE`移除nuisance loss。
- 必需评测：每臂训练成功后，仅用其`final_only`最终checkpoint自动执行一次canonical联合评测，分别保留`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`的独立log与metrics；任一场景缺失或严格checkpoint重建失败均不得写`ARTIFACTS_COMPLETE`。
- 唯一launch owner：本主Agent负责六臂一次发布与健康检查；不允许第二运行者重复启动或修改矩阵。
- 输出与日志：候选根分别为`M3_NO_PRIOR`、`M3_NO_PROTO`、`M3_NO_TEMPORAL`、`M3_NO_SATELLITE`、`M3_NO_CROSSRX`、`M3_NO_NUISANCE`；外层启动日志根为`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_muse_ssdg_ablate6_s392002_20260820`。
- 本地验证：`ssr-gpu`环境内16个指定测试文件共184项全部通过；launcher `bash -n`和`git diff --check`通过。每个命名消融dry-run均必须产生1条训练命令、1条联合评测命令和4个场景输出定位，且不创建run root。
- release映射：本地`E:/type10-7/local_artifacts/releases/phase1_adv3b02_muse_ssdg_ablate6_s392002_20260820.tar.gz`→远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_muse_ssdg_ablate6_s392002_20260820.tar.gz`→解压根`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_muse_ssdg_ablate6_s392002_20260820`。
- 唯一release归档SHA-256：本地与远端均为`edacf428589f404463167f6170cf6fc607c3bccb5f4ad168c172633e462d6e4e`；远端三个运行期Python文件编译和launcher语法检查通过。
- N607发布前检查：直连普通账户成功；项目、主数据和固定Python可见；GPU0/1分别有既有M0/M2任务，GPU2–7均为1MiB/0%且无compute app；新run root、log root、归档和release root在发布前均不存在。
- 六臂远端dry-run：GPU2–7对应六个预登记arm均通过；每臂恰好1条训练命令、1条联合评测命令、4个场景输出定位，candidate ID与精确消融参数一致；六次dry-run后run root仍不存在。
- 首次启动结果：GPU2、4–7的五臂已进入训练；GPU3的`NO_PROTO`在prototype bank初始化时因旧验证器拒绝`unlabeled_weight=0`而确定性`TRAIN_FAILED`，原run ID下的config、status和train log完整保留，不覆盖重启。
- GPU3定点修复：提交`24ab0a3cd4493a55407ddaf9b89c1ce10f9bfc27`把0定义为未标注prototype更新的显式禁用哨兵，并保证prototype、class count和domain count均不变化；正常启用仍只接受0.05–0.10。本地新增RED→GREEN测试后，16个指定文件共185项全部通过，相关Python编译、launcher语法和diff检查通过。
- 修复重启边界：只在空闲GPU3以新run ID`phase1_adv3b02_muse_ssdg_noproto_r1_s392002_20260820`重启`NO_PROTO`；GPU0/1既有队列及GPU2、4–7健康消融均不改变。修复run同样固定seed 392002、200epoch及训练后clean与三种LEO弱信道自动评测。
- GPU3修复release：本地`E:/type10-7/local_artifacts/releases/phase1_adv3b02_muse_ssdg_noproto_r1_s392002_20260820.tar.gz`→远端同名`releases`归档→解压根`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_muse_ssdg_noproto_r1_s392002_20260820`；唯一归档SHA-256本地与远端均为`6fe8f5cc49b4955458576335bdb5e22c45f229eca1c51835f7a457ae66f6d306`。远端编译、launcher语法及1 train/1 eval/4 outputs dry-run通过且无run-root副作用。
- 正式启动PID：GPU2 `NO_PRIOR`=`3699905`；GPU3修复`NO_PROTO`=`3704934`；GPU4 `NO_TEMPORAL`=`3699907`；GPU5 `NO_SATELLITE`=`3699908`；GPU6 `NO_CROSSRX`=`3699909`；GPU7 `NO_NUISANCE`=`3699910`。GPU0/1原队列PID`3680151`和`3684782`仍存活。
- 启动健康证据：六个launcher及各自训练子进程均存活；子进程cmdline逐一绑定对应release、run root和`M3_<ARM>`目录，`CUDA_VISIBLE_DEVICES`分别严格为2–7。实际子进程CWD均为`/home/szu2070436088`，但训练脚本、`PYTHONPATH`、release与输出均使用已核对的绝对路径，不存在checkout歧义。
- 运行状态：GPU2、4–7各已有4个epoch标记，GPU3修复臂已越过原prototype构造失败并进入GPU训练；GPU0–7均有非零训练占用。六臂训练日志当前均无Traceback、RuntimeError、OOM、UnboundLocalError或ValueError，状态为`RUNNING`，尚不是`ARTIFACTS_COMPLETE`。
- 停止规则：仅因协议/路径/checkout/output冲突、训练或评测执行错误、缺失final checkpoint、缺失prediction/metrics闭合、OOM/NaN或相同确定性启动前异常停止对应运行；不得因中间或最终性能高低停止。

## 2026-08-20T11:21:49+08:00进度快照

本次以只读方式完整解析4个相关run root中的全部候选`train.log`、launcher log、status、checkpoint及metrics文件，并核对活动PID和GPU占用；没有启动、停止、重启或修改远端任务。

| GPU | 当前候选 | 最新完成epoch | 进度 | 根训练PID | 状态 |
|---:|---|---:|---:|---:|---|
| 0 | `M0` | 30/200 | 15.0% | 3680160 | `RUNNING` |
| 1 | `M2`修复run | 26/200 | 13.0% | 3684788 | `RUNNING` |
| 2 | `M3_NO_PRIOR` | 18/200 | 9.0% | 3699926 | `RUNNING` |
| 3 | `M3_NO_PROTO`修复run | 16/200 | 8.0% | 3704937 | `RUNNING` |
| 4 | `M3_NO_TEMPORAL` | 18/200 | 9.0% | 3699923 | `RUNNING` |
| 5 | `M3_NO_SATELLITE` | 18/200 | 9.0% | 3699929 | `RUNNING` |
| 6 | `M3_NO_CROSSRX` | 18/200 | 9.0% | 3699925 | `RUNNING` |
| 7 | `M3_NO_NUISANCE` | 18/200 | 9.0% | 3699927 | `RUNNING` |

- GPU0队列中的`M1`与GPU1修复队列中的完整`M3`尚未创建候选目录，确认是等待前序`M0`与`M2`训练完成后由现有launcher自动启动，不是丢失任务。
- 8张GPU当前显存占用约2.39–2.94GB，利用率约19%–38%；全部launcher和训练进程存活，日志mtime持续更新。
- 活动训练日志完整扫描中，Traceback、RuntimeError、OOM、UnboundLocalError、ValueError、AssertionError和Killed均为0。训练内出现的`nan% (0/0)`、`sat_cos=nan`等是target/final评测与未激活统计项的占位值；对应active flag与分母为0，未伴随loss NaN、异常退出或停滞，当前不判为数值故障。
- 当前8个活动候选均尚未生成`final_ssdg.pth`、`metrics_clean.json`或三种LEO弱信道metrics；因此clean与`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`自动评测尚未开始。状态不能标记为`ARTIFACTS_COMPLETE`。
- 原始M2的`UnboundLocalError`失败目录与原始`NO_PROTO`的权重验证失败目录继续保留为技术证据；当前表格只列修复后健康run，不把旧失败计入活动候选。

## 2026-08-20T15:55:55+08:00进度快照

本次再次只读完整解析4个run root的所有训练日志与artifact，并实时核对8张GPU、launcher及排队目录；没有修改远端状态。

| GPU | 当前候选 | 最新完成epoch | 进度 | 异常指纹 | 当前阶段 |
|---:|---|---:|---:|---:|---|
| 0 | `M0` | 62/200 | 31.0% | 0 | 训练 |
| 1 | `M2`修复run | 57/200 | 28.5% | 0 | 训练 |
| 2 | `M3_NO_PRIOR` | 48/200 | 24.0% | 0 | 训练 |
| 3 | `M3_NO_PROTO`修复run | 48/200 | 24.0% | 0 | 训练 |
| 4 | `M3_NO_TEMPORAL` | 59/200 | 29.5% | 0 | 训练 |
| 5 | `M3_NO_SATELLITE` | 59/200 | 29.5% | 0 | 训练 |
| 6 | `M3_NO_CROSSRX` | 59/200 | 29.5% | 0 | 训练 |
| 7 | `M3_NO_NUISANCE` | 59/200 | 29.5% | 0 | 训练 |

- 8个launcher均存活；GPU显存约2.83–6.14GB，实时利用率21%–99%，无空卡或失联任务。
- 活动日志完整扫描中，Traceback、RuntimeError、OOM、UnboundLocalError、ValueError、AssertionError和Killed均为0。
- GPU0队列的`M1`和GPU1队列的完整`M3`仍等待当前`M0`、`M2`完成后自动启动，候选目录尚未创建。
- 所有活动候选仍未生成`final_ssdg.pth`、clean metrics或三种LEO metrics，自动评测尚未开始；当前状态仍为`RUNNING`。

## 2026-08-20T16:03:24+08:00 ETA与ADV3B02耗时归因

本次完整读取历史`ADV3B02_CORE90_SOFT_E200/metrics_epoch.csv`和`metrics_epoch.jsonl`共200个epoch记录，并以各活动训练最近5个已完成epoch重新估计。ETA仅用于资源规划；MUSE阶段切换、同卡第二任务结束或启动、最终评测耗时都会改变实际完成时间。

| GPU | 当前候选 | 最新完成epoch | 最近5 epoch均值 | 预计剩余训练 |
|---:|---|---:|---:|---:|
| 0 | `M0` | 63/200 | 487.5秒 | 约18.6小时 |
| 1 | `M2`修复run | 58/200 | 486.7秒 | 约19.2小时 |
| 2 | `M3_NO_PRIOR` | 49/200 | 529.7秒 | 约22.2小时 |
| 3 | `M3_NO_PROTO`修复run | 49/200 | 518.3秒 | 约21.7小时 |
| 4 | `M3_NO_TEMPORAL` | 60/200 | 339.2秒 | 约13.2小时 |
| 5 | `M3_NO_SATELLITE` | 60/200 | 331.0秒 | 约12.9小时 |
| 6 | `M3_NO_CROSSRX` | 60/200 | 338.3秒 | 约13.2小时 |
| 7 | `M3_NO_NUISANCE` | 61/200 | 337.6秒 | 约13.0小时 |

- 历史ADV3B02的200个`epoch_time_s`总和为3.98小时，均值71.6秒、中位数60.2秒；前30个epoch均值46.7秒。当前各活动候选与历史相同epoch前缀比较，单epoch约慢6.0–10.2倍。
- 首要原因是epoch语义改变。历史ADV3B02为`L/U/V=0.10/0.70/0.20`，legacy训练循环每epoch由`len(L_s loader)`决定；当前MUSE协议为`0.07/0.63/0.30`，所有M0–M3均固定由`len(U_s loader)`决定，M0也循环L_s直至走满U_s长度。按全池比例近似，单epoch optimizer step预算从约10%扩大到63%，即约6.3倍；这是M0也明显变慢的主因。
- 当前source校准/选模池由历史20%增至30%，评估数据量约增加50%；训练继续启用`concat_masked`，每个step包含clean与卫星视图两次前向。M1–M3还增加domain/GRL/self/nuisance、融合与H/M/L路由；M3进一步包含temporal、classification prototype、satellite identity和cross-receiver路径。MUSE的S1/S2A/S2B/S3A/S3B/S3C阶段还会改变后半程单epoch成本。
- GPU0、1、4–7各只有本矩阵一个compute app；GPU2同时运行`M3_NO_PRIOR`和独立NTRS候选，GPU3同时运行`M3_NO_PROTO`和独立NTRS候选，因此GPU2/3显存约5.6–6.1GB、利用率约98%–99%，其ETA长于GPU4–7。该并发符合每卡最多两个训练的既定资源规则，本次监控未干预任何任务。
- 服务器有96个逻辑CPU，检查时load average约13、可用内存442GiB，不是整机CPU或内存耗尽；GPU0/1/4–7利用率多为21%–38%，说明主要是单任务的数据生成、星地增强、多视图前向和DataLoader供给间歇，而非纯GPU算力饱和。
- GPU4–7单臂预计再需约13–18小时训练；GPU2/3若第二任务持续并发，预计约22–30小时，若NTRS先结束则可能缩短。每个训练结束后还需执行clean与三种LEO弱信道评测，暂按每候选额外0.5–2小时保守预留。
- GPU0的`M1`要等待M0训练和四场景评测完成后才启动；GPU1的完整`M3`同理等待M2。按当前速度外推，包含这两个排队候选及全部最终评测的整个矩阵预计还需约42–60小时，即大致在2026-08-22上午至2026-08-23凌晨闭合。该时间窗不是完成承诺，下一次阶段切换后应使用新近epoch重新估计。

## 候选矩阵

| 候选 | 固定基座 | 能力 | seed | epoch | source角色比例 | checkpoint选择 |
|---|---|---|---:|---:|---|---|
| M0 | `ADV3B02_CORE90_SOFT_E200` | 同协议ADV3B02控制；不进入MUSE能力路径 | 392002 | 200 | `0.07/0.63/0.15/0.15` | `final_only` |
| M1 | 同M0 | 基础domain/GRL/self/nuisance | 392002 | 200 | 同M0 | `final_only` |
| M2 | 同M0 | M1+fusion+H/M/L路由 | 392002 | 200 | 同M0 | `final_only` |
| M3 | 同M0 | M2+satellite student+cross-receiver+classification prototype | 392002 | 200 | 同M0 | `final_only` |

四个候选固定同一`tx_rx_day_1_7_2`数据split及`L_s/U_s/V_cal/V_select`角色定义，均以`len(U_s loader)`作为每epoch optimizer step预算。M0只按该长度循环L_s，不读取U_s batch、不计算U_s损失、不创建MUSE state。四臂共同启用ADV3B02 PAIC guard：`enabled=true`、`sat_ce_delta=0.12`、`grad_delta=3.0`、`reliable_drop=0.01`、`cooldown_epochs=1`、`sat_scale=0.75`。

## Commit

- Tasks 1–7代码HEAD：`4c66489ea058f5fe8401c29a237a58708bd7451f`，固定本报告审计的3个生产文件实现。
- Task 8修复前文档提交：`66ba28c48f5961100483cf6794252e15ca9bfb3b`（`docs: close MUSE SSDG implementation evidence`）。
- Task 8 fix round 1文档提交：本文件所在修复提交；提交后的精确OID、push状态和远端分支读回记录在Git忽略的`.superpowers/sdd/2026-08-19-adv3b02-muse-ssdg/task-8-report.md`“Fix round 1”节。文档提交不改变上述代码HEAD。

## 命令

```bash
bash code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh --only=M0,M1,M2,M3
```

本次Task 7 fix round 1仅执行本地`bash -n`、pytest、`--dry-run --only=M3`及临时fake trainer/evaluator非dry-run控制流，不连接N607、不启动真实训练。

## 环境与CWD

- 计划环境：N607普通账户，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 计划CWD：`/home/szu2070436088/2510044040/CV-SincNet`。
- 本地验证环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`。
- 本地验证CWD：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-muse-ssdg`。

## 输入、输出与GPU

- 输入：`${ROOT}/Dataset_WigSig/ManySig.pkl`。
- 输出根：`${ROOT}/runs/phase1_adv3b02_muse_ssdg_20260819/{M0,M1,M2,M3}`；已存在的候选根禁止覆盖。
- GPU：默认`GPU=0`，所有子命令映射为`CUDA_VISIBLE_DEVICES=${GPU}`与进程内`cuda:0`。

## 停止规则

- 训练命令非零退出、`final_ssdg.pth`缺失或为空时停止当前候选并保留全部产物。
- 联合clean+三LEO评测命令失败或联合artifact为空时写`EVAL_FAILED_JOINT`并保留训练产物；拆分时任一场景缺失、计数非法、准确率与计数不一致或对应日志/metrics为空时停止，写`EVAL_FAILED_<SCENARIO>`。
- 不因中间或最终性能高低停止。

## 预期artifact

每个候选根必须包含非空`train.log`、`config.json`、`final_ssdg.pth`、`eval_clean.log`、`eval_leo_clear_weak.log`、`eval_leo_low_elev_weak.log`、`eval_leo_rain_weak.log`、`metrics_clean.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`、`metrics_leo_rain_weak.json`。真实评测器只调用一次并生成联合JSON；控制层从真实row计数重算四个场景aggregate，四份JSON顶层`scenario`、`aggregate.scenario`和row语义必须一致。仅当四组评测日志与metrics均非空时，`status.txt`才写`ARTIFACTS_COMPLETE`。

## Task 8：追踪闭合与发布准备

### 状态

Tasks 1–7实现已完成Task 8本地聚焦验证和正反追踪。当前结论是`LOCAL_IMPLEMENTATION_VERIFIED_WITH_RUNTIME_EVIDENCE_PENDING`，不是`ARTIFACTS_COMPLETE`或`ANALYZED`：MUSE-002实际loader receipt、MUSE-014真实M0–M3矩阵和MUSE-018训练外precision诊断入口尚未闭合；没有clean或三种LEO弱场景的性能结果。

### 完整聚焦测试

- 环境：`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`；解释器前缀读回为`C:/Users/lh594/.conda/envs/ssr-gpu`。
- CWD：`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-muse-ssdg`。
- 文件映射：brief列出的12个测试文件均实际存在，无需使用等价文件替换。
- 命令：`python -m pytest code/tests/test_muse_ssdg_schedule.py code/tests/test_muse_ssdg_routing.py code/tests/test_muse_ssdg_losses.py code/tests/test_muse_ssdg_memory.py code/tests/test_muse_ssdg_training_heads.py code/tests/test_muse_ssdg_train_integration.py code/tests/test_muse_ssdg_satellite.py code/tests/test_muse_ssdg_checkpoint.py code/tests/test_phase1_muse_launcher.py code/tests/test_meta_ssl_pseudo_gate.py code/tests/test_concat_sat_channel_aug.py code/tests/test_phase1_p1_protocol.py -q`。
- 结果：退出码0；12个文件共收集107项，107项全部通过；测试进程未输出warning，也没有warning升级为error。

### 真实checkpoint无query smoke

- checkpoint：`E:/type10-7/automation_reports/CV-SincNet/qknnv42_strict_dual125_20260714_183556/artifacts/best_joint_safe_ssdg.pth`，8,582,116字节。
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 路径发现方式：先在仓库报告中定向检索已登记的ADV3B02路径，再仅对3个精确候选路径执行只读`test -f`；未扫描数据根、未连接N607。
- 输入边界：CPU、1个batch、batch size 2；输入为确定性构造的source-shaped张量；`dataset_read_count=0`、`support_input_count=0`、`query_input_count=0`、`target_truth_read_count=0`。
- 执行结果：严格重建0 missing、0 unexpected；前向输出`tx_logits=[2,6]`、`z_id=[2,160]`且有限；反向、optimizer step、MUSE训练态保存和重新加载全部完成；退出码0。
- 输出artifact：`E:/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820/m3_real_checkpoint_no_query_smoke.pt`，274,118字节；第二个独立进程以`weights_only=True`回读schema、batch数、checkpoint严格加载标志和零query/truth计数，全部一致。
- 非失败关注：模型前向输出1条`torch.cuda.amp.autocast`弃用`FutureWarning`。该warning来自既有`code/model.py`调用，不影响本次退出码和数值有限性；完整聚焦pytest没有输出该warning。

实际承载方式：命令由`C:/Program Files/Git/bin/bash.exe`在上述CWD中执行，通过quoted here-doc把Python源码直接送入`ssr-gpu`解释器stdin；当时没有创建临时`.py`脚本。以下为实际运行的主smoke命令，未重构为新的入口：

```bash
mkdir -p /e/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820
PYTHONPATH=code PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /c/Users/lh594/.conda/envs/ssr-gpu/python.exe - <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from post_stage_common import load_checkpoint
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from SSDG import train_ssdg

checkpoint_path = Path(r"E:/type10-7/automation_reports/CV-SincNet/qknnv42_strict_dual125_20260714_183556/artifacts/best_joint_safe_ssdg.pth")
artifact_path = Path(r"E:/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820/m3_real_checkpoint_no_query_smoke.pt")
device = torch.device("cpu")

checkpoint = load_checkpoint(str(checkpoint_path), device)
input_len = int((checkpoint.get("args") or {}).get("wisig_out_len", 256))
model, load_audit = build_exact_ssdg_model_from_checkpoint(
    checkpoint,
    input_len=input_len,
    device=device,
)
model.train()

args = train_ssdg.build_arg_parser().parse_args([
    "--output_dir", str(artifact_path.parent / "unused_output"),
    "--use_muse_ssdg", "true",
    "--muse_level", "M3",
    "--epochs", "200",
    "--checkpoint_selection", "final_only",
])
muse_state = train_ssdg._initialize_muse_training_state(args, model, device)
assert muse_state is not None and muse_state["level"] == "M3"
muse_state["schedule_state"] = train_ssdg.muse_schedule_for_epoch(69, muse_state["config"])

batch_size = 2
x = torch.linspace(-1.0, 1.0, steps=batch_size * 2 * input_len, device=device).reshape(batch_size, 2, input_len)
synthetic_source_labels = torch.tensor([0, 1], device=device, dtype=torch.long)
source_domains = torch.tensor([0, 1], device=device, dtype=torch.long)

optimizer = torch.optim.SGD(train_ssdg._optimizer_parameters(model, muse_state), lr=1e-5)
optimizer.zero_grad(set_to_none=True)
output = model(x, return_aux=True)
logits = output["tx_logits"]
z_id = output["z_id"]
z_dom = output["z_dom"]
heads = muse_state["heads"]
local_prob = heads.local_prob(z_id, source_domains)
loss = (
    F.cross_entropy(logits, synthetic_source_labels)
    - 0.05 * local_prob.clamp_min(1e-8).log().mean()
    + 0.05 * heads.self_supervised_loss(z_id, z_id * 0.99)
    + 0.05 * heads.nuisance_loss(
        z_dom,
        torch.zeros(batch_size, int(args.muse_nuisance_dim), device=device),
        torch.ones(batch_size, dtype=torch.bool, device=device),
    )
)
assert torch.isfinite(loss)
tracked = next(parameter for parameter in model.parameters() if parameter.requires_grad)
before = tracked.detach().clone()
loss.backward()
assert tracked.grad is not None and torch.isfinite(tracked.grad).all()
optimizer.step()
parameter_delta = float((tracked.detach() - before).abs().max().item())

state_payload = train_ssdg._muse_checkpoint_state(muse_state)
artifact = {
    "artifact_schema": "adv3b02_muse_ssdg_m3_no_query_smoke_v1",
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "checkpoint_load_audit": load_audit,
    "muse_level": "M3",
    "batch_count": 1,
    "batch_size": batch_size,
    "input_origin": "deterministic_source_shaped_tensor_no_dataset_read",
    "dataset_read_count": 0,
    "support_input_count": 0,
    "query_input_count": 0,
    "target_truth_read_count": 0,
    "forward_finite": bool(torch.isfinite(logits).all() and torch.isfinite(z_id).all() and torch.isfinite(z_dom).all()),
    "backward_finite": True,
    "optimizer_step_complete": True,
    "parameter_delta_max": parameter_delta,
    "tx_logits_shape": list(logits.shape),
    "z_id_shape": list(z_id.shape),
    "z_dom_shape": list(z_dom.shape),
    "loss": float(loss.detach().item()),
    **state_payload,
}
torch.save(artifact, artifact_path)

restored_artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
restored_state = train_ssdg._initialize_muse_training_state(args, model, device)
train_ssdg._restore_muse_checkpoint_state(restored_state, restored_artifact)
for key, value in muse_state["heads"].training_state_dict().items():
    assert torch.equal(restored_state["heads"].training_state_dict()[key], value)
assert restored_state["schedule_state"] == muse_state["schedule_state"]
assert restored_artifact["batch_count"] == 1
assert restored_artifact["query_input_count"] == 0
assert restored_artifact["target_truth_read_count"] == 0
assert restored_artifact["forward_finite"]
assert restored_artifact["optimizer_step_complete"]

summary = {
    "status": "VERIFIED",
    "artifact": str(artifact_path),
    "artifact_bytes": artifact_path.stat().st_size,
    "checkpoint": str(checkpoint_path),
    "checkpoint_sha256": artifact["checkpoint_sha256"],
    "checkpoint_load_strict": load_audit["checkpoint_load_strict"],
    "missing_keys": load_audit["missing_keys"],
    "unexpected_keys": load_audit["unexpected_keys"],
    "batch_count": 1,
    "query_input_count": 0,
    "target_truth_read_count": 0,
    "forward_shape": list(logits.shape),
    "z_id_shape": list(z_id.shape),
    "loss_finite": True,
    "optimizer_step_complete": True,
    "muse_state_roundtrip": True,
}
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY
```

主smoke已记录输出：

```text
E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-muse-ssdg\code\model.py:695: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
{"artifact": "E:\\type10-7\\local_artifacts\\adv3b02_muse_ssdg_task8_20260820\\m3_real_checkpoint_no_query_smoke.pt", "artifact_bytes": 274118, "batch_count": 1, "checkpoint": "E:\\type10-7\\automation_reports\\CV-SincNet\\qknnv42_strict_dual125_20260714_183556\\artifacts\\best_joint_safe_ssdg.pth", "checkpoint_load_strict": true, "checkpoint_sha256": "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98", "forward_shape": [2, 6], "loss_finite": true, "missing_keys": 0, "muse_state_roundtrip": true, "optimizer_step_complete": true, "query_input_count": 0, "status": "VERIFIED", "target_truth_read_count": 0, "unexpected_keys": 0, "z_id_shape": [2, 160]}
```

同一次原始Git Bash调用随后执行以下独立readback命令；它重新打开落盘artifact，不复用主进程内存：

```bash
PYTHONPATH=code PYTHONUTF8=1 PYTHONIOENCODING=utf-8 /c/Users/lh594/.conda/envs/ssr-gpu/python.exe - <<'PY'
from pathlib import Path
import torch
p=Path(r"E:/type10-7/local_artifacts/adv3b02_muse_ssdg_task8_20260820/m3_real_checkpoint_no_query_smoke.pt")
a=torch.load(p,map_location="cpu",weights_only=True)
assert a["artifact_schema"] == "adv3b02_muse_ssdg_m3_no_query_smoke_v1"
assert a["batch_count"] == 1 and a["query_input_count"] == 0 and a["target_truth_read_count"] == 0
assert a["checkpoint_load_audit"]["checkpoint_load_strict"] is True
print(f"ARTIFACT_READBACK_OK path={p} bytes={p.stat().st_size} schema={a['artifact_schema']}")
PY
```

readback已记录输出：

```text
ARTIFACT_READBACK_OK path=E:\type10-7\local_artifacts\adv3b02_muse_ssdg_task8_20260820\m3_real_checkpoint_no_query_smoke.pt bytes=274118 schema=adv3b02_muse_ssdg_m3_no_query_smoke_v1
```

该smoke只验证真实历史ADV3B02 checkpoint与M3训练路径、optimizer和MUSE state回环兼容，不使用真实source batch，也不产生准确率、DG收益、LEO鲁棒性或晋级证据。

### 18项正向追踪与反向审计

- 逐项状态：MUSE-001、003至013及017为`verified`；MUSE-002、014、015、016、018为`implemented`；`pending=0`。
- MUSE-002没有实际loader receipt证明四角色物理ID互斥、source/target receiver不相交和零target进入；synthetic smoke不提供该证据。
- MUSE-014只完成launcher dry-run、fake控制流和step预算测试；真实M0–M3单seed矩阵未运行，须以四臂真实run artifact升级状态。
- MUSE-018训练内precision遥测保持`N/A`，避免训练读取`U_s`真值；当前没有独立训练外precision诊断入口，因此该子要求与真实telemetry/泄漏探针结果均未闭合。
- 汇总：总要求18，`verified=13`、`implemented=5`、`pending=0`；实现映射18/18，但运行期证据未闭合。
- 生产文件审计范围：`0e1019beb8f9c3217b4ae84f1a56a4be6dd5ba9e..4c66489ea058f5fe8401c29a237a58708bd7451f`。
- 反向结果：`code/cvsrffi/muse_ssdg.py`映射MUSE-003至012、017；`code/SSDG/train_ssdg.py`映射MUSE-001至013、017、018；launcher映射MUSE-014至016。3/3个新增或修改生产文件均有规范来源，未发现需要删除或重新审批的规范外生产逻辑。
- 完整逐项证据见`analysis/adv3b02_muse_ssdg_traceability_20260819.md`。

### 单一release归档准备

拟定归档名：`adv3b02_muse_ssdg_code_4c66489ea058.tar.gz`。归档仅包含Tasks 1–7在代码身份`4c66489ea058f5fe8401c29a237a58708bd7451f`下的3个生产文件：

1. `code/cvsrffi/muse_ssdg.py`
2. `code/SSDG/train_ssdg.py`
3. `code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`

Task 8不创建归档、不执行SSH/SCP、不启动实验。后续唯一runner在N607 preflight通过后创建这一个归档，只做一次本地/远端归档SHA比较，并在远端执行两个Python文件的`py_compile`和launcher的`bash -n`；不增加成员SHA、seal、receipt或额外发布gate。

### Task 8关注与下一状态

- 最高风险剩余项是尚无真实M0–M3单seed训练及其clean/三LEO逐场景结果；同时缺少MUSE-002实际loader receipt和MUSE-018独立训练外precision诊断入口。不能据当前本地证据判断性能、晋级或发表价值。
- MUSE-002须在实际loader receipt读回四角色物理ID互斥、source/target receiver不相交和target计数0后升级；MUSE-014须在真实四臂矩阵落盘后升级；MUSE-015、016、018须在真实训练、评测与训练外诊断闭合后升级。
- release归档创建、N607资源/路径preflight、单次SHA比对、远端编译、启动后PID/CWD/cmdline/GPU/log增长检查均留给后续唯一runner；本Task未越权执行。

## Final fix wave（FFR-1至FFR-7）

### 修复结论

- FFR-1：M2/M3第三路融合证据已改为`MUSEClassificationPrototypeBank`基于`z_id`产生的真实classification prototype概率；缺失类概率为0，概率有限且归一化。`L_s`标签与domain计数产生的全局/source-domain prior已在真实融合主链routing前执行alignment；不读取`U_s` TX truth。
- FFR-2：`proto_momentum`已控制有标签和稳定高置信未标注classification prototype的EMA更新，并与0.05–0.10未标注贡献分离。epoch 181进入S3C后，temporal memory、classification prototype、threshold/statistics、`L_s` prior和local teacher均冻结；后续`train()`与optimizer step不能改变local teacher。
- FFR-3：M3已按稳定SHA mask逐样本从strong或satellite/nuisance输出中唯一选择identity logits与`z_id`，H/M/L及相关identity consistency只消费所选分支；M1/M2强制只使用strong identity。
- FFR-4：MUSE训练入口在`--muse_external_final_eval true`时返回`DELEGATED_TO_MUSE_LAUNCHER`，不执行内部target held-out评测，也不生成`frozen_phase1_heldout_eval.json`；launcher保持唯一一次canonical joint target eval。非MUSE默认内部评测行为不变。
- FFR-5：formal evaluator新增strict checkpoint reconstruction模式，使用`strict=True`、禁止direct-builder fallback，并在任何missing、unexpected、shape mismatch或重建异常时于metrics写入前非零退出。launcher强制strict模式并验证`reconstruction_audit`，不合格时写`EVAL_FAILED_JOINT`而非`ARTIFACTS_COMPLETE`。未请求strict时保留旧fallback行为。
- FFR-6：`full_ablation_spec.py`和`phase1_ablation_factory.py`的活动生产配置已从parser非法的`source_validation_only`迁移为`final_only`；共享入口解析与formal final checkpoint角色测试已闭合。
- FFR-7：traceability与本报告已引用真实调用链测试；完整RED/GREEN、文件、commit及push/OID记录写入`.superpowers/sdd/2026-08-19-adv3b02-muse-ssdg/final-fix-report.md`。

### 真实调用链证据

- prototype与prior：`test_classification_prototype_probabilities_are_normalized_with_explicit_missing_classes`、`test_m2_fusion_uses_global_local_prototype_and_l_s_prior_alignment`。
- schedule与S3C冻结：`test_proto_momentum_boundary_is_095_then_099_at_s3b_and_s3c`、`test_prototype_momentum_and_unlabeled_contribution_are_distinct_controls`、`test_epoch_181_freezes_muse_statistics_prior_and_local_teacher_state`、`test_s3c_checkpoint_round_trip_restores_frozen_local_teacher_and_prior_state`。
- identity选择：`test_m3_sha_mask_selects_exactly_one_identity_student_per_row`、`test_m1_m2_never_enable_satellite_identity_student`。
- 唯一评测与strict恢复：`test_muse_can_delegate_final_target_eval_without_changing_legacy`、`test_fake_joint_evaluator_runs_once_and_writes_four_semantic_metrics_before_complete`、`test_strict_reconstruction_failure_exits_before_metrics_are_written`、`test_launcher_rejects_non_strict_or_fallback_reconstruction_metadata`。
- factory迁移：`test_active_phase1_row_factories_emit_parser_valid_final_only_selection`、`test_active_ablation_configs_pass_shared_checkpoint_parser`。

### 发布与证据边界

- final fix实现提交：`3f3809b1527c840a72f6ff75edd92c74cd87e085`（`fix: close MUSE SSDG final review findings`）；该提交已由post-commit hook推送并读回同OID，最终证据提交仍将在本轮结束时再次独立核对远端分支OID。
- 验证范围：聚焦RED/GREEN、完整MUSE/launcher/evaluator/protocol/factory pytest、changed Python `py_compile`、launcher `bash -n`、M3 dry-run、真实ADV3B02 one-batch no-query smoke和`git diff --check`。不连接N607，不执行target评测。
- final fix合并测试：16个文件、175项全部通过；退出码0。
- final fix真实checkpoint smoke：`E:/type10-7/local_artifacts/adv3b02_muse_ssdg_final_fix_20260820/m3_true_prototype_identity_strict_state_no_query_smoke.pt`，279,773字节；严格加载0 missing/0 unexpected；真实三头为global/local/prototype，prior alignment最大变化`0.06568282842636108`；稳定SHA mask选择2条strong与2条satellite；S3C strict状态回环和独立artifact回读通过；query、target truth与target eval计数均为0。
- 真实M0–M3训练：未运行。
- 真实clean及三LEO场景性能：未产生。
- 当前状态仍是本地实现与验证闭合，不是`ARTIFACTS_COMPLETE`、`ANALYZED`或性能晋级证据。

### Final fix后续单一release清单

旧Task 8的3成员归档计划已由本final fix覆盖。后续若由唯一runner执行N607发布，归档代码身份必须为`3f3809b1527c840a72f6ff75edd92c74cd87e085`，并包含`code/cvsrffi/muse_ssdg.py`、`code/SSDG/train_ssdg.py`、`code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`和`code/scripts/eval_ssdg_sat_per_rx.py`共4个运行文件；strict evaluator不能遗漏。两个活动factory迁移文件不被MUSE运行链消费，不加入该归档。不在本final fix wave创建归档或连接N607。

## 2026-08-21T11:14:58+08:00进度与失败快照

### 结论

- 8张GPU的首波同seed实验中，7条候选已完成E200训练并写出`final_ssdg.pth`，GPU2上的`M3_NO_PRIOR`仍在E196/200；GPU0、1、3–7已空闲，GPU2仍有约3.4GB显存占用且进程、CWD和run root归属匹配。
- 已完成训练的7条候选均在E200之后、外部final eval之前发生同一技术失败：`SSDG Phase2 prototype export failed: endpoint_accept_v1 zero-direction fraction exceeds per-class limit: class=3 fraction=1.000000000 limit=0.001000000`。终态为`FAILED_EXPORT`、launcher状态为`TRAIN_FAILED`，训练checkpoint与完整日志均已保留。
- 因launcher在导出失败后立即退出，7条候选都没有生成`metrics_clean.json`、`metrics_leo_clear_weak.json`、`metrics_leo_low_elev_weak.json`和`metrics_leo_rain_weak.json`，也没有对应评测日志。因此当前没有可报告的held-out clean或三种LEO弱信道最终准确率，任何训练期source-val数据都不能替代最终测试。
- GPU0排队的`M1`和GPU1排队的完整`M3`均未创建输出目录，状态为`NOT_STARTED`；它们被前一候选的非零退出阻断。旧首轮失败的M2和NO_PROTO仍原样保留，不计入下表有效首波结果。

### 训练期详细数据

共同设置为seed=`392002`、E200、`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。下表“源验证”与“源验证卫星”均是训练期source validation诊断，不是held-out target clean/LEO最终测试。

| GPU | 候选 | 状态 | 最新epoch | 末轮训练TX | 末轮源验证TX | 历史最高源验证TX | 末轮源验证卫星均值/底线 | 最近10轮均时 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | M0 | `FAILED_EXPORT`，checkpoint已保留 | 200/200 | 0.00% | 33.33% | 99.27% | 33.33%/33.33% | 223.8秒 |
| 1 | M2（修复轮） | `FAILED_EXPORT`，checkpoint已保留 | 200/200 | 48.70% | 93.66% | 93.66% | 40.32%/39.77% | 450.3秒 |
| 2 | M3_NO_PRIOR | `RUNNING` | 196/200 | 51.67% | 93.90% | 93.90% | 39.85%/39.36% | 343.9秒 |
| 3 | M3_NO_PROTO（修复轮） | `FAILED_EXPORT`，checkpoint已保留 | 200/200 | 42.14% | 88.39% | 88.39% | 45.07%/44.71% | 339.3秒 |
| 4 | M3_NO_TEMPORAL | `FAILED_EXPORT`，checkpoint已保留 | 200/200 | 44.39% | 92.25% | 92.34% | 35.92%/35.44% | 361.2秒 |
| 5 | M3_NO_SATELLITE | `FAILED_EXPORT`，checkpoint已保留 | 200/200 | 45.91% | 93.39% | 93.39% | 40.76%/40.35% | 445.7秒 |
| 6 | M3_NO_CROSSRX | `FAILED_EXPORT`，checkpoint已保留 | 200/200 | 43.06% | 91.55% | 91.55% | 38.11%/37.82% | 352.2秒 |
| 7 | M3_NO_NUISANCE | `FAILED_EXPORT`，checkpoint已保留 | 200/200 | 28.69% | 81.33% | 81.33% | 41.48%/41.17% | 446.8秒 |

M0末轮从历史最高99.27%降至33.33%，属于明显末期退化；但在held-out clean和三LEO测试缺失前，不能据此完成方法排名。其余已完成候选的末轮源验证TX为81.33%–93.66%，末轮source-val卫星均值为35.92%–45.07%；这些仅用于定位训练行为。

### 剩余时间与状态边界

- `M3_NO_PRIOR`按最近10轮343.9秒/轮估计，剩余4轮约23分钟，保守范围20–35分钟。它是否触发同一导出错误须等真实终态，当前只记为高风险，不预判失败。
- 当前矩阵按“计划候选”计为：7条E200但导出失败、1条训练中、2条排队未启动、0条完成四场景评测；因此没有任何候选可标为`ARTIFACTS_COMPLETE`或`ANALYZED`。
- 本次为只读监控，没有启动补评、修改代码、重启队列或干预任何进程。下一步应先定点修复或绕开训练后Phase2原型导出对Phase1 final eval的阻断，再对已保留的7个checkpoint逐一执行strict clean+三LEO评测；该动作尚未在本快照中执行。

## 2026-08-21 final checkpoint恢复评测预登记

- 恢复评测ID：`final_eval_recovery_20260821_1135`；目的仅为使用已保留的`final_ssdg.pth`补齐held-out clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`四场景结果，不重训、不改checkpoint、不覆盖原训练目录中的既有文件。
- 代码与配置：每条checkpoint使用其原release中的`eval_ssdg_sat_per_rx.py`，共同设置`--eval_on unseen_rx --scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak --max_batches -1 --sat_seed 392002 --strict_reconstruction`；任何missing、unexpected、shape mismatch或fallback均作为该条评测技术失败。
- 首批GPU映射：GPU0=M0，GPU1=M2，GPU3=M3_NO_PROTO，GPU4=M3_NO_TEMPORAL，GPU5=M3_NO_SATELLITE，GPU6=M3_NO_CROSSRX，GPU7=M3_NO_NUISANCE。GPU2上的M3_NO_PRIOR仍在训练，训练结束并释放GPU2后再加入同一恢复评测ID；不干预其现有进程。
- 输出路径：每个候选原目录下新增且禁止覆盖的`final_eval_recovery_20260821_1135/`；预期包含`metrics_joint.json`、`eval_joint.log`、四个`metrics_<scenario>.json`、四个`eval_<scenario>.log`、评测状态和PID记录。
- 技术停止规则：strict重建失败、评测异常、指标计数不一致、输出目录冲突或缺少任一场景artifact时，仅停止并保留对应候选的恢复评测输出，不影响其他候选、原训练checkpoint或无关进程。完整四场景artifact产生前不标记`ARTIFACTS_COMPLETE`。

## 2026-08-21`final_ssdg.pth`严格测试与原型失败综合分析

### 结论先行

本轮把两个表面相似、实际独立的问题分开闭合：

1. 8条已完成训练的候选全部在Phase2原型导出时失败，直接根因不是“class3的`z_id`全部为零”，而是`V_cal`没有任何class3样本。校准代码把缺类的`0/0`强制记为零方向比例`1.0`，随后抛出具有误导性的zero-direction异常。该缺陷同时解释8条候选为何在同一class、同一阈值处失败。
2. 8个`final_ssdg.pth`均可严格重建并完成clean与3种LEO弱信道测试，因此checkpoint本体并未因原型导出异常损坏。但是，除数值崩溃的M0外，其余候选clean只有64.91%～75.85%，LEO均值只有31.23%～34.09%，显著低于历史`ADV3B02_CORE90_SOFT_E200`。这是真实的表征/训练失败，不能用导出bug解释或掩盖。
3. 当前可用候选中`M3_NO_SATELLITE`在clean和全部LEO聚合/底线指标上最佳。这是强诊断信号：当前卫星分支没有带来预期鲁棒性，反而与性能下降一致。但完整M3没有启动，故不能把该观察写成严格的单因素消融因果结论。

### 最终checkpoint严格测试结果

评测共同使用seed=`392002`、`--eval_on unseen_rx`，实际loader为`test_unseen_day_unseen_rx`：5个未见接收机、2个未见日期、每场景60,000条样本。每条恢复评测均满足：

- `strict_requested=true`；
- `checkpoint_load_strict=true`；
- `fallback_used=false`；
- missing/unexpected/shape mismatch均为0；
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`各有独立JSON和日志；
- 10个预期文件全部存在且非空，日志未发现Traceback、RuntimeError或`EVAL_FAILED`指纹，状态均为`ARTIFACTS_COMPLETE`。

| 候选 | clean | clean底线 | clear | low-elev | rain | LEO均值 | LEO三场景最差接收机底线范围 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 16.67% | 16.67% | 16.67% | 16.67% | 16.67% | 16.67% | 16.67%–16.67% |
| M2 | 73.33% | 55.62% | 33.91% | 33.19% | 33.48% | 33.53% | 22.92%–23.35% |
| M3_NO_PRIOR | 73.06% | 57.82% | 33.22% | 32.63% | 32.69% | 32.85% | 22.52%–23.27% |
| M3_NO_PROTO | 68.56% | 45.27% | 33.62% | 33.33% | 33.16% | 33.37% | 26.35%–26.61% |
| M3_NO_TEMPORAL | 72.21% | 59.91% | 31.89% | 30.99% | 30.80% | 31.23% | 22.12%–22.58% |
| **M3_NO_SATELLITE** | **75.85%** | **61.05%** | **34.57%** | **33.85%** | **33.86%** | **34.09%** | **26.05%–26.35%** |
| M3_NO_CROSSRX | 68.08% | 48.33% | 31.61% | 31.37% | 31.25% | 31.41% | 20.91%–21.49% |
| M3_NO_NUISANCE | 64.91% | 41.58% | 32.81% | 32.27% | 32.08% | 32.39% | 22.39%–22.67% |

恢复artifact统一位于各候选目录下的`final_eval_recovery_20260821_1135/`。例如M2完整结果位于：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_muse_ssdg_20260820_m23_r1/M2/final_eval_recovery_20260821_1135/
```

其中包含`metrics_joint.json`、`eval_joint.log`、4份`metrics_<scenario>.json`、4份`eval_<scenario>.log`和`status.txt`。其余候选分别沿用原训练目录；本次没有覆盖原日志或checkpoint。

### 与历史ADV3B02同口径比较

历史`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`保存于epoch194。其checkpoint保护指标和同epoch原始`metrics_epoch.csv`给出的严格UDU结果为：clean 86.09%、clear 72.56%、low-elev 69.87%、rain 69.27%、LEO均值70.56%。这与本轮同为`test_unseen_day_unseen_rx`口径，可用于直接诊断差距。

| 对象 | clean | clear | low-elev | rain | LEO均值 |
|---|---:|---:|---:|---:|---:|
| 历史ADV3B02，epoch194 | 86.09% | 72.56% | 69.87% | 69.27% | 70.56% |
| 当前最佳M3_NO_SATELLITE | 75.85% | 34.57% | 33.85% | 33.86% | 34.09% |
| 差值 | -10.25pp | -37.99pp | -36.02pp | -35.41pp | -36.47pp |
| M2 | 73.33% | 33.91% | 33.19% | 33.48% | 33.53% |
| M2差值 | -12.76pp | -38.64pp | -36.68pp | -35.79pp | -37.04pp |

模型在clean上仍保留部分区分能力，但LEO条件下额外损失约36～37pp，说明主要失败点不是简单的分类欠拟合，而是身份表征对接收机/信道扰动缺乏稳定性。

### 为什么“原型到处失败”

#### P0：`V_cal/V_select`切分按TX全局排序后直接一刀切

`code/SSDG/train_ssdg.py:1155-1173`先把整份source validation按`(tx_i,rx_i,day_i,eq_i,sig_i,index)`全局排序，再在总长度中点切成`V_cal`和`V_select`。在6个均衡TX、0.15/0.15等比例划分下，真实类别计数是：

| 角色 | 总数 | TX类别计数 |
|---|---:|---|
| L_s | 5,880 | 0–5各980 |
| U_s | 52,920 | 0–5各8,820 |
| V_cal | 12,600 | class0/1/2各4,200；class3/4/5为0 |
| V_select | 12,600 | class3/4/5各4,200；class0/1/2为0 |

因此，这不是随机偶发现象，而是当前排序键与50/50切点共同导致的确定性类分离。它有三重后果：

- Phase2原型校准要求逐类真实source-val样本，但`V_cal`只覆盖前3类；
- source选模/诊断只在后3类上计算，93%左右的`V_select`准确率不是6类完整验证结果；
- M1和完整M3所在的顺序launcher被首个候选非零退出阻断，至今未产生可比较checkpoint。

#### P0：校准器把“缺类”错误编码成“100%零向量”

`code/cvsrffi/phase2_prototypes.py:933-948`逐类统计零方向比例。当`class_count==0`时，代码直接令`class_zero_fraction=1.0`，随后抛出：

```text
endpoint_accept_v1 zero-direction fraction exceeds per-class limit:
class=3 fraction=1.000000000 limit=0.001000000
```

所以异常中的class3是第一个缺失类，不是已经证明其特征范数全为0。后续同文件`990-997`本来具有更准确的`insufficient true-class samples`错误，但程序在前面的零方向审计已经提前退出，无法到达该分支。

#### P1：导出失败错误地阻断了Phase1必需的最终评测

训练先写出`final_ssdg.pth`，随后原型导出抛异常，训练进程以`FAILED_EXPORT`结束；launcher把它统称为`TRAIN_FAILED`并跳过外部final eval。结果是权重本来可严格加载，却因独立的Phase2 bundle后处理错误而没有自动执行Phase1 clean和三LEO测试。本次恢复评测证明这8个checkpoint均可正常评测。

### 为什么方法性能失败

#### 1. 名义上“基于ADV3B02”，实际从头训练并丢弃强底座

M2最终checkpoint明确记录`from_scratch=true`、`baseline_ckpt=''`、`teacher_ckpt=''`，教师KL/MSE权重为0。也就是说，本轮继承了结构和部分目标配置，却没有继承历史86.09% clean、70.56% LEO均值的权重或教师约束。仅用7%有标签数据从头学习，再依赖63%无标签伪标签自举，风险远高于从强底座微调或蒸馏。

#### 2. 伪标签从第1轮起全量接收，形成无外部制动的确认偏差

M2在E1即`reliable=1.000`并接收52,864/52,864个`U_s`样本，平均置信度仅0.588；E200仍是52,864/52,864全接收，置信度0.949。与此同时E200有标签训练TX准确率只有48.70%。按协议，训练内不读取`U_s`真值，因此precision保持`N/A`是正确的数据权限行为；但“全部接收+低有标签准确率+验证只覆盖半数类别”说明伪标签没有可信的独立质量控制，极易把早期错误持续放大。

#### 3. 卫星分类损失很大，真正的信道不变性约束却为0

M2 E200的加权损失主项为：clean TX CE 6.2184、卫星CE 8.5318、domain CE 1.5409、group CE 1.4483、adversarial 0.9151；而prototype 0.0011、open-world feature 0.0003、source episode 0.0002。与此同时`lambda_sat_cons=0`、`lambda_zid_channel_invariance=0`，未标注域/对抗/卫星一致性/直接几何/quarantine权重也均为0。

这意味着卫星分支主要要求模型对强扰动样本继续做分类，却没有直接约束同一身份的clean与LEO`z_id`保持一致。主CE比几何正则大2～4个数量级，身份空间的跨信道稳定性无法与分类梯度竞争。`M3_NO_SATELLITE`反而成为当前最好候选，与这一失衡一致。

#### 4. `z_id`明显携带接收机、日期和信道信息，但泄漏guard被关闭

M2最终checkpoint的source train-to-val泄漏probe为：receiver excess 0.64897（上限0.20）、day excess 0.26746（上限0.15）、channel excess 0.42472（上限0.15）。三项都显著超限，但`zid_leakage_probe_required=false`，`joint_guard.zid_leakage_probe_fired=false`。这直接说明`z_id`没有实现所需的身份/域解耦，也是未见接收机和LEO条件下性能骤降的最强内部证据。

#### 5. 身份几何目标没有建立健康形态

M2 E200记录`source_overflow=0.8858`，约88.6%的source episode样本超出安全半径；`OW-FEAT`的domain alignment损失原值仍为0.6047，但加权open-world项只有0.0003。最终source-val几何又只覆盖3类，因此其known TPR、accept和proxy accept等指标不能代表完整6类。当前几何既没有足够的优化权重，也没有完整校准数据。

#### 6. M0在E173后发生数值崩溃，final-only仍导出了坏尾点

M0 E172尚为train 97.78%、source-val 99.06%左右；E173加权adversarial损失从0.9156跃升到364.3285，训练准确率降至36.82%；E174 adversarial升到704.7528、梯度总量986.274、训练准确率0%，有效step rate仅0.005；E175有效step rate为0，之后尾段基本无法更新。其最终严格测试恰为6类随机水平16.67%。

该候选设置`max_grad_norm=0`，没有梯度裁剪；source-val健康guard关闭、joint guard关闭、final-only不保留或选择E166附近的健康状态。因此M0不是“正常训练但泛化差”，而是训练尾段已经数值毁坏，仍机械写出了E200最终权重。

#### 7. 当前实验矩阵不能给出完整消融因果

完整M3和M1因前序导出非零退出而未启动。因此现在只能说：在8个已得到checkpoint的候选中，去卫星分支最好、去temporal最差之一、去nuisance损害clean明显；不能计算“完整M3减去某消融”的合法配对效应，也不能判定prototype本身对性能的独立贡献。`M3_NO_PROTO`能正常严格测试且LEO均值33.37%，进一步说明“所有候选导出失败”来自共享校准数据链，不是prototype训练分支是否启用。

### 修复优先级与下一轮最小实验

P0必须先修：

1. 将`V_cal/V_select`改为按TX分层的确定性切分；最好进一步在`(tx,rx,day,eq)`cell内分层，强制两边都覆盖全部已知TX，并新增类别覆盖负测。
2. 校准器先检查每类`class_count>=min_class_samples`，缺类时报`insufficient true-class samples`；零方向比例只对存在样本的类计算，不能把`0/0`改写成1.0。
3. 解耦Phase1最终评测与Phase2原型导出：只要`final_ssdg.pth`已写出且可严格恢复，clean和三LEO测试仍必须运行；导出失败单独标记，不得再次吞掉Phase1评测。
4. M0/MUSE统一启用有限梯度裁剪、非有限step硬停和泄漏超限硬停；final-only协议可保留，但不得在系统性数值崩溃后继续把坏尾点当有效最终模型。

下一轮只建议先做同seed最小配对，不立即扩大矩阵：

- A：历史ADV3B02权重初始化+当前7/63/15/15合法分层数据；
- B：A+延迟伪标签，仅在warmup后按类平衡、置信度/时间一致性联合门控，并限制每轮接收比例；
- C：B+降低卫星CE，同时启用明确的clean/LEO`z_id`一致性；
- D：C+MUSE prototype/local/prior完整机制。

每条仍使用seed392002、E200、final-only，并在训练后自动进行clean和3种LEO弱信道严格测试。先要求B/C相对A在LEO均值和接收机底线上出现同向增益，再进入完整M1/M2/M3消融。当前8条结果全部属于负证据，不晋级为新的Phase1底座。

### 状态与声明边界

- 8个已有`final_ssdg.pth`：四场景评测`ARTIFACTS_COMPLETE`，本节分析完成后记为`ANALYZED / NEGATIVE_EVIDENCE`。
- 原训练run的`FAILED_EXPORT`状态保留，不回写或伪装成训练成功；恢复评测是独立、不覆盖的补测。
- M1与完整M3：`NOT_STARTED`，没有checkpoint与性能数据。
- 该评测是Phase1冻结checkpoint的目标接收机诊断，不涉及Phase2 support、适配、新类注册或真实unknown声明。
