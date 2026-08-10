# Phase1 P1-RECTE后冻结42步实验报告

## 1.状态与目标

- 实验ID：phase1_recte_postfreeze_20260810_v1
- 日期：2026-08-10
- 当前状态：PRELAUNCH_REPAIR_VERIFIED / PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT
- 操作边界：主代理冻结评价合同、矩阵和判定门；唯一N607 Runner只负责release落地、唯一启动、技术监控和小工件回收。
- 训练输入：phase1_recte12_20260810_v1，已技术闭合12/12臂；训练报告SHA256=013216a56da310ea4ae0b082904719255099fcc54985724870bdb8fc4c4a85bf，Git镜像commit=cd8daa75。
- 目标：对同fold C/G执行固定clean、三LEO、fixed400 proxy和连续Gaussian-NLL公平评价，产出6份pair JSON及F6矩阵聚合。
- 声明边界：技术完成不等于性能通过；任一非补偿门失败即REJECT_P1_RECTE_PERMANENT，全部通过也只能PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW。

## 2.冻结版本、本地文件与独立审查

- Git仓库：E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt
- 后冻结实现commit：cb92090fd3247e9e9fb0d4bff9f28496a071ef6c
- 独立actual-diff终裁：P0=0 / P1=0 / ALLOW
- 审查边界：ALLOW只允许技术发布与Runner交接，不包含性能结果、方法晋级或N607已执行声明。

|文件|工作树SHA256|用途|
|---|---|---|
|analysis/phase1_recte_postfreeze_design_20260810.md|47c24a0e0cf45082e7849fd39bbad9f4d3804a87d8f1caea1d6a22f2076c1865|后冻结设计、追踪和证据边界|
|code/export_phase1_recte_features.py|d3a99ede833d961fa4092841e2eb69f5c8cf608b12cf827c6cc2ba1fa76c2ea3|clean L/V/proxy专用导出|
|code/export_phase1_recte_leo_features.py|37b248a0c80e05538d941ee8e51df8f3aaed6d49ecefeb55cc85d0aa8d6359bb|三LEO导出与物理绑定|
|code/evaluate_phase1_recte_postfreeze_pair.py|78bd0de61d1ddb64c75fb6f3dcea0e4465592fc019dc1d310112876c5ad2ab00|同fold C/G评分与F6原始工件聚合|
|code/tests/test_phase1_recte_postfreeze.py|1921589e0d1d1f4dd0c9989a97866992230af4966d3a2da1ae96f5c46f477540|receipt、物理绑定、篡改与门测试|
|code/scripts/launch_phase1_recte_postfreeze_20260810.sh|4aa54708540be5e4eeae5d42165e9ef1ca35ad4218981a7c33c49a2f4f8a1abe|冻结42步launcher，Git mode100755|

本地ssr-gpu串行验证：

- 四个Python文件py_compile通过。
- RECTE后冻结专项：33 passed。
- 真实F1C签字terminal receipt内存回归：source/frozen receiver IDs、source_receiver_count=7、SHA/provenance、28/378全部通过；不再要求训练core从未定义的frozen_source_receiver_count。
- RCAT+RCRMD共享公平核回归：60 passed。
- bash -n通过。
- dry-run精确42步=12 clean+12 LEO/binding+12 proxy+6 pair，旧ICMT/RCAT/RCRMD/CAGM运行身份为0。
- git diff --check通过。
- 定向负测覆盖旧身份注入、source/LEO/proxy篡改、1-row proxy、F6 summary/raw tamper、非有限值和非补偿拒绝。

## 3.冻结训练原件

ManySig预期SHA256=2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f。训练root固定为/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte12_20260810_v1。

|候选|final checkpoint SHA256|
|---|---|
|F1C_RECTE12|4b8c24733196644b0cf0839efc7a165ba79a268086eb1457920633ae9e2afd49|
|F1G_RECTE12|fe0688d39d42b1746b3f2137e58764167395f09e14fd0bb397f8e4bc8b8e3d3a|
|F2C_RECTE12|5cc84ad5984a4c45627228734caff0f6b8c9e0cb36830bf93bcbd33122c3b1bc|
|F2G_RECTE12|d9971a30d7718fe75effb43cad0cc0995c660eaf0dc6e822cdbb9a6641d12a92|
|F3C_RECTE12|1b143700a15e378c6961f4a232d2a25151caa448c3fb3300ae29e2555668577c|
|F3G_RECTE12|53dc14892610ca9908170ab2d939a32beba7b3a2a3408f3fe4086b0b9b14c4d9|
|F4C_RECTE12|e8940f6b1366b6ebfe7f00f2f0b5a6fb1c5d8dff41a7fdfdd7b1a05e2fd25ac7|
|F4G_RECTE12|117f58eca2277c9dc212e1dbe8ffd07c879d9411fa4eed9080eb3f0e783d7172|
|F5C_RECTE12|5e4df2aea0065e08a401826dc977094349a16958d3aaa5021c4c496efd67f98a|
|F5G_RECTE12|fcb5406a56b23a65627bf556cc00f31371ff931a927aa78d3a06bed533f0d4a9|
|F6C_RECTE12|9055408c27b034cf02b4a67b6ddaae1d9bf7dea36c27b6289fe99e6d946478c8|
|F6G_RECTE12|728bea80024a758fe473ace53196233448553165091571b7d236140b51428416|

Runner必须逐项重验ManySig、12个checkpoint SHA和RECTE terminal receipt，不下载checkpoint。

## 4.冻结评价合同与非补偿门

- clean只以source-L的feat_joint拟合float64 totalized-L2对角Gaussian；精确zero映射0并保留，nonfinite fatal；V/proxy零fit。
- 每类方差使用ddof=1，按0.9×class+0.1×class-equal pooled收缩，方差下限1e-6；评分使用完整Gaussian-NLL和稳定logsumexp连续u。
- fixed proxy：days=2021_03_01,2021_03_08；RX=1-1,1-19,14-7,18-2,19-2,2-1；selection seed=7281148；max/TX=400；total=400。
- LEO绑定封存ManySig path/SHA、selection、physical ID及每scene TX/RX/day完整覆盖。
- pair重开C/G训练receipt并验证Rs0..6、28格、fixed378、lambda、functional equality、三scene positive-tail/VJP和共同训练投影。
- F6必须重开F1--F5的clean NPZ、LEO NPZ、binding、proxy JSON/CSV与当前checkpoint，在当前SHA下重算；不得信任prior自报summary。
- clean要求6/6折overall、min-class、min-RX、min-day均不低于C−2pp。
- LEO要求18/18场景格四floor均不低于C−2pp。
- 每fold三场景等权overall增量及全18格等权overall增量均不得为负。
- proxy要求每foldΔAUROC>0且Δ(mean u_proxy−mean u_V)>0，必须6/6。
- clean、LEO、四floor、fold/global overall与proxy双门互不补偿。

## 5.N607路径、资源与唯一命令

- Python：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
- 项目根：/home/szu2070436088/2510044040/CV-SincNet
- 预计release：/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_cb92090f
- 精确CWD：<release>/code
- 训练root：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte12_20260810_v1
- 后冻结root：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte_postfreeze_20260810_v1
- log root：/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1
- outer：/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1_launcher.out

GPU映射沿用训练：F1C+F5G/0，F1G+F5C/1，F2C+F6G/2，F2G+F6C/3，F3C/4，F3G/5，F4C/6，F4G/7；pair按F1至F6 CPU串行。启动前只读记录现有GPU任务，不干预无关进程；每GPU总实验数不得超过2。

唯一启动命令：

    cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_cb92090f/code && nohup env POSTFREEZE_RUN_ID=phase1_recte_postfreeze_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_cb92090f/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte12_20260810_v1 POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte_postfreeze_20260810_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_cb92090f/code/scripts/launch_phase1_recte_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1_launcher.out 2>&1 < /dev/null &

调用只能1次，retry=NO。调用端超时后先清理本地SSH/TCP22，再只读核run/log/PID/CWD/cmdline是否已landed，严禁重发。

## 6.技术健康、停止规则与预期工件

启动前必须通过direct preflight、archive/member/mode、release静态py_compile/help/bash-n/dry-run42、ManySig/12 checkpoint SHA、run/log/outer ABSENT和GPU资源核验。完整Git archive必须无prefix、LF-only、code/code=0；不得直接复制Windows mixed-EOL工作树。

预期工件：12 clean NPZ、12 LEO NPZ、12 LEO binding、12 proxy JSON、12 proxy CSV、6 pair JSON、18阶段日志、PID表和outer。技术门检查schema、root、matrix、training-root、checkpoint、RECTE receipt、common/proxy/physical binding及F6 aggregate。

仅在错误checkout/hash、覆盖风险、协议/P0违反、launcher-wide确定性故障，或至少两个distinct candidate在产出目标工件前出现相同确定性异常时停止。停止前精确核run-owned PID/CWD/cmdline，只停止本run并保留partial。不得读取accuracy、floor、AUROC、u-gap等性能字段决定是否停止。

Runner只回收小JSON/CSV/binding/log/PID/manifest，不下载checkpoint或NPZ；技术交接先标记PAIR_JSON_READY / NO_PERFORMANCE_INTERPRETATION。主代理在6/6 pair和原始工件闭合后才读取性能并作最终判定。

## 7.预注册交接状态

- 本地实现：已提交。
- 本地验证：已完成。
- 独立P0/P1：已通过。
- 根报告与Git镜像：本次预注册后应逐字一致。
- N607 release/42步：尚未执行。
- 性能分析：尚未开始。

## 8.Runner技术交接（2026-08-10）

- 状态：`PRELAUNCH_BLOCKED_RECEIPT_SCHEMA_MISMATCH / NO_PERFORMANCE_RESULT`。
- Runner边界：唯一启动调用次数为0；未读取或解释任何性能字段；未启动后续实验。
- Direct preflight：`tools\\n607_ssh_preflight.ps1`通过。N607时间为2026-08-10 22:35:48 CST，项目根可见，8张RTX3090均无compute app、显存占用各1MiB。
- 启动目标保护：run root、log root和outer在静态门后仍为`ABSENT`；无run-owned PID、无GPU任务、无阶段日志。
- 训练原件只读核验：ManySig SHA256与12个`final_ssdg.pth`逐项匹配§3预注册值；未下载checkpoint。
- Release archive：从实现commit`b95aac57b82f623f729c2ac24c1793664c112ca1`生成raw Git archive（SHA256=`f9502a3bc48fe6f2fc5dfdda6b3bb3fec2aba7ebd3becdd841b2a6023e76529a`），仅做launch前机械CRLF→LF规范化后最终archive SHA256=`bf734f3be45d0dd3b4a2fc87b8dcdf686c10c92a09546fbc58cbfa417f4b69bc`、bytes=`266762240`。最终archive为无prefix、4951成员、文本CR计数0；六个冻结member SHA全匹配，launcher mode=`775`。远端archive保留于`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_b95aac57.gitarchive_lf.tar`。
- 原子落地：首轮stage误将含顶层`code/`的archive解压到`stage/code`，触发`code/code=1`并在`mv`前安全退出；partial证据保留于`.../phase1_recte_postfreeze_20260810_v1_b95aac57.stage`。随后使用全新`stage2`从archive根解压，核验`code/code=0`、4951成员、无prefix、launcher可执行和六member SHA后原子`mv`到`.../releases/phase1_recte_postfreeze_20260810_v1_b95aac57`。
- 远端静态门：release内5个Python源以无写盘compile通过；4个CLI`--help`通过；`bash -n`通过；launcher`--dry-run`精确42步。未产生pycache、run、log或outer。
- RECTE terminal receipt只读审计：12/12臂均为`schema=cvs.phase1.recte_receipt.v1`、`method=P1_RECTE`、`frozen_mode=true`、`checkpoint_role=training_final_only`，C/G的`enabled/lambda`与候选一致，`recte_terminal_contract_passed=true`；共同`common_l_base_head_input_path_verified=true`、`common_batch_sequence=1200/153600`、三场景各28cells、`source_receiver_count=7`。但12臂均缺少键`frozen_source_receiver_count`（不存在/None），而当前release validator要求该键严格等于7；首个F1C在`validate_recte_training_checkpoint`处因该确定性schema缺键失败。该训练原件与实现commit均冻结，Runner未修改任何方法、代码、checkpoint或远端证据，因此不启动。
- 工件与回收：预期42步均为0；12 clean NPZ、12 LEO NPZ、12 binding、12 proxy JSON、12 proxy CSV、6 pair JSON、18日志、PID表和bundle均未生成/未回收；未下载NPZ、pth、pt、npy。
- SSH清理：每次SSH/SCP返回后本地`ssh.exe=0`且N607/bridge TCP22 established=0；当前无遗留连接。
- 报告边界：本节记录旧release的技术落地与预启动阻断，不构成方法晋级或任何性能结论。由于唯一启动调用仍为0且run/log/outer始终不存在，修复后保留同一冻结run ID，但必须使用新的非覆盖release路径；旧release与partial证据继续保留。

## 9.真实receipt兼容修复与重新放行

- 根因：签字训练core只产出frozen_source_receiver_ids和source_receiver_count，从未定义frozen_source_receiver_count；旧后冻结夹具人为补入该字段，掩盖了首次真实receipt不兼容。
- 最小修复commit：cb92090fd3247e9e9fb0d4bff9f28496a071ef6c，仅在export/pair/test删除4行虚构字段要求；方法、评价公式、门、矩阵、launcher和运行ID均未改变。
- 严格性保持：两个receiver-ID列表仍必须为0..6，source_receiver_count必须为7，且IDs SHA、provenance、28 cells、378分母和C/G共同投影继续fail-closed。
- 验证：py_compile通过；RECTE后冻结33 passed；真实F1C terminal receipt内存校验通过；git diff --check通过。
- 独立修复终裁：P0=0 / P1=0 / ALLOW。该ALLOW只覆盖prelaunch兼容修复，不代表N607已执行或产生性能结论。
- 新release固定为.../phase1_recte_postfreeze_20260810_v1_cb92090f；旧b95aac57 release不得覆盖或用于启动。

## 10.新release Runner执行与技术闭合（2026-08-10）

- 状态：`ARTIFACTS_COMPLETE / TECHNICAL_BINDING_PASS / PAIR_JSON_READY / NO_PERFORMANCE_INTERPRETATION`。
- 冻结版本：实现commit=`cb92090fd3247e9e9fb0d4bff9f28496a071ef6c`，reprereg commit=`cc17988e8c2f5a93408c35b6487745aa3dbee001`，新release=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte_postfreeze_20260810_v1_cb92090f`。最终LF-only无prefix archive SHA256=`b1b3f5aa694bc8b32842d34501ccc9f3f67ed484cf470fb507deb92376d3c1af`、bytes=`266772480`、4952成员；code/code=0，launcher mode=`775`，六member SHA全匹配。
- 唯一启动：冻结新路径命令调用恰好1次，SSH exit=0（约112秒返回），retry=NO；未覆盖旧b95 release、stage或raw archive。
- PID/GPU/CWD：`candidate_pids.tsv`记录12个候选PID及预注册GPU映射，启动CWD为新release`code`；首波核验时12个run-owned PID均已退出，8张GPU均无compute app、显存占用各1MiB，未发现run-owned残留进程。
- 工件闭合：run内文件总数66（12 clean NPZ、12 LEO NPZ、12 binding、12 proxy JSON、12 proxy CSV、6 pair JSON）；log内19文件（18阶段日志+PID表）；outer存在且为0 bytes；0个禁止的`.pth/.pt/.npy`输出。
- 技术schema核验：12/12 binding的LEO schema、P1_RECTE receipt、terminal contract、source receiver count=7、28 cells、378分母、common physical/RX/class/scene与batch-order绑定均通过；12/12 proxy JSON可解析、CSV非空；6/6 pair schema、C/G receipt revalidation、common training binding、proxy recomputation和F6 raw-reopen标志通过，F6 aggregate键存在。
- 健康证据：阶段日志均非空；预定义`Traceback/RuntimeError/CUDA error/OOM/unrecognized argument/No such file/Permission denied`技术指纹计数为0。未按任何性能字段停止或作解释。
- 小工件bundle：远端`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte_postfreeze_20260810_v1/phase1_recte_postfreeze_20260810_v1_small_bundle.tar`，SHA256=`1642deea7824b73a8654e2e3eba6d0cda45805d132077d5aa6b2d6a71afff28e`、bytes=`55910400`、63成员；manifest为同目录`phase1_recte_postfreeze_20260810_v1_small_bundle_manifest.tsv`（6793 bytes）。本地回收副本为`E:\type10-7\automation_reports\CV-SincNet\phase1_recte_postfreeze_20260810_v1\retrieved_cb92090f\phase1_recte_postfreeze_20260810_v1_small_bundle.tar`，SHA/bytes一致，bundle内无NPZ/pth/pt/npy。
- SSH清理：启动及回收后均确认本地`ssh.exe=0`、N607/bridge TCP22 established=0；当前无遗留连接。
- 审计备注：首个落地探针曾因错误使用`nan`匹配模式触发单行日志内容输出；该输出未被读取、解释或用于任何判断，随后已改为仅计数技术错误指纹和校验结构字段。本节及Runner交接不包含性能结论；主代理在同一run的完整原始工件闭合后自行进行性能分析。
