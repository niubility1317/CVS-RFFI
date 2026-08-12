# Phase1 CLIC后冻结source clean v3预注册报告

## 状态与目标

- 实验ID：`phase1_clic_postfreeze_20260812_v3`。
- 当前状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一运行席待当前target cache任务完成后交接。
- 目标：从不可变训练run`phase1_clic12_20260812_v5`重新导出F1—F6×C/G共12份source clean/source-V/fixed400 proxy特征工件，使clean manifest携带后续真实G bundle所需的source split、TX partition、source receiver/day及其完整性绑定。
- 本阶段只补齐不可变证据字段；不改变checkpoint、模型、source TX角色、fixed400 proxy、场景、阈值、loss、seed或C/G矩阵。

## 冻结输入、矩阵与输出

- 发布commit：`bbcef61506728bc9316ede7ca3b82df1cc524b42`。
- 训练输入：`runs/phase1_clic12_20260812_v5/F{1..6}{C,G}_CLIC12/{final_ssdg.pth,phase1_clic_terminal_receipt.json}`。
- ManySig：`Dataset_WigSig/ManySig.pkl`，SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- 输出根：`runs/phase1_clic_postfreeze_20260812_v3`；日志根：`logs/phase1_clic_postfreeze_20260812_v3`；均须预先不存在且不可覆盖。
- launcher：`code/scripts/launch_phase1_clic_postfreeze_source12_v3_20260812.sh`，SHA256=`8571C44883E2B1ADDAA22A1308E59B30D9307CC760A4C0C30F0D5DF2CF2C70E1`。
- exporter：`code/export_phase1_clic_features.py`，SHA256=`81C01F07C77091B7C84197B7A44301DC6109CCFE23A0581EA7C693A2A3A39BC4`。
- 固定12臂GPU映射：`0,1,2,3,4,5,6,7,0,1,2,3`；每GPU并发不超过2。

## 协议与完成条件

- 入口只读取ManySig、v5 checkpoint和terminal；target/query/truth/role访问为0，不读取target cache或任何目标指标。
- 每臂应产生一份`source_clean_proxy.npz`，角色仍为source-L=3920、source-V=16800、fixed400 proxy=400，总行数21120。
- 新manifest必须由真实exporter派生并封存source split receipt、TX partition receipt、source receiver/day集合及SHA；不得由调用方手写注入。
- 正式launch恰1次，retry=`NO`。错误checkout/hash、覆盖风险、协议越权或至少2臂在输出前出现同一确定性异常时，仅停止本run并封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 技术成功要求12/12 NPZ可重开、finite、schema/行数/角色/物理顺序/terminal绑定闭合；只标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`，不读取或报告accuracy、AUROC、u-gap等性能。

## 本地门与独立终审

- `ssr-gpu`完整postfreeze测试`132/132`通过；Phase1 CLIC核心回归`190/190`通过。
- 真实`CLEAN.export→bundle`、v5 checkpoint无`split_info`路径、C/G Torch2.1+NumPy2安全桥、PAIR原件/TOCTOU、union6→每折local4及ADV class×RX/day重算均有回归覆盖。
- launcher`bash -n`通过；dry-run精确12行，forbidden target/query/truth/role参数为0。
- 独立Terra终审：`P0=0，P1=0，ALLOW`。本结论仅授权进入真实技术发布，不构成性能结果。

## 待N607回填

- archive/SCP/release SHA与路径、静态门、唯一launch、PID/GPU/日志、12工件技术闭合与SSH清理证据。

## N607运行封存（2026-08-12）

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。这是一次source-only技术产物运行，不是性能实验；未读取accuracy、AUROC、u-gap或其他性能值。
- 冻结commit=`a5fd5668e1d676834d72cc00beaedb2f0bee02ad`；archive直接由该commit生成，未解包本地、未改dirty树；bytes=`267499520`，SHA256=`9C761CFE9C9006A20818EC229374A2C08E0526E33E7A733A450C276D2567177D`。SCP恰1次，远端bytes/SHA闭合。release为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_postfreeze_20260812_v3_a5fd5668`，stage解包后原子改名。
- 静态门：launcher物理SHA=`8571C44883E2B1ADDAA22A1308E59B30D9307CC760A4C0C30F0D5DF2CF2C70E1`、ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；exporter canonical SHA=`81C01F07C77091B7C84197B7A44301DC6109CCFE23A0581EA7C693A2A3A39BC4`，release/archive物理字节SHA=`101A3725D6FAF0186582D4F0FB0B238C5BF87AE535E2FEF4FEED946BE643F3B0`（Git archive物理CRLF表示；canonical化后与冻结81C一致）。`py_compile`、`--help`、`bash -n`、12行dry-run及forbidden target/query/truth/role=0均通过；F1C/F1G真实checkpoint只读validation smoke通过。
- 唯一formal命令：`nohup bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_postfreeze_20260812_v3_a5fd5668/code/scripts/launch_phase1_clic_postfreeze_source12_v3_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_postfreeze_20260812_v3_outer.out 2>&1 &`；formal invocation=`1`，retry=`NO`，outer PID=`2698889`。outer在run/log根外且0B；launcher创建run/log和完整12行PID表后启动12臂，GPU映射为`0,1,2,3,4,5,6,7,0,1,2,3`。
- 系统性故障：12/12臂在产生prediction/NPZ前退出，12个日志均706B且SHA256均为`dfbb9a33e8aec1d096f1657418a60da172b8432bd82791ad1446e062263b0ef6`。统一top指纹为`export_phase1_clic_features.py:708`（`main`）→`:702`（`export`）→`:603`，异常为`__main__.CLICSplitExportError: CLIC clean export source receiver aggregate drifts from split receipt`。这满足至少两独立row同一确定性异常的系统性停止规则；未重试、未重启、未清理任何非本run对象。
- 工件结果：run目录无`source_clean_proxy.npz`，12/12输出缺失（应有每臂21120行的L/V/proxy与manifest）；log目录保留12个错误日志及`pids_source12.tsv`（1818B），outer错误输出为空。该运行不可进入`ARTIFACTS_COMPLETE`或任何性能分析。
- 收尾证据：outer与12个child均已退出；GPU0—7均`0%`利用率、约`1MiB`显存；本地`ssh/scp`进程和到N607:22连接均为0。后续修复必须在本地新commit和全新run ID上进行，v3不可覆盖、不可续跑、不可重试。
