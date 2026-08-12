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
