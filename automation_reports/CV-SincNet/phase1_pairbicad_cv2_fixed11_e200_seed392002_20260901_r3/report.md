# Phase1 PairBiCAD-CV2修复版E200正式矩阵r3

## 当前状态

- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- run ID：`phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r3`。
- r1、r2均按预登记系统技术失败规则停止，0行完成且partial artifact保留，不产生性能结论。

## 冻结矩阵

- 12候选：`CV2-B0/B1/B2/B3/D0/D1/D2/D3/T0/T1/T2/T3`；fold1/8；seed392002；共24行。
- 每行从头训练完整200epochs，不使用6500updates、coverage、收敛状态或墙钟作为正常终止条件。
- ManySig day1/day2/day3；`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；严格source-only。
- `concat_sat_ce_only`、`lambda_sat_cons=0`和三种真实`LEO_WEAK`课程。
- GPU0存在一个无关任务时仅分配1槽，GPU1—7各2槽；15行并发、9行排队；不得影响无关任务。

## 11项机制与r3定点修复

11项机制保持不变：严格LEO pair、固定200epoch、真实CoverageLedger、coverage warmup、可执行no-early-freeze、显式双时间尺度、pair梯度5%上限、困难组30%上限、四反馈动态GRL、`V_cal/V_select`隔离、final/EMA/SWAD一次选模。

r3按数据视图选择同源ID：

- 真正经过`_MUSEUnlabeledDatasetView`且TX字段被删除的U，批次与冻结subset都使用不可变`base_index`；
- 普通U批次与冻结dataset index都使用`(tx_i,rx_i,day_i,eq_i,sig_i)`五元物理ID；
- 两条路径均检查非空、长度、整数、唯一性和账本成员关系，不伪造缺失ID；MUSE路径不恢复TX真值。

## 预期artifact与停止规则

每行必须闭合epoch200 final checkpoint、严格重建、final/EMA/SWAD与一次`V_select`选择、Coverage/LR/机制/梯度遥测、Clean和三种LEO弱场景独立JSON以及`ARTIFACTS_COMPLETE.json`。

只允许因数据/query越权、错误candidate/fold/receiver/day/seed/epoch、输出冲突、错误release/CWD、命令无法运行、同一确定性异常重复、进程归属不清或无法形成合法checkpoint/四场景artifact而停止精确run进程树。低性能和中间指标下降不得停止、重启、热补丁或选择性重跑。

## 本地验证

- 新增普通U五元ID与MUSE U `base_index`两条行为测试；r2错误先复现为RED，分流修复后转为GREEN。
- 完整`code/tests/phase1_bicad_xr`回归472项全部通过；仅3条既存PyTorch autocast弃用警告。
- 训练入口与launcher编译通过，`git diff --check`通过。
- launcher静态读回：r3、24行、12候选、全部200epochs、每GPU最多2槽。

## N607发布与启动证据

- 代码提交：`394bee074bc191538fc1e09c3713a016b705fb62`；已自动push并独立核对远端OID一致。
- release：`phase1_pairbicad_cv2_e200_394bee07`；归档本地/远端SHA256均为`06bd988e8f362a2d10d91ed060c192d9e8c4f1fcf8acf02138727b57cc9476d8`；远端编译通过。
- 历史真实checkpoint无query烟测通过：严格重建缺失/意外/shape mismatch均为0，一次优化器步完成，Clean和三种LEO弱场景均有限值。
- 真实ManySig双路径烟测通过：普通`CV2-B0`批次96个五元ID全部属于冻结U集合且CoverageLedger接受；MUSE `CV2-T3`批次32个`base_index`全部属于冻结U集合、CoverageLedger接受且TX字段不存在。
- 启动时间：`2026-09-01T02:33+08:00`；dispatcher PID3235006，launcher wrapper PID3235005；CWD精确绑定上述release。
- `plan.json`独立读回：24行、全部200epochs、GPU容量`0:1,1:2,...,7:2`、15行并发、9行排队。
- 启动127秒检查：15个直属worker持续存在；GPU上15个本run计算进程加受保护的无关PID3208551；`TECHNICAL_FAILURE=0`、致命异常为空。健康运行不得因中间性能停止或热补丁。

## 03:10系统技术停止

- 运行约37分钟后，`CV2-B0-F1-S392002`和`CV2-B0-F8-S392002`在必需source-LORO评估阶段重复触发同一确定性异常：`TypeError: _forward_unimplemented() got an unexpected keyword argument 'y_tx'`。
- 异常发生于`_evaluate_bicad_xr_source_loro`调用静态B0模型包装；它不是性能结果。两行均生成`TECHNICAL_FAILURE.json`，0行达到`ARTIFACTS_COMPLETE`。
- 按预登记规则先冻结dispatcher PID3235006，再仅终止其精确后代树及wrapper PID3235005；全部partial artifact保留。独立读回两个绑定PID及其后代均不存在，GPU只剩受保护的无关PID3208551。
- r3结论固定为`NO_PERFORMANCE_RESULT`，禁止用其未闭合source-LORO曲线作候选比较。后续修复必须让静态B0模型通过统一评估适配器，而不是直接把训练模型调用约定施加到无`forward`实现的包装器上，并以新run ID发布。
