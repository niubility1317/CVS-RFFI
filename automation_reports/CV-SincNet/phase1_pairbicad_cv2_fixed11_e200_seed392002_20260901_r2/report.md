# Phase1 PairBiCAD-CV2修复版E200正式矩阵r2

## 当前状态

- 状态：`LOCAL_FIX_IN_VERIFICATION / RELEASE_PENDING`。
- run ID：`phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r2`。
- r1因U批次物理ID与标签隔离冲突触发预登记系统技术停止；0行完成，partial artifact完整保留，不产生性能结论。

## 候选与矩阵

- 候选：`CV2-B0/B1/B2/B3/D0/D1/D2/D3/T0/T1/T2/T3`。
- fold：1、8；seed：392002；总计24行。
- 每行从头完整训练200epochs；不使用6500updates、coverage周期、收敛状态或24小时作为正常停止条件。
- ManySig day1/day2/day3；fold1训练RX3/4/6/8、held-out RX1；fold8训练RX1/3/4/6、held-out RX8。
- `L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，严格Phase1 source-only。
- `concat_sat_ce_only`，`lambda_sat_cons=0`，使用`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`三种真实LEO弱信道。
- N607普通账户；GPU0—7每张最多2个本run训练进程；资源全空闲时16行并发、8行排队。

## 11项机制与r2定点修复

- 11项：严格LEO pair、固定200epoch、真实CoverageLedger、coverage warmup、可执行no-early-freeze、显式双时间尺度、pair梯度5%上限、困难组30%上限、四反馈动态GRL、`V_cal/V_select`隔离、final/EMA/SWAD一次选模。
- r2定点修复：U批次继续删除`tx_i/true_tx_i`等TX真值；CoverageLedger以数据集已输出的不可变`base_index`作为标签无关物理行ID，并与冻结U子集`selected`索引逐项对齐。完整五元`(tx_i,rx_i,day_i,eq_i,sig_i)`路径仍用于标签可见批次和L组审计。
- 不允许伪造缺失ID；`base_index`缺失、长度不一致、非整数、重复或账本未知ID仍直接失败。

## 预期artifact与停止规则

每行必须闭合epoch200 final checkpoint、严格重建、final/EMA/SWAD候选与一次`V_select`选择、Coverage/LR/机制/梯度遥测，以及Clean和三种LEO弱场景独立JSON，最后生成`ARTIFACTS_COMPLETE.json`。

只允许因数据/query越权、错误candidate/fold/receiver/day/seed/epoch、输出冲突、错误release/CWD、命令无法运行、同一确定性异常重复、进程归属不清或无法形成合法checkpoint/四场景artifact而停止精确run进程树。低性能和中间指标下降不得停止、重启、热补丁或选择性重跑。

## 本地验证与审查

- r1异常先由回归测试复现为RED；修复后标签无关`base_index`批次ID、冻结subset ID和CoverageLedger同源记录测试转为GREEN。
- 完整`code/tests/phase1_bicad_xr`回归471项全部通过；仅3条既存PyTorch autocast弃用警告。
- `train_ssdg.py`和新launcher编译通过；`git diff --check`通过。
- launcher静态读回：run ID为r2、24行、12候选、fold1/8、seed392002、全部200epochs、每GPU上限2槽。
- 原实现已经完成一次独立P0/P1审查；本次针对运行时异常仅核对ID同源、标签隔离、L五元路径和r2不可覆盖性，未发现未解决P0/P1。两次Luna定点复审实例均未在限时内返回，按最小流程记为`NONBLOCKING/REJECTED_EXTRA_GATE`，不增加重复审查轮次。
