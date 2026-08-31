# Phase1 PairBiCAD-CV2修复版E200正式矩阵

## 当前状态

- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- run ID：`phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r1`。
- 旧run：`phase1_pairbicad_cv2_screen24_seed392002_20260831_r1`已由用户停止；16个完整行和partial artifact保留，仅作历史证据。
- 设计冻结：[PairBiCAD-CV2-E200修复版冻结说明](../../../docs/superpowers/specs/2026-09-01-pairbicad-cv2-e200-repair.md)。

## 候选与矩阵

- 候选：`CV2-B0/B1/B2/B3/D0/D1/D2/D3/T0/T1/T2/T3`。
- fold：1、8；seed：392002；总计24行。
- 训练：每行从头完整200epochs；不使用6500updates、coverage周期或24小时作为正常停止条件。
- 数据：ManySig，day1/day2/day3；fold1训练RX3/4/6/8、held-out RX1，fold8训练RX1/3/4/6、held-out RX8。
- 角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，Phase1 source-only。
- 信道：`concat_sat_ce_only`；`lambda_sat_cons=0`；三种`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- 资源：N607普通账户；GPU0—7每张最多2个本run主训练进程；全空闲时16行并发、8行排队。

## 本次修复范围

严格LEO pair、200epoch终止、真实CoverageLedger、coverage warmup、可执行no-early-freeze、显式双时间尺度、pair梯度5%上限、困难组30%上限、四反馈动态GRL、`V_cal/V_select`职责隔离以及final/EMA/SWAD一次选模。报告明确延期机制不进入本run。

## 预期artifact

每行必须生成：

- epoch200 final checkpoint及严格重建信息；
- final/EMA/SWAD候选和一次`V_select`选择记录；
- CoverageLedger、LR阶段、机制调用和梯度比例遥测；
- Clean及三种`leo_*_weak`独立评估JSON；
- worker终态和`ARTIFACTS_COMPLETE.json`。

## 技术停止规则

只允许因数据/query越权、错误candidate/fold/receiver/day/seed/epoch、输出冲突、错误release/CWD、命令无法运行、同一确定性异常重复、进程归属不清或无法形成合法checkpoint/四场景artifact而停止精确run进程树。低性能、未达到科学门槛或中间指标下降不得停止、重启、热补丁或选择性重跑。

## 本地实现与验证

- 11项修复均已进入真实训练路径，不是仅增加配置字段；实现追踪为`verified=11/deferred=0/rejected=0/blocked=0`。
- 新launcher静态计划为24行、12个候选ID、fold1/8、seed392002、200epochs、GPU0—7每卡2槽；满资源时16行并发、8行排队。
- 完整`code/tests/phase1_bicad_xr`回归470项全部通过；仅3条既存PyTorch autocast弃用警告。
- 改动模块`py_compile`通过，`git diff --check`通过。
- 独立P0/P1审查曾发现分离优化器入口仍间接依赖`detached_adversarial`；现已改为直接消费`adversarial_two_time_scale`及其1.5倍LR契约，定点RED→GREEN闭合。当前无未解决P0/P1。
- 尚未把“本地行为验证”冒充“N607运行时验证”；真实epoch200、CoverageLedger、V_cal/V_select、final/EMA/SWAD选择、机制审计和四场景评估须等待新run artifact。

## N607发布与技术停止

- Git/release代码提交：`f045c6c77d3e937f75e670cc8502388da57e5a5a`；release：`phase1_pairbicad_cv2_e200_f045c6c7`。
- release归档本地/远端SHA256一致：`00e021c99f7c4ad07cbb7c38c68b97356558263827e8706649188c28d15859a0`；远端编译和真实checkpoint无query烟测通过。
- dispatcher：PID3187719；启动时16个worker、每GPU2个训练进程均精确绑定本run和release。
- 至少14行在prediction前重复触发同一异常：`ValueError: CV2 U batch is missing physical sample ID metadata: tx_i`。这是预登记的确定性系统技术失败，不是性能结果。
- 已先冻结dispatcher，再按精确父子树终止PID3187736、3187739、3187720、3187722、3187719、3187718；独立读回无残留绑定进程，GPU计算进程为0。
- 0行达到`ARTIFACTS_COMPLETE`；22个非空训练日志和全部partial artifact保留，远端已写入`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE.json`。
- 根因：MUSE按协议从U批次删除TX真值，但CoverageLedger错误要求`tx_i`参与物理ID。r2改用冻结数据集已有、标签无关且唯一的`base_index`，不恢复或泄露U标签。
