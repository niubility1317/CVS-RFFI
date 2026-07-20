# D88地面sigma逐类Pareto保护实验报告

## 预注册设计

- 实验ID：`d88_ground_sigma_pareto_guard_20260720`
- 状态：`PLANNED_LOCAL_DEVELOPMENT_DIAGNOSTIC`
- 目标：修复D87虽使注册后旧类准确率由82.78%升至85.00%、遗忘由10.00%降至7.78%，但新类准确率由84.67%降至83.33%的同row冲突。
- 假设：D87的地面半径sigma方向具有真实旧域适应信号；新类退化来自每步只约束聚合smooth-worst sigma风险，允许个别类的clean OOF CE上升。若把同一方向投影到所有已注册类clean OOF CE共同非增锥，再做逐类精确回溯，可保留部分旧类收益并消除新类系统性回退。
- 单一主要差异：相对D87只增加逐类、角色无关的common-descent cone projection与相对未更新D62点的exact per-class line-search guard；地面v2组件、14个方向、半径幅度、sigma权重、20步、rank13空间、D78 trust ball、D79中心化仿射编译全部不变。
- 类对称性：当前row全部11个注册类使用同一公式；不读取old/new角色、class ID、query标签、query组成或class quota。
- 数据边界：复用匹配`VALIDATED_ONCE`的固定单LEO弱观测；反事实sigma view仅为同一received IQ的数学视图，不增加K；不读取clean/source样本。
- 预期可观察结果：全部目标row均满足`max_class_oof_clean_ce_delta<=数值容差`；D87发生变化的4/15个outer row中，clear/fold2的新类损失应被抑制，同时尽量保留low/fold0和rain/fold3的旧类收益。
- 失败/停止条件：若残差15/15全部回退为零，则说明D87地面sigma方向没有全类共同下降空间；若仍有`seen_new_acc<84.67%`或注册后旧类不高于D85的82.78%，本路线不进入seed2/125；不扫描权重、半径或trust参数。
- 最小验证矩阵：先跑核心与probe单测；通过后锁定development seed、K10/new5、3场景×5fold×INT8/FP32 matched的105行完整probe，并报告同row总体、逐场景、15行、逐类、混淆、量化和资源。
- Phase1组件声明：当前v2组件仍为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，因此D88即使性能改善也只能作为forced development diagnostic，不能直接晋升正式确认。

## 版本与执行计划

- 本地Git工作树：`E:\type10-7\code\snapshots\d81wt`
- 核心：`code/cvsrffi/stage2_d88_ground_sigma_pareto_guard.py`
- probe：`code/scripts/probe_d88_ground_sigma_pareto_guard.py`
- 测试：`tests/test_stage2_d88_ground_sigma_pareto_guard.py`、`tests/test_probe_d88_ground_sigma_pareto_guard.py`
- 环境：`ssr-gpu`
- 工作目录：本地开发cell；本轮无需N607，不占用GPU，不创建SSH连接。
- 输出根：`E:\type10-7\automation_reports\CV-SincNet\d88_ground_sigma_pareto_guard_20260720\ground_sigma_pareto_guard_centered_head`
- 预期artifact：`training_log.jsonl`、`predictions.jsonl`、`predictions.receipt.json`、`D88_PROBE_METADATA.json`、完整性能汇总与本报告结果段。

## 完成后待填

- 最终状态、耗时、PID、退出码。
- 同row总体/逐场景/15行/逐类/混淆指标。
- 逐类Pareto保护是否真正消除D87的新类回退。
- INT8/FP32量化差异、资源审计、主要缺陷及下一轮建议。

## 尝试记录

- attempt0于2026-07-20 08:35:42 CST启动，PID=`12868`，在首个目标fit的最终Pareto审计前退出，training row=`0`。stderr显示逐步数值容差累计后超过更小的最终容差；属于验证器/数值闭包失败，不是性能结果。
- 直接修复：每步精确验收统一改为相对初始D62逐类clean OOF CE上界，而不是相对上一步并累计容差；最终审计复用同一个固定容差。未改变数据、地面组件、目标函数、方向投影、步数、trust或实验矩阵。
