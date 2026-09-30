# Joint channel完整训练日志分析

`tools/collect_d92_channel_training_diagnostics.py`是仅依赖Python标准库的只读采集器。它先核对四row、160个parent全部完成及实际配置、来源和计数，再逐条读取各row的`fit_stages.jsonl/csv`、`training_events.jsonl`、`training_events_compact.jsonl/csv`、完整`training.log`和`probe.log`，以及指定的release `run.log`。未完成四row时直接拒绝，不读取部分训练或性能流。

它不打开`fit_trace.jsonl`、性能`compact.jsonl/csv`、support汇总、query、源域数据或checkpoint，不执行拟合或评分。完整训练事件已经包含每阶段的8步、最终目标和继承锚点，因此无需读取held score trace。

## 核验与输出

- 对账完整/紧凑JSONL、CSV及文本中的每条训练事件和阶段事件。
- 按row、split、scope、fold或anchor、state、trainK区分实际阶段，核对8步、9次目标、最终无反向、头拟合及adjoint三角求解的实际成本。
- 逐坐标复算Adam矩、梯度裁剪和更新提议，以KKT条件检查每块零和盒投影；核对参数继承和实际非零更新。普通零梯度阶段仍保留8步，K1跳过保持null目标。
- 监督目标只汇总`held_margin_loss_sum`；头训练的ridge损失独立标明，不混入内层监督目标。正确数、准确率及margin均按物理样本池化，属于训练指标，不能解释为outer-held改善。
- 全量扫描文本错误、警告、非有限、OOM、恢复、配置及环境标记。`early_stop_disabled_declaration`与`early_stop_actual_message`分列；配置中的`early_stopping:false`不代表早停。标记计数可能重叠，不自动认定故障。
- 原736维向量继续留在原始日志。输出只保留每块范数、和、极值、非零坐标数及条件数等摘要，覆盖全部8步和最终点。日志没有逐样本inner预测转换时，该指标为null，不能根据正确数相同推断预测相同。

本地新目录中生成`training_diagnostics.json`、`summary.json`、`stages.csv`、`stage_curves.csv`、`group_counts.csv`、`stratified_counts.csv`、`snapshots.csv`、`curves.csv`、`resource_groups.csv`及`report.md`。已有输出目录会被拒绝，原始日志不修改。

## 调用

由唯一执行owner在完整run结束后运行；本说明不授权干预或重启正在运行的实验。

```text
python tools/collect_d92_channel_training_diagnostics.py --run-root LOCAL_RUN_ROOT --run-log LOCAL_RELEASE_RUN_LOG --output NEW_LOCAL_DIRECTORY
```

也可经已有SSH配置将采集器源码送入远端Python stdin，仅返回JSON到本地：

```text
python tools/collect_d92_channel_training_diagnostics.py --ssh-host n607 --ssh-config E:/type10-7/tools/n607_ssh_config --remote-python REMOTE_PYTHON --run-root REMOTE_RUN_ROOT --run-log REMOTE_RELEASE_RUN_LOG --output NEW_LOCAL_DIRECTORY
```

远端命令不包含输出目录参数、不写文件；实际报告只在本地新目录落盘。`--run-log`应由执行owner核实，不依赖默认位置推断release日志路径。

验证：collector初轮8项中5项被metrics字典合并错误阻断；修复仅影响分析工具，未改运行算法。第二轮8项全部通过，证据为本run/evidence/pytest_utf8_1790757619871199200.stdout。完整实际扫描尚未执行，当前实验仍运行，不能从合成检查声称已有日志分析结果。
