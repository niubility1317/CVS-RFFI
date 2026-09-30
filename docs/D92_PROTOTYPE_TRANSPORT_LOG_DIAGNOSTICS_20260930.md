# PrototypeTransport完整训练日志诊断

状态：SYNTHETIC_VALIDATED_ACTUAL_LOGS_NOT_SCANNED。collector和测试仅处理确定性合成日志；未读取实际run、query、ABC、历史评分、registry或handoff，未拟合或评分。r02健康运行期间不热修改、不干预，正式pytest及后续完整采集由主Agent串行执行。

入口为`tools/collect_d92_transport_training_diagnostics.py`，仅依赖Python标准库，可本地读取完成run，也可将单文件通过SSH stdin发送给指定远端Python只读采集。远端命令不含`--output`；只有本地writer新建输出目录，已有目录拒绝覆盖。collector不import方法、NumPy或scoring入口，不启动任何训练。

先核对4 row完成状态、当前run release commit、冻结算法/矩阵/selection、capsule/checkpoint/model seed绑定及禁止query/source访问的元数据，然后完整读取每row的`fit_stages.jsonl`及CSV、`training_events.jsonl`、对应compact JSONL及CSV。所有structured记录逐行对照，完整扫描`training.log`、`probe.log`和指定`run.log`。不打开`fit_trace.jsonl`、parent `compact.jsonl`或任何query文件；文本里的SUPPORT_PARENT_COMPLETE仅识别事件标记，绝不反序列化held结果载荷。

训练身份使用row/split/scope/fold/outer_trial/state/train_k；事件的`trial`保留Armijo试探1/2/3，外层proxy anchor单独用`outer_trial`。共享准备、每个inner prototype物理ID、B参数anchor和C顺序继承分别核对。FINAL event含额外prepared envelope，collector逐字段核对其与candidate fit的共同字段，并要求完整final-state字段；不把不存在于candidate compact里的prepared顶层字段误判为丢失。

完整保留initial、每个accepted-cache gradient、全部接受和拒绝trial、最后接受cache。每个objective先跨不等内折累计每类半平方hinge，再计算class RMS；proximal按物理support总数缩放。独立核对梯度坐标/范数、beta box与eta零和box投影KKT、Armijo条件及冻结float64容差、step/trial关联、cache复用、四类有限solver stop和最终参数。head/factorization/forward/backward/triangular solve按实际日志加总，不把13/39/24上界写成实测，也不把有信息但零更新的阶段费用归零。

`information_stage`表示物理train K≥2且注册类数≥2；`updated_stage`表示至少接受一次更新。二者分开报告。零梯度information stage仍可产生initial objective、cached backward，C_seq继承非零adapter后即便无更新也可新增最终全类head。no-information的loss等未测量项为null/N/A，source validation和未记录的逐样本prediction transition也为N/A。

实际B fit只计一次，其C_seq/C_reset继承上下文保存于B_reuse_contexts，不能按三条路径重复收费。不同new-count parent可能重复拟合相同旧物理support：另按row、scope、parent K、train K、class IDs和有序训练physical IDs分组，保留全部实际context、重复次数及最终参数variant数。deduplicated_B_statistics仅取每组第一个训练代表并明确标注；常规统计和资源总量继续保留每次实际拟合，重复context不视为独立科学证据。

输出`training_diagnostics.json`、`summary.json`、`report.md`、`stages.csv`、`stage_curves.csv`、group/stratified counts、initial/final变化与完整trial曲线，以及资源和B复用/去重CSV。β/η参数与梯度逐坐标核对后输出范数、RMS、范围、零和残差及anchor距离；原完整参数保留在引用的source logs。cache梯度与final cache虽可携带此前forward_seconds，当前前向工作仅在head_fit_count非零时收费，避免重复计时。实际硬件、线程、RSS和payload仍来自run元数据；未记录的部署/传输或独立耗时不猜填。

文本错误/warning/NaN/OOM/recovery/solver stop标记按全部行统计，标记可能重叠且可来自配置声明。普通bounded stop不是技术故障，也不作为性能门槛；structured failure、非有限值及失败artifact独立报告。该工具不根据训练指标选择方法、预算或重新运行。

建议主Agent执行聚焦验证：

```text
conda run -n ssr-gpu python -m pytest -q tests/test_collect_d92_transport_training_diagnostics.py
```

8项测试使用手写冻结schema合成fixtures，不import任何model。覆盖完整事件/JSONL/CSV/text对应、B继承共享与跨parent去重、zero-update information阶段成本、拒绝trial全部保留、no-information N/A、Armijo/RMS/cache/stop/anchor篡改、不完整run不得打开训练流、输出拒绝覆盖和SSH stdin只读命令。子Agent只做语法/UTF-8检查。

主Agent实际串行执行该聚焦suite，8/8通过，耗时1.72 s；完整证据为`E:/type10-7/.codex_tmp/pytest_utf8_1790766451524824900.stdout`及`.stderr`。这是合成schema与I/O合同验证，实际完整训练日志尚未扫描，不能据此报告真实损失、solver分布或方法收益。
