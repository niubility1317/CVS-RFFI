# D92 FCR8 完整训练诊断收集器

`tools/collect_d92_fcr8_training_diagnostics.py` 在固定 support-only run 完成、独立 summary 达到 `COMPLETE_FCR8_PROBE_VERIFIED` 后，派生完整训练诊断。当前 run 仍在训练时不读取其部分产物；由 launch owner 在完成核实后单次调用。

收集器只打开 summary 的白名单元数据、`training_objectives.jsonl`、每个 lane 的完整 `training_events.jsonl` 和其中引用的原始 `state_arrays/*.npz`。summary 中的外层分数值不反序列化；不打开 parent `fit_trace.jsonl`、query、历史结果或实验索引。它不拟合、评分、调整参数或复验优化器数学，也不增加 hash、receipt 或数据重验。

## 派生口径

- 完整消费 `FCR_INNER_PREPARED`、`FCR_INITIAL`、`FCR_GRADIENT`、`FCR_TRIAL`、`FCR_STEP` 和 `FCR_FINAL`。初始化、每次梯度、所有接受及拒绝的试探、接受步和最终缓存都保存到曲线中。独立 summary 已核实阶段及数学正确性，收集器只检查描述性曲线没有漏掉最终事件或事件位置。
- FCR 训练坐标为 `Z∈R^(736×r)`，`r` 为实际保留秩。`U=anchor_U+Z@W.T` 保留原始 `736×8` 坐标；`V` 是固定 DCT 字典。实际可训练参数为 `736r`，最大值为 5888；不报告 `V` 梯度或参数球投影。零秩空数组的 norm/count 为 0，min/max 为 N/A。
- task 梯度精确按 `g_Z-keep_g_Z-Z` 派生。近端项为 `0.5*||Z||²`，近端梯度是 `Z`，不除以物理样本数。分别给出 total/keep 与 task/keep 的点积、夹角和冲突；任一向量范数为 0 时夹角与冲突为 N/A。guard 方向变化和真实 `Z/U` 更新范数只使用归档数组。
- `H`、`W`、singular values、原始 `U/Z`、梯度和教师数组继续引用原 NPZ。谱、保留秩、rank threshold、dictionary RMS 和 whitening residual/tolerance 来自实际准备阶段；不重新 SVD 或重新检查白化。`rank_estimated=false` 时谱与白化保留 N/A，不能把归档占位零谱当作实测谱。
- pre-tangent displacement 的均方、RMS 和 reconstruction error 使用训练代码已测值。`function_coordinate_path_bound=0.5` 与接受步的 `Z` 路径长度描述的是当前 support 上 tangent 投影前的函数残差，不是 `U` 范数约束，也不是最终特征、角度或分类变化的直接度量。真实功能变化不能由 `U` norm 推断。
- 前向机制保留每个训练折的实测 block angle、tangent/κ、joint/adapted distance 相对变化、bandwidth、kernel norm/change、held score RMS 和 held winner change。分布摘要按已有 count 加权合并，不增加前向或 head fit。`tau0=0` 且非零 `U` 导致 adapted distance bypass 时，保持 `None` 及 `ZERO_BANDWIDTH_BYPASSES_ADAPTED_DISTANCE`；合法 `tau0=None` 同样保持 None，不替换为 0。
- 教师 q 的完整分箱、正 margin 正确数、zero-q 的错误下界和并列不确定数由合法训练归档派生。`q=0` 包含负 margin 或并列；没有 held truth 标签时不声称精确教师错误数。
- 内层 held 标签参与训练，`inner_training_accuracy`、margin 和 winner change 仅是监督训练诊断，不能称为独立验证或外层性能。source validation 和 query prediction change 保持 N/A。无信息与有信息零更新阶段分开，有信息零更新仍计实际前向和反向工作。
- run coverage 保留全部动态计数器。stage `counts` 计阶段工作，`all_event_counters` 保留最终事件完整计数负载。最终事件继承的 preparation-only counters 不重复计入各消费阶段；共享 C preparation 按原坐标 NPZ 引用去重，单列实际消费阶段数。实际 B 重复上下文继续计费，另按同一物理 support 绑定给出独立标注的 B 去重统计。
- 接受步重复展示其 trial objective；禁止把所有曲线行相加作为资源总量。资源沿用独立 summary 实测值。阶段耗时求和表示工作量，不等于并发 wall time；未测量的推理、传输、硬件或内存项不补造。

## 调用与输出

本地 NumPy 环境可运行：

```text
python tools/collect_d92_fcr8_training_diagnostics.py --summary-root COMPLETE_VERIFIED_SUMMARY --run-root COMPLETE_RUN_ROOT --output NEW_LOCAL_DIAGNOSTICS
```

由主 Agent 使用已授权普通账户时，接口与 MC 收集器一致：

```text
python tools/collect_d92_fcr8_training_diagnostics.py --summary-root REMOTE_COMPLETE_VERIFIED_SUMMARY --run-root REMOTE_COMPLETE_RUN_ROOT --output NEW_LOCAL_DIAGNOSTICS --ssh-host HOST --ssh-config CONFIG --remote-python NUMPY_PYTHON
```

SSH 仅经 stdin 发送独立脚本，以远端 `--collect-stdout` 只读返回 JSON；只在本地新目录写诊断。`--collect-stdout` 禁止 SSH 嵌套和输出写入。输出目录必须不存在，原始训练产物不复制或覆盖。

输出包括 `summary.json`、`stages.jsonl/csv`、`curves.jsonl/csv`、`teachers.jsonl/csv`、`preparations.jsonl/csv`、`by_mode_train_k.csv`、`curve_statistics.csv`、`B_binding_groups.csv`、`archive_by_phase.csv` 和 `report.md`。JSON 的缺失值为 null，CSV 为 N/A；完整向量仍在原 NPZ 中。

## 验证范围

`tests/test_collect_d92_fcr8_training_diagnostics.py` 只构造临时合成训练产物。覆盖动态秩与空数组、完整接受/拒绝/零更新曲线、动态 stage/preparation/run 计数、共享 C 准备与重复 B 绑定、task/keep/近端分解、零带宽 N/A、元数据跳过外层分数、缺尾部拒绝、独占输出以及只读 SSH stdin 命令形状。

执行命令由主 Agent 在项目 `ssr-gpu` 环境串行运行：

```text
python -m pytest tests/test_collect_d92_fcr8_training_diagnostics.py -q
```

实现子 Agent 仅做本地 AST 和 UTF-8 检查，不运行 Conda、pytest、SSH、真实数据拟合/评分或 Git 操作。真实诊断收集仍需完整 independently verified summary；这份文档不声明当前 run 已完成。

主Agent已实际执行：8/8 passed，1.13s。证据 `automation_reports/CV-SincNet/20260930-phase2-d92-fcr8-support-m2-r01/evidence/pytest_utf8_1790783379165414900.stdout`。没有采集当前部分run；55个不同核心/入口/汇总/调度/collector检查通过，合成验证不证明性能。
