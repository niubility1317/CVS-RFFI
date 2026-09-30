# AFFINE 分析阶段流封装修复

日期：2026-10-01。状态：`LOCAL_VERIFIED_R02_NOT_STARTED`。

主任务报告 AFFINE 训练已完成 160 parents，首次分析 r01 在 `Stage stream mismatch` 处退出。本次只检查实现与合成测试，未读取真实 run、分数、NPZ、query 或历史结果，也未执行数值测试、Git、SSH、发布或实验。

根因是阶段记录的顶层字段不一致，并非再次遗漏 AJLR 的顺序修复。[入口 `evaluate()`](../tools/evaluate_d92_affine_joint_probe.py) 写 `fit_stages.jsonl` 时，使用 `dict(compact_event(stage), schema=SCHEMA, method=METHOD, split_id=...)`；[汇总器](../tools/summarize_d92_affine_joint_probe.py) 的逐条比较只添加 `split_id`，遗漏 `schema` 和 `method`。当前汇总已按真实 `prepare B → fit B → prepare C → fit C` 顺序重建，但第一条记录的完整字典比较仍会失败。

原[合成测试](../tests/test_summarize_d92_affine_joint_probe.py)直接把内部 `probe_affine_joint()` callback 写盘，其自定义 callback 同样未添加入口的 `schema/method`，因此没有覆盖生产入口的封装边界。

最小修复将严格阶段流比较提取为 `verify_stage_stream`，重建与入口一致的 `schema/method/split_id`。全部内容与原始顺序仍逐条相等检查，未删除字段、重排日志或放宽比较。数学核验、缓存、VJP、核心、训练配置及既有产物均未修改。

新增回归通过真实 `evaluate()`、`probe_affine_joint()` 和核心 callback 生成并落盘合成阶段记录，只替换外部 cache 加载和 selection。它独立重建并核对新增类及 `new0` 两种路径，反向检查错 schema、错 method、错 split、缺少封装字段、乱序、缺失 stage 及内容变化均被拒绝。测试直接调用生产汇总使用的比较函数，避免再用手写理想日志替代入口行为。

两份修改的 Python 文件已使用 stdlib `ast.parse` 通过 UTF-8 静态语法检查。首次静态命令因 `cmd.exe` 的 `-c` 引号传递失败，没有执行载荷；改用允许的短 `powershell.exe` 调用后得到 `AST_OK`。没有导入 NumPy 或执行数值测试。待主任务串行执行 `tests/test_summarize_d92_affine_joint_probe.py`；本说明不宣称测试通过或远端分析成功。

本次修改路径仅为汇总器、其合成测试与本文。无需重新训练；原分析失败证据和训练产物继续保留。重新分析、版本交付及结果判断由主任务负责，没有新增性能或审批门槛。

ANALYSIS_REPAIR_READY/VERIFIED：Affine分析器严格阶段比较已补齐真实入口schema/method；B→C顺序、全部字段、数学核验和容差保持原样。真实evaluate→core写盘及篡改回归在内的11项相关测试通过（106.31s），其中1项新增、10项既有回归，累计109项不同检查。r01失败和完整训练产物保留，新release d92_affine_joint_analysis_20261001_r02尚未启动；提交推送核对版本后单次独立分析，不重跑训练，目标ACTIVE。

首次失败证据：`analysis_r01_failure_20261001.json`；原analysis source33070cd3a1c27f50c9df001d7d3e2ffc03d7d9e8、PID489447、handle44670已退出。未生成summary，原训练runtime81a226a8d1cef34ea817ada87070bd89912d027d不变。
