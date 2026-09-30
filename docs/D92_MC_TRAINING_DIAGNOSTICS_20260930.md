# MC 完整训练机制诊断工具

日期：2026-09-30。状态：`IMPLEMENTED_SYNTHETIC_TESTS_VERIFIED`。本 agent 完成语法/UTF-8 检查；root 已串行执行全部 7 个聚焦合成测试，7/7 通过，耗时 0.99 s。只读远端执行和数据下载由 root 负责。当前健康 MC run 没有被停止、重启或热修改，本工具尚未对实际 run 做分析。

## 输入和边界

工具为 `tools/collect_d92_mc_training_diagnostics.py`。它只在独立摘要状态为 `COMPLETE_MC_RESIDUAL8_PROBE_VERIFIED`、完整固定 160 parent / 4576 阶段以及四 row 的训练来源齐全后派生诊断。输入是摘要目录的 `training_objectives.jsonl` 和本轮完整 `training_events.jsonl`，需要时只读同一 row 的 `state_arrays/*.npz`。

`summary.json` 只选择训练来源、状态、覆盖、算法、资源和归档元数据；外层统计字段直接跳过，不反序列化。工具从不打开 parent `fit_trace.jsonl`、held-result、query、ABC、registry、handoff 或历史结果；不拟合、不评分、不调参、不新增 gate、成员 hash 或数学审查。完整 source binding、teacher/投影/接受条件和逐坐标认证由现有独立 summarizer 完成，这里只消费已核验产物并检查派生所需的完整事件数量。

## 已实现的派生统计

- 按 B、C_seq、C_reset_init 及实际 train K 汇总全部阶段，区分有信息、有更新、零更新有信息与各停止原因；初始和最终 task、keep、proximal、总损失及内部训练准确率均保留。
- 保留每个 initial、gradient、accepted/rejected trial、accepted step 和 final_cached/no_information 位置，不抽样；accepted step 重复对应 trial 的目标展示，不能再次计费。
- 教师 q 给出每折均值/范围和固定区间分布。q>0 表示严格正 margin 的正确教师；q=0 包含错误或并列。q=0 且 score 唯一 winner 给出确定错误下界，其他 q=0 并列标为无法区分，精确错误数为 N/A，不猜 held truth 或解析物理 ID。
- 记录保持上限、余量、实际超额、guard_active、方向范数、球投影对实际位移的影响、Armijo/总目标非增/保持条件三个 pass flag 和全部拒因。工具不重复证明这些条件正确，也不把内部保持约束解释为外层旧类效果保证。
- total_vs_keep 直接使用记录的总梯度与保持梯度。task_vs_keep 使用 `g_task = g_total − g_keep − (theta_current − theta_anchor)/N`，N 为该阶段全部 outer-train 物理样本数。两种冲突分别报告，任一范数为零时 cosine 为 N/A。
- 完整 NPZ 只用于教师 q/score 分布、梯度点积以及 U/V norm、相对 anchor/DCT 的变化和球边界统计，不重新计算头、距离或优化器。数组不进入 JSON；原始引用继续保留。
- 实际资源使用独立 summary 的覆盖/成本/硬件/归档计数。阶段 fit 成本来自 MC_FIT；MC_FIT 在 held scoring 前产生，缺少阶段 score_seconds 时记 N/A，实际评分总量仍保留在已核验的 summary resources 中。
- 重复 B 按 row、scope、parent K、train K、类别和有序训练物理 ID 分组，单独提供去重描述统计；实际资源继续保留重复拟合，不能将多个 new-count 上下文解释为独立证据。

零更新有信息阶段仍包含初始前向/反向成本。物理 K1 的损失为 N/A。源验证、逐样本预测变化和未测量的部署/传输项均为 N/A。本工具只派生诊断，不选择候选或设计新方法。完整合法 support 的训练机制证据可用于后续机制分析；禁止 query 反馈，以及本轮完成前根据部分结果改变冻结候选。内部训练指标不是独立验证，不能据此宣称泛化收益。

## CLI

在完整产物所在机器本地执行：

```text
python tools/collect_d92_mc_training_diagnostics.py --summary-root <完整独立summary目录> --output <不存在的新诊断目录>
```

若下载后的 source 引用仍指向远端，可传 `--run-root <本地完整run目录>`，按 `<run-root>/<row_id>/probe` 映射本轮事件/NPZ。

沿用只读 stdin 路线，无需新发布/分析 launcher：

```text
python tools/collect_d92_mc_training_diagnostics.py --summary-root <远端完整summary目录> --run-root <远端run目录> --output <本地新目录> --ssh-host <已授权host> --ssh-config <现有config> --remote-python <具备NumPy的CVS-RFFI Python绝对路径>
```

该模式将 collector 源码送入指定 Python，只读远端已有产物并返回派生数据；远端不创建输出，不修改方法或任务。正常模式本地输出排他创建；已有目录不覆盖，无自动重试。

## 输出

`summary.json` 为紧凑的覆盖、机制分组、资源、归档和限制说明，不复制源事件或全部阶段曲线。完整派生记录分别保存在 `stages.jsonl`、`curves.jsonl`、`teachers.jsonl`；同名 CSV 便于阅读。`by_mode_train_k.csv` 汇总初末损失、停止/拒因、实际计数和阶段资源；`curve_statistics.csv` 聚合全部曲线位置；`B_binding_groups.csv` 描述重复绑定；`archive_by_phase.csv` 保留真实 NPZ 文件数和字节分层；`report.md` 给出损失表和解释边界。

原始大体积 NPZ 保留产物路径，不拼入摘要。归档字节属于训练日志成本，不是部署传输成本；嵌套 archive_seconds 不和 fit_seconds 再次相加。阶段时间累计表示工作量，不等于并行 run elapsed。

## 合成验证

新增 `tests/test_collect_d92_mc_training_diagnostics.py`，7 个测试覆盖：全部事件与拒绝 trial、零更新有信息阶段、教师 q=0 的错误/并列边界、精确扣除 proximal 后的两种梯度冲突、跳过外层字段反序列化、未完整时不打开事件/NPZ、尾部事件缺失、JSONL/CSV 行数与无损引用/排他输出，以及只读 SSH stdin 参数路由。测试只使用合成 NPZ/日志。

建议 root 仅执行一次：

```text
python -m pytest -q tests/test_collect_d92_mc_training_diagnostics.py
```

依赖为 Python 标准库与 NumPy，不导入 fitting/core/summarizer 数值审计函数。root 实际结果为 7 passed in 0.99s；证据：[stdout](E:/type10-7/.codex_tmp/pytest_utf8_1790774207012280500.stdout)、[stderr](E:/type10-7/.codex_tmp/pytest_utf8_1790774207012280500.stderr)。测试后仅更新本文，不新增指标或重复测试。完整实际 pilot 和独立 summary 核验前不执行真实分析。

主Agent聚焦验证：7 passed in 0.99s。完整stdout/stderr保留于本run/evidence/pytest_utf8_1790774207012280500。实际完整训练诊断仍待160配置全部完成；本次只交付分析工具，不改变已运行算法。
