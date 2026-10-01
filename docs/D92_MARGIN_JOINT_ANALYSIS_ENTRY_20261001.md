# MarginJoint 独立分析与完整训练诊断入口

本次新增独立分析发布入口和训练诊断 collector，不改变 Margin 数学、训练预算、任务矩阵或方法参数。这里只完成源码及合成测试；未运行真实分析、读取真实训练产物或形成性能结论。数值测试、提交、发布及唯一 launch owner 均由 root 负责。

## 分析入口

```text
python tools/analyze_d92_margin_joint_probe.py --spec <owner-spec-relative-path> --analysis-release <new-exclusive-release-name>
```

入口使用现有 spec，不生成或选择 QP 预算。`probe.qp_resources` 必须含 `max_transitions`、`max_factor_buffer_bytes` 两个显式正整数。它们是技术资源界限，factor buffer 上限不是进程内存上限。身份为 `d92_margin_joint_local_ridge_v1` / `D92-MarginJointLocalRidge-v1`。

运行前核对本地 HEAD 与 origin 当前分支一致、白名单源码和实际 spec/cohort config 已提交。远端只读预检要求全部声明 row 完成，动态物理覆盖与 spec 推导一致、`workload_complete=true`、QP 资源完全一致；完整 `startup.spec` 必须等于本次 owner spec，startup 与 complete 的 runtime commit 必须一致。它不从支持或 query 成绩选参数、配置或重跑对象。

随后只向新独占 release 传输 committed archive，不修改训练 release。解包拒绝越界、符号链接和硬链接。release 保存完整 `analysis_resolved_config.json`；独立 summary 通过下列接口读取该文件：

```text
python tools/summarize_d92_margin_joint_probe.py --spec <saved-resolved-config> --run-root <declared-run-root> --output <run-root>/results/support_summary
```

输出、release、本地证据目录和远端 archive 均不得预先存在。保存实际 PID、argv、分析 commit、runtime commit、QP 资源和 `analysis.log`。summary 非零退出时保留这些证据和已有产物，不写成功执行证据、不自动重试。

成功要求 summary 的 `status=COMPLETE_MARGIN_JOINT_PROBE_VERIFIED`、`summary_schema=d92_margin_joint_support_summary_v1`，以及方法、run、row 数、动态覆盖、algorithm、QP 资源和 runtime commit 全部匹配；query/source 使用量均为 0，未接入实际 A 时 `actual_A=null`。写入 `analysis_execution.json` 后，另一次只读请求同时读回 summary、执行证据和完整 resolved config；下载后的 summary 再与读回值一致，才报告 VERIFIED。退出码本身不是完成证据。

## 完整训练诊断

本地一次派生：

```text
python tools/collect_d92_margin_joint_training_diagnostics.py --summary-root <verified-summary-dir> --run-root <run-dir> --output <new-diagnostics-dir>
```

只读远端采用两个显式阶段，保存 snapshot 后再遍历数组：

```text
python tools/collect_d92_margin_joint_training_diagnostics.py --summary-root <remote-summary-dir> --run-root <remote-run-dir> --ssh-host <host> --ssh-config <ssh-config> --remote-python <python> --snapshot-output <new-local-snapshot.json>
python tools/collect_d92_margin_joint_training_diagnostics.py --snapshot <local-snapshot.json> --ssh-host <host> --ssh-config <ssh-config> --remote-python <python> --output <new-local-diagnostics-dir>
```

summary 只反序列化明确的身份、配置、资源与训练来源字段；跳过 statistics 结果表。collector 不打开 fit_trace、outer/query 结果、独立 held scorer truth 或原始样本。它读取完整训练事件及全部引用的合法训练 NPZ，包括训练内部交叉验证所需的 inner-held 状态。namespace 必须绑定当次 run/row 和支持训练 scope；任何原始 `probe_failed.json` 都阻止将该来源派生为 COMPLETE，原始失败不覆盖、不重跑。

snapshot 固定全部事件索引与数组引用。extract 再核对流顺序、完整 FINAL/梯度/trial/step 引用、所有数组的键、shape、dtype 和字节数，然后遍历全部曲线与数组，不抽样。输出 `summary.json`、`report.md` 及 stages、curves、preparations、archives、strata 的完整 JSONL/CSV。

诊断口径：

- 梯度直接使用 `g_Z`，不减去或添加 `Z`；proximal 梯度为 0。检查 0.5 硬球和实际投影位移的 Armijo 条件。
- 记录接受前后 RMSCE 目标与已有 loss 字段，同时从保存的 class CE sums/counts 计算全部 inner-held 观测的算术平均 CE。二者分别报告，不把“接受”解释为二者都下降；缺失 class CE 数据记 null。
- preparation 与 FINAL stage audit 是累计计费来源；曲线中重复展示的累计账不再相加。14 个 QP 工作项各有 forward/adjoint 家族，包括实际尝试/完成 factor、condition、RHS、transition、全约束扫描及谱检查，不沿用 Conditional 固定两 factor 或三次谱诊断公式。
- 一般实际工作按 SUM，`margin_qp_peak_factor_buffer_bytes` 和 `margin_qp_peak_explicit_solve_temporary_bytes` 按 MAX。任一适用记录缺失或 null 会使该项总数为 null，并保留未知记录数，不填 0；已知总数与独立 summary 的已知账核对。
- 账的范围为 preparations 加 FINAL stage audits，排除基线、分析器自身成本及重复曲线。未测设备、进程峰值、传输和部署包字节不从 numeric buffer 推断。训练 margin/松弛量与乘子诊断不证明 query 决策保留。

snapshot 请求采用 `ensure_ascii=False, allow_nan=False` JSON，UTF-8 → gzip（`mtime=0`）→ base64；远端只用 stdlib 解码为数据。大型 snapshot 不作为 Python 字典字面量解析。不删字段、不改变诊断数学；非法 gzip、非有限 JSON 或操作失败都直接失败且不自动重试。

## 依赖与合成验证

Analyzer 复用 Margin publisher 的实际源码闭包，另包含独立 Margin summary、Affine summary/evaluate/run/math helper、本入口及本文档。这些数学/档案 helper 的导入不等于采用旧方法的头或计费公式。root 负责将 summary 最终直接导入闭包加入 publisher，并进行隔离 import 验证；文件存在本身不证明正式 launch 就绪。

Collector 的直接运行依赖仅为 Python stdlib、NumPy 和 `tools/collect_d92_affine_joint_training_diagnostics.py` 的通用 JSON/NPZ/CSV primitives；stdin 传输内嵌这两个 collector 的已提交源码，不要求远端导入训练 core。计数 schema 以字面 tuple 固定，合成测试核对其与冻结 Margin core 的公开计数字段一致。

root 串行验证入口：

```text
python -s -m pytest tests/test_analyze_d92_margin_joint_probe.py tests/test_collect_d92_margin_joint_training_diagnostics.py
```

测试使用手工 metadata、真实生产 Margin callback 的纯合成 features，以及 mock Popen 的本地控制脚本；覆盖动态 row/覆盖、完整配置和身份替换、独占输出、失败保留、不重试、全训练档案遍历、CE/RMSCE 区别、SUM/MAX/unknown，以及 Unicode/深结构/大于 1 MiB 压缩 payload/损坏 gzip/非有限值。作者仅执行 AST 和 UTF-8 静态检查，数值验证结果由 root 另行记录。

root 已在 verified ssr-gpu 环境串行验证：analyzer 14 项通过；collector 初次 7 项通过、1 项 NumPy bool JSON 序列化失败，修复源头类型后相关全部 8 项通过。publisher/prepare 和独立数学检查已通过；独立 summary 全部 49 项通过。详见 [验证记录](D92_MARGIN_JOINT_PIPELINE_VALIDATION_20261001.json)。这些是正确性与分析能力验证，尚无真实 Margin 性能或设备成本。
