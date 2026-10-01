# Ground A 支持集配对补充入口

本入口只顺序调用已冻结的 `score_d92_ground_a_support.score_support`，为显式声明的 Affine 与 Conditional 各 4 行创建独立补充目录。它不拟合、重建核、执行 SVD、修改方法或原始分析产物。A 仍由原生 float32 CosFace 小包逐样本面对全部 6 个原 head 列；现有 scorer 负责先持久化 A scores，再连接合法 support truth 与同源 B/C held 记录。Query 和源样本不进入该入口。

当前交付包含新 runner、publisher、对应合成测试与本文。未读取真实 packet/cache/trace/summary、权重或成绩；没有实际评分结论。真实 spec、预登记、串行测试、Git、发布和唯一 launch 均由 root 负责。

## API 与显式 spec

Runner CLI：`python tools/run_d92_ground_a_support.py --spec <relative-or-explicit-spec-path> --commit <actual-runtime-commit>`。Python API 为 `run(spec, commit, *, score_fn=score_support)`；默认调用现有 production scorer，测试可注入合成记录器。

| 层级 | 必填字段及口径 |
|---|---|
| 根 | `run_id`、`spec_path`、`code`、`execution`、`rows`。`spec_path` 是 release 内的相对 JSON 路径。 |
| code | `cwd` 与 `environment` 为明确绝对路径。无 native training code 依赖。 |
| execution | `remote_run_root` 绝对路径；`launch_owner=root`、`cpu_lanes=1`、`blas_threads_per_lane=2`。 |
| row 身份 | 唯一分析 `row_id`、原始 `source_run_id/source_row_id`；8 行恰好来自两个 source run，每个方法各 4 行，原 source row 不重复。 |
| row 输入 | `packet`、`support_features`、`fit_trace`、`source_summary` 全为显式绝对路径。Trace 文件名为 `fit_trace.jsonl`。 |
| row 来源 | `expected_summary_status`、`expected_summary_schema`、`expected_method`、`expected_runtime_commit`。`expected_summary_schema` 对应既有 summary 的 `schema`，不要求旧 Affine 未提供的 `summary_schema`。 |
| row 数据绑定 | `expected_checkpoint_sha256`、`expected_capsule_id`、`expected_model_seed`。不推断 model seed 或 packet 行序。 |
| row 输出 | `output_root` 严格等于新 run root 下的分析 row_id 目录；新 run 与输入目录、release 不重叠。 |

相同 source run 的 4 行必须声明相同 summary 路径、status/schema/method/runtime commit。调用 scorer 时使用原始 `source_run_id/source_row_id`，不能用新 supplement 的名称替换 B/C lineage；分析 row_id 仅组织新输出与 supervisor 记录。`score_arguments(row)` 返回完整实参；不额外提供 selection 覆盖值，scorer 继续使用实际 lane startup 的冻结 selection。

## 加载顺序与只读边界

任何 A 评分前，supervisor 一次性核对所有来源。每个 source run 的独立 summary 只读取一次，顶层白名单为 `status/scope/run_id/release_commit/coverage/old_class_count/query_rows_used/source_rows_used/schema/method`。`statistics`、资源分析和其他结果结构仅按 JSON 词法跳过，不反序列化、不作参数或输入选择。要求对应方法的 verified status、固定 support-only scope、6 个旧类、query/source 使用均为 0，且 `coverage.episodes=160`、`coverage.sequence_paths=1800`。

每个 lane 只从 `startup.json` 与 `probe_complete.json` 读取原始 5 项身份、schema/method/scope、输入权限、完成状态、episodes/sequence_paths、support_features 和 selection。Startup 的 config 只解码 `selection` 子项，其他算法字段词法跳过。Startup 与 completion 的 selection 必须相同，split ID 唯一，split 数等于 episodes；按既有 K1 一条 full-support 路径、其他 K 的 `min(K,3)+K` 条 OOF/proxy 路径核对实际计数。4 个 lane 的 episodes/path 总和必须对应该 source summary 的 160/1800。

既有 lane 没有 `runtime_commit` 字段；工具不发明该证据，也不要求修改历史产物。`expected_runtime_commit` 只绑定独立 summary 的 `release_commit`。实际 B/C 与物理 held、state archive、truth 等完整配对仍由已冻结 scorer 检查，supervisor 不重新拟合或重新计算方法。

来源核对失败时不调用任何 scorer。通过后按 spec 行序调用，每次只运行一行。Scorer 返回后，supervisor 独立白名单读回该行 `complete.json`，核对 schema/status、原始 binding、parent count、query=0、原 summary 未修改及 A prediction 已固定的标记；返回状态本身不能代替产物证据。未知或缺失字段直接报告原错误，不回退到其他行或方法。

## 日志、资源与失败保留

新 run 目录独占创建，保存实际 spec、PID、argv、runtime commit、Python/CPU/线程信息。`source_bindings.json` 只含来源元数据；`state.json` 保存各行实际状态。完整事件进入 stdout、`scoring.log`、`events.jsonl`；紧凑 JSONL/CSV 不带数组。每行开始记录实际函数名及完整实参，完成后记录既有 scorer 测量的资源与 wall time、可测 process peak RSS。没有伪造 CLI 子进程；当前入口是同一 supervisor 内的直接函数调用。

Scorer 的每行报告、统计表和固定 prediction 文件保持独立。Supervisor 只汇总完成行数、parent 数与输出引用，不重新混合准确率或将 K1 的 N/A 改成测量值。CPU 时间、文件/数组字节与部署常驻内存、传输字节和能耗分开；没有测量的 GPU peak、网络增量、能耗记 null。

首个技术失败即终止顺序队列，写 `failed.json`，保留此前完成输出及故障行已固定的 predictions。未开始行继续标为 PENDING，不自动重试，不停止或重启任何原始任务。已有新 run 根目录在读取任何来源前拒绝。成功才写 supervisor `complete.json`；原始方法及已运行 analysis 全部只读。

## 发布白名单与验证范围

Publisher CLI 为 `--spec`，显式恢复仍沿用既有 transport 的 `--resume-staged-commit`。远端 runtime 只包含 7 个文件：

- `code/cvsrffi/__init__.py`
- `code/cvsrffi/d92_ground_classifier_a.py`
- `tools/export_d92_ground_classifier_a_packet.py`
- `tools/score_d92_ground_a_support.py`
- `tools/export_d92_branch_support_features.py`
- `tools/cvs_native_artifacts.py`
- `tools/run_d92_ground_a_support.py`

Branch cache helper 仅提供既有 cache constants；它的直接 `cvs_native_artifacts` import 也纳入白名单。运行时不调用 helper 的原数据导出或 checkpoint 加载函数。Packet loader 只读已有 float32 bin，不需要 `baseline_origin_sat_view`、`muse_ssdg` 或原 encoder 代码。最小 bundle 隔离导入测试明确核对上述 7 个模块均来自临时目录，并确认 native/encoder 模块没有被导入。

Publisher 自身、既有 `publish_d92_branch_support_probe.py` 与 `run_d92_branch_support_probe.py` 是明确的本地 transport 依赖，纳入 committed/pushed archive。运行环境需要既有 Torch/NumPy；不安装或改动环境。旧 transport 保持原 committed/pushed 检查、archive/SCP、目标 release/run 不得存在和 sole supervisor 规则；只移除 GPU0 占用检查并替换 CPU supervisor 路径。隔离导入只证明当前解释器中的最小源码 import 链，不等同于远端评分已完成。

合成测试覆盖 8 行顺序与去重 summary 读取、结果字段词法跳过、全部来源预核对、错误身份/commit/count/selection 拒绝、独占输出、失败保留与不重试、完成标记独立读回、最小 bundle 隔离导入及 mock transport。另用既有合成 production trace 分别调用两个方法对应的真实 scorer，验证新分析行名不替换原 lineage，以及真实 K1 无独立 held A。Worker 仅执行 AST/UTF-8 静态检查，数值测试由 root 串行执行；此文不宣称实际 launch 或评分已完成。

## Root验证与当前状态

主Agent在项目ssr-gpu环境串行执行上述两个测试文件，36 passed，6.78 s；输出prefix为 `E:/type10-7/.codex_tmp/pytest_utf8_1790828105225733000`。原有Torch2.1/NumPy2 ABI警告仍在，当前路径采用已有list/scalar桥接并通过合成production scorer用例。直接P0/P1只读审查未发现问题。

实际8行配置 `configs/d92_ground_a_support_20261001.json` 已预登记并经同一纯spec validator核对。新run `20261001-phase2-d92-ground-a-support-m2-r01` 尚未启动；Conditional完整独立summary尚在运行，真实A与B−A保持N/A。来源核对在sole supervisor启动后、任何A评分前完成，不宣称Popen前读取了远端summary。
