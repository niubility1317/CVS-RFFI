# MarginJoint query benchmark 的监督与发布入口

本入口为冻结的 `D92-MarginJointLocalRidge-v1` 生成 A/B/C query 预测，随后由 root 单独调用独立 scorer。监督器不调用 scorer，不连接 truth，不从成绩筛选 row，也不继承 support probe 的模型状态。方法、QP 误差规则、4×12、CE-only、硬球和原核参数保持[冻结 core](../code/cvsrffi/d92_margin_joint_local_ridge.py)的字面配置。资源范围仍按[资源边界文档](D92_MARGIN_JOINT_RESOURCE_BOUND_20261001.md)解释，不将 factor 字节代理当 RSS 或星载可行性。

本次只新增[监督器](../tools/run_d92_margin_joint_benchmark.py)、[发布器](../tools/publish_d92_margin_joint_benchmark.py)及两个合成测试文件。没有生成真实 spec/config、发布 release、启动实验或读取真实 packet/cache/checkpoint/成绩。worker 不运行数值测试，数值验证、Git、远端发布与预登记由 root 串行完成。合成契约 fixture 的 archive 文件只模拟引用结构，不冒充真实数值 head；真实 candidate 生成的预测由 predictor 自己的合成测试验证。

## 1. 最小显式 spec

schema 为 `d92_margin_joint_query_benchmark_v1`，所有字段由本次 root 登记，不提供真实路径、seed 或 QP 额度默认值。

| 字段 | 内容 |
|---|---|
| `run_id/group_id/spec_path` | 新 run、稳定 group、Git 内相对 JSON 路径 |
| `code` | `cwd` 新 remote release、`environment` 显式 Python、`commit` 冻结 preparation parent OID |
| `execution` | `remote_run_root` 新 run；`launch_owner='root'`；`cpu_lanes=2`；`blas_threads=2` |
| `benchmark.config` | exact `{'algorithm': core.FROZEN_CONFIG, 'qp_resources': {'max_transitions': int, 'max_factor_buffer_bytes': int}}`；两个值必须显式正整数，bool 不合格 |
| `benchmark.cohorts` | exact `rx3/rx1`；每项 `capsule/truth/expected_capsule_id`；truth 路径仅独立 scorer 访问 |
| `rows` | 每行 exact `row_id/cohort/expected_model_seed/expected_checkpoint_sha256/row_root/branch_features/ground_packet/output_root` |

`row_root` 是固定 cache loader 所用的原 Phase1 来源目录，供复用既有来源核对；不是新 run。`branch_features` 是同 checkpoint 的固定 received 五分支 cache，`ground_packet` 是 source-only 原 native 六旧类头 packet。监督器和 predictor 不加载 checkpoint 或 encoder。predictor preflight 使用已有 loader 的来源、capsule、cache 和 packet 绑定，不新建数据验证或 receipt 链。

所有声明 model 必须覆盖两个 cohort，禁止重复 model/cohort、漏行或为筛结果删行。同 model 的 checkpoint SHA 与 Ground packet 必须一致；不同 cohort 的 `row_root` 可以不同，每行原 prediction/cache 来源目录仍由 predictor/cache provenance 独立核对。不同 model 显式绑定各自 checkpoint。新 run、release、row output 不得与输入重叠，每行 `output_root` 固定为 `remote_run_root/row_id`，prediction 目录固定为其下 `predictions`。现行 source-only、checkpoint 继承和 `p2_min_v1/VALIDATED_ONCE` 边界继续适用，配置变化不触发数据 builder 重验。

spec 的 `code.commit` 可以早于包含新 spec 的发布 commit。实际 runtime OID 由 publisher 核对 HEAD 与远端分支，再作为 supervisor `--commit` 和 predictor `--release-commit` 传入。startup/complete 首次写明实际 OID，不用 preparation parent 代替，不允许 null 或事后补写。

## 2. 全 row preflight 后预测

监督器先创建新 run，保存 `resolved_algorithm.json/startup.json/state.json`。每 row 使用[预测器](../tools/evaluate_d92_margin_joint_benchmark.py)的相同显式参数执行 `--preflight-only`：

```text
--run-id --row-id --release-commit --row-root --capsule --output
--config --expected-capsule-id --expected-checkpoint-sha256
--branch-features --ground-packet
```

preflight 返回 `MARGIN_QUERY_PREFLIGHT_COMPLETE`，不创建 prediction 目录、拟合或推断。监督器比较返回的实际 model seed、checkpoint、capsule、源路径、方法和权限，并保存完整 split 的 support old/new IDs、query IDs、registry 与 K 元数据。同 cohort 的 model rows 必须对应同一物理 split 表。所有 row 都取得 preflight 决定后，才开始任何预测。一个 row 失败不触发重试，其他通过的 row 继续。

两个 CPU lane 的子进程固定 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=2`、`CUDA_VISIBLE_DEVICES=''`。每 row 记录实际 argv、PID、环境、`preflight.log` 和 `prediction.log`。预测器自己保留完整结构化训练事件、紧凑 JSONL/CSV、详细文本日志和真实状态数组；监督器不替其伪造算法或成本计数。

每 split 的 A 是原 Ground native 六旧类头，B 仅适应该 split 的合法 old support，C 继承该次实际 B 后使用全部 registered support；new0 原样复用 B。三个流面对同一 query IDs，逐物理记录执行，`technical_query_chunk_size=1`。query 无监督 fitting、role/truth、配额、类别数反馈或跨样本重排；source 样本和 source per-record 特征不参与适应。

监督器独立只读 completion verifier 不调用 candidate fit/forward/adjoint。它比较 preflight 与 predictor 首次 startup/complete，核对全部 split/query 物理坐标、A 原列序、B/C 的实际注册列、有限完整 scores 和既有 argmax tie policy；检查 actual B→C state ref、namespace、new0 同 ref，以及 archive manifest 的 COMPLETE 状态。`predictions_A/B/C.jsonl` 必须全部覆盖同一声明 query 坐标，`predictions.jsonl` 必须逐字节等于 C 文件。exit code=0 不能替代这些证据；矛盾的进程返回与 completion 保留为失败，交 root reconcile。

## 3. 全局 metadata ABI 与 truth-last

`startup.json` 的 schema 为 spec schema，status 为 `MARGIN_QUERY_SUPERVISOR_STARTED`，包含 `run_id/group_id/runtime_commit/code_commit/resolved_spec/pid/argv/python/environment/rows`。初始 `rows` 按 row ID 映射到全部声明身份和 PENDING 状态；`state.json/events.jsonl` 保留之后的实际状态。

`complete.json` 使用相同 schema，status 为 `MARGIN_QUERY_BENCHMARK_PREDICTIONS_COMPLETE` 或 `MARGIN_QUERY_BENCHMARK_PREDICTIONS_FAILED`，保留完整 `resolved_spec`、两种 commit、`row_count/completed_row_count` 和 `rows` 映射。每完成 row 的 status 为 `COMPLETE`，含 `release_commit/expected_model_seed/expected_checkpoint_sha256/expected_capsule_id/source_paths/output_root/prediction_output/predictions_complete_path/pid/argv/environment/preflight/marker`。失败 row 保留失败阶段、错误、日志与 partial output，不冒充完整工作量。

只有全部声明 rows 独立核对通过，`all_predictions_fixed` 才为 true。全局首次写明 `truth_read=False/scorer_invoked=False/automatic_retry=False`。独立[scorer](../tools/score_d92_margin_joint_benchmark.py)先再核对全部 rows 的三流、completion 和 actual commit，之后统一连接合法 truth。其 CLI 为 `--spec --output [--run-root]`；root 可显式映射 remote run 到本地镜像，不能从目录名猜行或混用旧 run。独立[report](../tools/report_d92_margin_joint_benchmark.py)使用 `--score --report --interpretation [--baseline-score]`，只读取 root 显式指定的完成结果，不搜索历史索引，也不反馈研发。

监督入口本身仅完成 prediction：

```text
python tools/run_d92_margin_joint_benchmark.py --spec <new-spec.json> --commit <actual-publisher-HEAD>
```

本入口不会因低分停机、重跑或选 peak；scorer 与 report 的命令必须由 root 后续单独执行。

## 4. 独占 source 发布与只读恢复

发布前由 root 完成相关本地数值验证、显式 stage/commit/push。publisher 检查当前 HEAD 的 40 位 OID、当前 branch 在远端 origin 的同一 OID、精确白名单与 spec 的 clean 状态；同时核对 preparation parent 是祖先，runtime 白名单源码从该 parent 到 HEAD 未变化。spec 可以在之后提交，方法源码不能借此悄悄变化。

`RUNTIME_PATHS` 是从 predictor/core 的真实 imports 展开的明确闭包：Margin head/core 及纯数学依赖、Ground head/packet loader、固定 received feature loaders、StateArchive 的已有纯入口依赖，以及新 run/evaluate/score/report。既有 `run_d92_registration_diagnostic` 在 import 时读取 `configs/d92_branch_local_ridge_frozen_20260929.json`，该固定源码常量也明确列入闭包；它不是本次 run 配置，不覆盖 Margin 算法。没有整个 `code` 目录、native encoder/checkpoint-loading 模块、真实 packet/数据、全局索引或既有结果。`PATHS` 另加本 publisher、本文、scorer 文档和本次显式 spec。完整列表以 publisher 源码为准，不按整个仓库扫描补包。

本地 isolated source import 仅复制 runtime 白名单，用 `-I -s` 隔离解释器并禁用 `np.load/torch.load`，检查项目模块确实来自临时 bundle，未加载 native encoder。root 测试运行此检查；worker 不调用数字环境。正式发布用 Git archive 创建新独占本地 archive 和新 remote release，已有本地 archive、remote archive、release 或 run 均拒绝覆盖。

remote 先核对传输 archive，再检查所有 tar 成员类型、路径、重复文件与精确白名单，拒绝 symlink/hardlink/device/path traversal，安全解包到新 release。之后使用声明 Python 再作 isolated source import，保留 stdout/stderr。source import 只证明依赖闭包能 import，不证明数值、实际输入或性能。它通过后，唯一 root owner 启动一次 supervisor，并保存 `launch.json` 与日志。

publisher 通过独立第二次只读 SSH 核对 actual PID、commit、resolved spec 和 supervisor startup，才返回 `VERIFIED`。这仅是发布和启动证据，不是预测完成证据。首次 startup 最多等候 30 秒；没有读回时报告 `UNKNOWN`，不是依据 SSH 返回或 Popen PID 宣称完成。失败 archive/release/log/state 全部保留，不自动重复传输、解包或 launch。

```text
python tools/publish_d92_margin_joint_benchmark.py --spec <Git-relative-new-spec.json>
python tools/publish_d92_margin_joint_benchmark.py --spec <same-spec.json> --reconcile
```

`--reconcile` 只读 archive 存在与传输 hash、launch、startup、state、complete 和 PID 存活证据，不调用 scorer，不访问 truth/成绩，不停止 healthy row。它不自动恢复或重试任何 mutation。是否需要新 run 或修复由 root 在确认失败层后处理，健康任务继续。

## 5. 本次静态与合成验证范围

[监督器合成测试](../tests/test_run_d92_margin_joint_benchmark.py)覆盖完整显式 config、bool 额度拒绝、输入输出分离、源模型绑定、同 model 跨 cohort 使用不同 `row_root`、每行 preflight 来源路径错配拒绝、跨 cohort checkpoint/packet 错配拒绝、全 row preflight 顺序、失败 row 隔离、不可覆盖、新0、完整物理三流、C alias、继承 namespace、预测固定性以及零 exit 缺 marker。结构 fixture 不含真实 feature、标签或权重。

[publisher 合成测试](../tests/test_publish_d92_margin_joint_benchmark.py)覆盖 pushed OID/clean 白名单、parent/runtime OID 区分、安全 tar 和 CPU 环境、只读 reconcile、独立 post-state、UNKNOWN 保留及不重试；Git/SSH 均用 fake calls。另有一次精确 source bundle 的 isolated import test，由 root 串行运行，不执行模型、拟合、query 或 truth。

worker 完成四个 Python 文件的 AST/严格 UTF-8 检查及本文的本地链接检查。随后 root 在实际激活的 ssr-gpu 环境完成新 query 流程的 99 项合成测试，包含 isolated import 和真实核心算法集成，见[验证记录](D92_MARGIN_JOINT_QUERY_VALIDATION_20261001.json)。尚未启动真实 query benchmark；没有 query 成绩或星载资源结论。
