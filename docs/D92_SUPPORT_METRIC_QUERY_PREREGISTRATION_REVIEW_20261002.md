# SupportMetric 复用 query 基准预登记审查

结论：`NO_UNRESOLVED_P0_P1`。本次发现一处 publisher 的直接 P1；owner 已最小修正，审查者已源码读回闭合。其余授权范围未发现会改变科学输入、提前读取 truth、混用旧适应状态、对子集评分或错误发布的 P0/P1。此结论只针对当前源码和 CONFIG_ONLY 元数据，不表示实验已启动、预测已固定或性能已验证。

## 范围

只读下列新源码及本次预登记，不复审既有数学 core 或本 agent 编写的 query predictor：

- `tools/score_d92_support_metric_joint_benchmark.py`。
- `tools/run_d92_support_metric_joint_benchmark.py`。
- `tools/publish_d92_support_metric_joint_benchmark.py`。
- [本次配置](../configs/d92_support_metric_joint_repeat_20261002.json)。
- [experiment.json](../automation_reports/CV-SincNet/20261002-phase2-d92-support-metric-joint-repeat-m2-r01/experiment.json)、[report.md](../automation_reports/CV-SincNet/20261002-phase2-d92-support-metric-joint-repeat-m2-r01/report.md) 和同目录 `events.jsonl`。

没有读取真实 features、weights、packet、query truth、预测、状态档案、评分、历史报告、索引或日志；没有执行测试、Git、SSH、Conda、发布或评分。源码中描述的数值读回是未来 scorer 的行为，本次审查未运行该行为。

检查时记录为 `CONFIG_ONLY`，`results=null`、`actual_runtime_commit=null`、`publication_status=NOT_LAUNCHED`。events 只声明冻结候选和矩阵、实现准备、未 launch 或访问 score。准备 parent commit 为 `19ad08a90d610feeccd3ecc42990e0db919a3edb`，实际 runtime OID 由 publisher 后续确定，不能把二者默认视为相等。

## 全矩阵后才连接 truth

scorer 的调用顺序满足当前 truth-last 边界：

1. `validate_declared_matrix`（第 1196 行）限定四个不同 row、两个固定 model seed 与 rx3/rx1 的完整交叉；rx3 每模型 900、rx1 每模型 300，合计 2400。完整 receiver、三个场景、五个 support seed、K=`1/5/10/20` 与新增=`0/2/5/10/20` 均固定。
2. `load_fixed_predictions`（第 972 行）先验证 startup/complete 的当前 run、row、capsule、checkpoint、model、release、算法及资源；拒绝失败产物、缺流、缺 split、错误类列或访问声明。五流为 A/B/C/R0_B/R0_C，兼容 `predictions.jsonl` 必须逐记录等于 C。
3. `_validate_archive`（第 503 行）核实际 NPZ 文件、路径范围、dtype/shape/nbytes/finite、manifest 和 phase 的文件/数值字节总量。`_training_ledger`（第 852 行）进一步核所有实际 preparation/stage 与其数值档案、B→C 和一次 row basis 绑定，以及四阶段 SUM/MAX 工作账。
4. `score_benchmark`（第 1265 行）在读取任何 truth 前，要求全局成功 marker、四行全部 COMPLETE、declared/completed episode 均 2400、全部 fixed flags 及当前 runtime 绑定；逐行完整验证后再次独立重读所有固定流和档案，并重读全局 startup/complete。
5. 第 1324 行才第一次 `read(cohort.truth)`，然后连接已固定预测。任一先前步骤失败都不能落入 truth join，也不对子集评分。

`validate_row_output`（第 1258 行）只返回零 truth 访问的完整 row marker。supervisor（第 172 至 175 行）调用这一个独立验证接口，不另写简化版完成检查，不调用 scorer 的 truth join。

## 注册类、实际 B 和 basis

`load_capsule_metadata`（第 385 行）只取 manifest、split 声明与 `received.ids`。新类数来自 `registered_classes` 的长度减去六旧类；old/new support 由合法 support labels 和该注册表分组。它拒绝 `query_labels/query_truth/query_roles/query_class_counts`，不读取 IQ 或 query 标签，也不从真实 query 角色或配额推断注册类。

`_stream`（第 561 行）验证每个 opaque query 的完整类列、分数和固定 prediction，五流 ID 集合及顺序相同。A 保留 native Ground 的原类序及 tie；B/R0 使用 canonical 类序。C 对全部注册类核验结构 decision（第 625 行）：组内 raw argmax、gate 与完整 log-probability gap，再按物理类 ID 处理真 tie；不用 rounded old argmax 替换公开 prediction。C 证书中的 old raw scores 必须逐值等于同 query 的实际 B scores。new0 的 C 记录及状态引用必须精确复用 B；R0 也复用其自身 B。

`_candidate_numeric`（第 735 行）区分 core 的 `context=dict(coords,stage=B/C_seq)` 与 archive 的 `namespace=dict(coords,state=...)`，核当前物理支持集、注册列、标签和内层 rank-mod fold。C 的 inner prior 只绑定 old inner-train；theta/U 来自当前 B。所有 `actual_B_*` 数值数组必须与该 split B 的实际 final state 对应，不只信 marker 的继承声明。`_training_ledger` 还要求 C stage 的 `final_prior_ref` 和 C preparation 的 `full_prior_ref` 都等于该 split B 的 final ref。

row basis 使用独立的 `ROW_SUPPORT_METRIC_BASIS` namespace，只有 run/row、无 split/fold/trial。`verify_row_basis`（第 276 行）检查唯一 owner、精确 rank/列号、U 及 component enclosure、certificate 文件和数值字节、显式资源、construction audit 和真实一次构建账。每个 preparation/stage/final state 必须绑定同一 row owner，U 与实际 `U.T@U` 读回一致；prep 构建计数为 0，binding/Gram 各为 1。四阶段 row_basis/preparation/stage/score 的完整 WORK 字段按实际 SUM/MAX 累计。

以上是档案数值一致性和浮点子问题 diagnostic；scorer 明确不声称完整 head interval certificate，也不把未测 public predict/R0 内部操作量补成 0。此审查不为这些范围另加完整区间证明或数据重验门槛。

## 科学来源和执行配置

配置与 experiment 的 `code/benchmark` 精确一致；登记中四行的所有执行字段和 execution 基础字段均与配置一致，额外字段只是 method、六种 seed 角色及权限说明。两 model seed 为 `2026092701/2026092702`；相同 model 跨 cohort 的 checkpoint SHA、Ground packet、ground summary 和部署状态一致，row_root/branch_features 按 cohort 显式不同。split/data seed 为 `2026092705`、augmentation 为 `2026092707`、support 为五个预声明 seed，evaluation 为 null，不猜补。

checkpoint 段声明 source-only、scratch、final200、精确 source contract/SHA/native architecture 的既有核验，不加载历史 support-probe adapter/head。源码 `_source_identity`（第 431 行）限定当前 checkpoint/model、source role 精确匹配、epoch 的精确 int200、空继承链、freeze 前 target access 为 false；native A 与 raw five-branch feature contract 分别核对。Ground geometry 只用冻结 center，禁 source 样本、逐样本源特征、teacher target、residual 重建和 query。来源既有核验声明并未在本次重新读权重或数据验证。

新算法使用独立 U 坐标和实际 Gram+prediction Fisher，eta 从 1 开始、最多一次更新和 12 次折半；六固定资源为 `100/64/167772160/65536/65536/128`。配置没有参数网格，也没有从 support 诊断或任何 query 分数初始化、选模或重排。支持性 SFT 仅依赖当前合法 support；候选 B 从零开始，C 只继承该 split 当次实际 B。R0 的 independent C 与候选顺序 C 区分。

runner 第 78 至 151 行严格解析当前 spec、完整矩阵、资源和每行显式路径；`predict_arguments` 不传 truth。第 225 至 290 行独占新 run，每行 preflight 后进入 CPU 两 lane/BLAS 两线程的固定 predictor；失败保留所属 row，健康 row 继续，无重试，不干预既有健康任务。全局 COMPLETE 只表示所有预测及独立 row 校验关闭，仍未调用 scorer。

publisher 第 17 至 22 行显式 source whitelist 包含新 scorer、runner、predictor、SupportMetric basis/step/core 和 shared helpers；`release_paths` 还加入本次 spec。实际不可变 pushed runtime OID 与准备 parent 分开；只启动一个 supervisor。发布只写新 release/run，碰撞或未知 dispatch 先只读 reconcile，不重复 launch。声明的 source pack 包含正确 scorer/config，但本次未执行隔离 import、远端发布或实数据 preflight。

## 本次 P1 的发现与闭合

发现时：publisher `REMOTE` 自身没有导入 `getpass`，却在启动唯一 supervisor 的 `Popen` 之后调用 `getpass.getuser()`。实际发布会在写 `launch.json` 前发生 NameError，留下已启动进程而无法闭合 landing 身份；前一个 collision probe 的导入属于另一个 SSH 进程，不能提供该名称。这是直接 P1，已即时通知 root 和 control owner。

owner 的最小修正已 FREEZE，源码第 99 行现为 `import getpass,hashlib,json,os,subprocess,tarfile`；第 102 行在路径处理、解包和 `Popen` 前核对 `c['expected_remote_user']`。审查者已只读回该局部修正。状态为 `SOURCE_FIX_READBACK_VERIFIED`；owner 告知新增 fresh remote namespace/order 离线回归，数值执行和结果闭合仍由 root 负责，本审查未读取其测试产物。

两项非阻断登记文字已告知 root，不是执行选择器，也不新增 gate：

- experiment 的 `data.contract_ref/label_map_ref/tx_sets_ref`（检查时第 17/19/44 行）仍指 `probe.cohorts.*`；本次实际配置入口是 `benchmark.cohorts.*`。
- `checkpoint.feature_cache_producer_run_id/role`（检查时第 116 至 117 行）仍写 support-probe/support-only。实际四行 `branch_features` 明确指向 rx3/rx1 的 branch-ridge-repeat raw cache，供合法 support 拟合及只读 query 推理，不是旧适应状态。root 应同步 provenance 描述，不能将 support-only cache 文字冒充本次 query cache 来源。

本次是明确透明的数据复用，不能称为全新独立确认。真实输出、完整预测关闭、truth-last 评分、指标和运行成本均尚未由本审查验证。没有增加审批、receipt、签名链、参数选择或白名单外泛化门槛。
