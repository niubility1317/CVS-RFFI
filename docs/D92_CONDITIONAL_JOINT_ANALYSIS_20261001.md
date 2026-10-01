# ConditionalJoint 独立支持集分析

日期：2026-10-01。实现状态：`IMPLEMENTED_SOURCE_FROZEN`。本 worker 未运行测试、Conda、Git、SSH、发布或实验；root 已完成 31 项合成测试。本文不提供新的实际性能证据。

## 文件与输入边界

独立分析入口为 `tools/summarize_d92_conditional_joint_probe.py`，发布包装器为 `tools/analyze_d92_conditional_joint_probe.py`。对应合成测试为 `tests/test_summarize_d92_conditional_joint_probe.py` 和 `tests/test_analyze_d92_conditional_joint_probe.py`。

summary 只接受已经完整结束的 ConditionalJoint 支持集归档。它读取实际 spec、运行/逐行完成标记、进程参数、来源契约和 cache 元数据、全部拟合轨迹、完整及紧凑训练事件、数值状态归档。它不加载 checkpoint、cache 数值特征文件、query、源域逐样本数据或实际地面 A 数据包。合法 support-held 标签仅在独立重建并确认预测固定后用于统计。

完整性从 spec 实际声明的 rows、selection 和 axes 推导。缺失 row、parent、stage、event、数组、末尾 marker、文件清单或 source/cache 绑定均拒绝分析；不从 partial 轨迹补齐。每次分析输出和发布目录均要求不存在，失败证据保留，不覆盖旧分析。

## 独立数学验证

summary 不调用 candidate 的 fit、forward、score 或 adjoint。既有 Affine 分析中的纯几何、数组证书和统计函数可复用；candidate 模块只经入口依赖载入，不能成为验证自身的数学路径。合成测试在生成真实 candidate fixture 后禁用这些 candidate 数学入口。

所有原始几何由五个 raw 分支独立恢复成实际归一化表示，再恢复冻结 DCT/tangent adapter、完整交互距离、old 固定带宽、tau/gamma 和 raw Gaussian 核。R0 没有自由截距，按当前 `fit_branch_local_ridge._solve` 的完整 train 均值中心化验证。其 `reference_kernel`、`reference_self`、`center_mean` 和 `center_grand` 从 raw 训练几何直接核对，不能借 Affine 自由截距的 gauge 等价改变 R0 函数。B/C 的部署原始距离也使用相同归一化 original 几何。

B 的系数、截距、中心化和完整伴随使用独立 Affine 方程。C 的冻结 actual B 按真实旧类列映射，在 C 的 train/held 原始点重新求值；再检查 `M_train`、`M_held`、`M_O`、`M_N`、`M_H`、`R_N`、全列 alpha/beta/v、actual old 物理点约束和完整预测。旧类代表压缩只能合并 original 完整 b/a 精确相同的点；正 tau 时还验证 adapted 点和全部 raw 行/cross 列的一致性。所有物理旧点、所有注册输出列都参与约束，不能仅检验代表或某一 q 均值。

正核 C 使用冻结数学 helper 的独立完整对称 KKT 残差证书。令 theta 为 `[beta; alpha; v]`，则

```text
Q = [[ A,   -B,    -1 ],
     [-B.T, D+I,   +1 ],
     [-1.T, +1.T,   0 ]]
Q theta = [0; R_N; 0]
score_H = M_H + E alpha - F beta + v
```

summary 采用 `certify_positive_kernel_head`，其 `positive_kernel_checked=False` 保持原值。该函数检查完整 KKT、系数约束、old residual、新点 stationarity 和 score 残差；它自身不运行谱检查，不宣称确认 PSD。summary 另由独立 raw Gaussian/精确等价几何、无损 old 压缩和归档的真实 projection/residual Cholesky 因子残差建立正核证据。它恢复 J/z/s、cN/cH、含 rank-one 常数项的 Kperp/Lperp、实际因子及 RHS，不增加每记录的独立完整谱 gate。

完整 C kernel VJP 调用数学 helper 的独立 Q 伴随 solve，右端为 `[-F.T G; E.T G; sum_rows G]`，Q 上游为 `-sym(lambda theta.T)`。独立核对 A/B/D/F/E 上游，constant adjoint `g_b`，projection 和 residual 两套伴随方程、原核 train/cross 上游、距离 reverse、adapter reverse。B 伴随也包含自由截距常数项。所有数组必须 finite/shape 正确；固定机器精度残差尺度，无 jitter、容差调参或 candidate forward 回验。

gamma 为 None 时，C 残差严格为零，预测等于 actual B prior。tau=0 采用精确特征等价核，不引入地板或梯度。new0 精确复用实际 B，不要求虚构 C head。上述退化均独立验证。

## 训练轨迹和来源绑定

目标为按真实物理点归并的 pooled class RMS CE；proximal loss 和 gradient 均为零，没有旧 `+Z` 项。summary 重建每个 inner fold 的 logits、CE、类累计数和 RMS，再用完整 CE 权重恢复梯度。

坐标 `U=anchor+Z W.T`、固定半径 0.5、4 次梯度和每次最多 12 次 trial 的预算按实际事件验证。trial 步长使用冻结序列，球投影后的实际 delta 必须满足实际 `g dot delta` Armijo；最终坐标必须等于最后接受状态，不选择训练 peak。初始、gradient、trial、step、final、preparation 事件逐项与归档 state refs 和实际 audit 对齐；FINAL 使用真实 `{mode,state_ref,audit}` envelope。

每个 state namespace 绑定当前 run、row、split、scope、fold/trial、parent K 和 train K。actual B 继承只能来自同一路径的 B final，核参数、物理 ID、类列映射、prior 数组和 adapter 均精确绑定。source-only scratch provenance、checkpoint 完整继承字段、cache/capsule 契约和末 marker/实际 argv 也纳入完整性验证。

## 输出 ABI 和统计口径

summary 的固定机器可读标识为：

| 字段 | 值 |
|---|---|
| `status` | `COMPLETE_CONDITIONAL_JOINT_PROBE_VERIFIED` |
| `summary_schema` | `d92_conditional_joint_support_summary_v1` |
| `schema` | `d92_conditional_joint_local_ridge_v1` |
| `method` | `D92-ConditionalJointLocalRidge-v1` |
| `scope` | `SUPPORT_ONLY_CONDITIONAL_JOINT_OOF_AND_PROXY_NOT_QUERY_EVALUATION` |
| paths | `R0`、`R_CONDITIONAL_seq` |

主要字段为 `coverage`、`resources`、`resource_statistics`、`statistics`、`state_archives`、`training_stage_count`、`raw_training_sources`。每项 raw training source 给出 `row_id`、`fit_trace`、`full_training_events` 和 `compact_training_events`。`training_stage_count == coverage.candidate_stage_count`，PREPARED 总数等于 candidate preparation 数。report/collector 可直接消费这些显式归档入口。

statistics 包含 `overall`、`by_k_new_count`、`by_receiver_scene`、`by_model_cohort` 四张表；使用既有指标名和基础统计列（包括 `measured_parent_count`）。保留完整 K×new 表及 receiver/scenario、model/cohort 分层。OOF 每个物理 held 点只计一次；proxy 先在每 parent 内平均全部 anchor，再等权平均 parent。H、差距和旧类下降先在 parent 内计算，再平均；不能对总体平均准确率重新求 H。

真实 K1 没有独立物理 holdout，OOF/proxy 指标保持 null/N/A，只保留完整训练 head 和资源记录。其他 K 的 one-shot proxy 单独标识，不把它当真实 K1 独立 held。`actual_A`、`adaptation_gain_B_minus_A` 均为 null，直到真实合法地面数据包绑定；B0/R0 不能替代 A。无自动晋级、性能门槛或性能反馈调参。

输出为 `summary.json`、四张统计 CSV、分层资源 CSV、`training_objectives.jsonl/csv` 和完整表格 `report.md`。分析包执行后另保留 `analysis.log`、`analysis_process.json`、`analysis_execution.json`、本地 stdout/stderr 和远端读回证据。

## 成本与资源证据

资源由实际每个 archive 的系统阶数和 RHS 宽度累计。正核 C forward 包括 projection/residual 两个 factor；完整 CE adjoint 包括 projection_adjoint/residual_adjoint 两套 RHS 与四次 triangular solve。实际三次谱诊断单独计数；rejected trials、inner priors、完整 final 和外层 prior/residual inference 均计入。不能使用 Affine 单 factor 的固定公式，也不能把结构预算当耗时。

summary 的残差 forward 证书不新增 KKT solve 或谱检查；完整 VJP 验证另付 general indefinite solve 成本，阶数为 m+p+1，不能伪称省算力。独立 B/C general solve、RHS columns/elements、dense cubic/RHS work proxy 在 `independent_analysis_work` 中单列，与 candidate 工作量区分。这些 proxy 不是测量 FLOPs。独立分析谱诊断数为 0。

实际 wall/head/forward/backward/inference 时间、CPU/BLAS 环境、process peak RSS、数值 buffer、常驻状态、最小部署状态、归档压缩/数值字节分别保留。NPZ 拷贝不能恢复原 alias 布局，不能拿 archive bytes 充当常驻内存。GPU 峰值、能耗、部署包和新增传输未测量时写 null/N/A；三次谱诊断耗时包含在实际 head 时间内，未单独计时。

## 发布依赖和执行方式

新增入口的实际传递依赖必须随白名单发布，尤其包括：

- `tools/summarize_d92_affine_joint_probe.py`
- `tools/evaluate_d92_affine_joint_probe.py`
- `tools/d92_affine_analysis_math.py`
- `tools/d92_conditional_analysis_math.py`
- `tools/evaluate_d92_conditional_joint_probe.py`
- `tools/run_d92_conditional_joint_probe.py`
- `tools/publish_d92_conditional_joint_probe.py`
- `tools/publish_d92_branch_support_probe.py`（analyzer 的 FLAGS 来源）
- Conditional core/kernel 及原白名单内的纯几何、registration/branch summary 与运行依赖

analyzer 还显式携带 `tools/run_d92_affine_joint_probe.py`，这是分析发布清单的一部分，并非新 summary 的直接 import。新 summary/analyzer、本文及 reporter/collector 的真实依赖由 publisher 白名单检查。仅在临时隔离 source bundle 中成功 import 相关模块才能证明这些 import 路径随包交付；本 worker 未执行该检查。

本地 summary CLI 为 `python tools/summarize_d92_conditional_joint_probe.py --spec <spec> --run-root <complete-run> --output <new-directory>`。analyzer CLI 为 `python tools/analyze_d92_conditional_joint_probe.py --spec <spec> --analysis-release <new-safe-name>`。analyzer 核验已推送源码、完整动态 row 证据和 exclusive 路径后，只发布并执行独立 summary，不启动训练。远端动作是否成功由 summary 与 execution 的独立读回确认，退出码本身不作为完成证据。

## 合成验证记录

测试用 synthetic raw 分支生成真实 candidate K3 训练、StateArchive 和完整事件，再执行独立 `verify_record` 主验证链。覆盖 prior、beta/constraint、actual-B adapter 继承、非零 `g_b`、遗留 `+Z`、proximal scalar、计数、物理 ID、缺失 FINAL 的篡改拒绝；并覆盖真实 K1、new0、零核、tau0、动态 rows/axes 和 COMPLETE/INCOMPLETE 不猜补。analyzer 测试覆盖动态完成/summary schema、路径安全、exclusive 输出、完整包依赖和保留进程证据。

root 首次运行 31 项 summary/analyzer 合成测试：28 passed、3 failed，15.44 s。失败定位为 raw/original 几何表示不一致（影响真实 trace 和旧 `+Z` 负测的前置 R0 检查），以及 incomplete fixture 使用非法 namespace。已按实际源码修复归一化 original 几何和 R0 中心项核对，并把 fixture 改成合法 JSON namespace；保留负测、score 比较和原容差。未修改 core、candidate、冻结 KKT helper 或原始数学设计。

root 修复后串行运行两份测试文件，结果为 **31 passed，26.33 s**。日志证据为 root 的 `.codex_tmp/pytest_utf8_1790821231973550500`（已结束）；首轮失败日志 `.codex_tmp/pytest_utf8_1790820898461554800` 保留。真实 candidate 完整 trace 在禁用 candidate fit/score/adjoint 的环境下通过独立核验，`+Z`、constant adjoint 等篡改负测通过。上述测试由 root 执行，本 worker 未自行运行。

最终数值验证状态：`ROOT_SYNTHETIC_TESTS_VERIFIED`。临时白名单 source bundle 的隔离 import 检查与 Git/发布由 root 另行执行，本文不把它们记录为已完成。没有读取或运行实际 Conditional 性能结果，没有新增实际 A、query 评分或实验 launch。
