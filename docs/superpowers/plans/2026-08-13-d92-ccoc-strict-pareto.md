# D92 CCOC Strict Pareto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`，先完成一个真实K10三场景truth-free G0；仅当机制、D42量子和资源全过时自动推进Hard9+K1。

**Architecture:** 新核心模块复用现有D92的group auto-shrinkage SPD FULL端点与非加权类均值，只从D81变换后support的逐类raw scatter中流式估计old/new两个off-block共识系数。probe负责单FULL生命周期和exact E0数值回退，E0D slim/query负责fit、query、state和receipt闭包；独立G0入口以隔离E0/CCOC技术执行验证部署量子，不把E0 reference计入candidate资源或fit库存。

**Tech Stack:** Python3.10、NumPy、scikit-learn LDA、现有D81/D92/D42模块、pytest、PowerShell7、Conda`ssr-gpu`、Git、N607普通账户。

## Global Constraints

- 协议固定`p2_min_v1/VALIDATED_ONCE`；不得重验数据，query及其view不得参与fit、update、selection、truth、role、quota、global reassignment、回退或停止。
- 候选ID=`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`，candidate=`d92_e0_full_cross_class_offblock_consensus`，registered D mode=`ccoc_full`。
- `x_ci`必须是与`Sigma_g^auto`同坐标的D81变换后288维support；块固定`160/96/32`。
- `S_c=(K-1)^-1 sum r_ci r_ci^T`只生成`Q_c=offblock(S_c)`、`u_c=Q_c/||Q_c||_F`；不得直接平均raw`S_c`作为协方差端点。
- 每个任务组分别用其全部类计算`rho_g=clip((||sum u_c||_F^2-m_g)/(m_g(m_g-1)),0,1)`；不设epsilon、阈值、温度、drop-class或扫描。
- 最终`Sigma_g*=rho_g Sigma_g^auto+(1-rho_g)blockdiag(Sigma_g^auto)`，`Sigma*=0.5Sigma_old*+0.5Sigma_new*`；真实Cholesky，禁止伪逆和jitter。
- 正常K>2路径actual FULL fit=1、dense solve=1、正式D42=1；BLOCK/LOO/Fisher/scan=0。K≤2精确D92 FULL alias。
- 数值失败exact E0_FULL_ONLY且真实记录candidate/ref库存；结构、registry、schema和seal错误继续抛出。
- row canonicalization=`lexicographic_float32_row_bytes_then_float64_reduce`；old/new组内label permutation等变、task swap不变。
- 流式内存不保留`C×288×288`；K10瞬时统计上界=`334336B`，query MAC/state与E0精确相同。
- G0固定`rx_7_7__seed_713106__k_10__new_5`三场景；至少一个rho严格位于`(0,1)`，state非E0，`max_j|Delta M_j|>=q>0`，wall P90≤150ms、paired ratio≤1.50、注册增量peak目标≤512KiB且硬门≤1MiB。query MAC与永久state仍须和E0精确相等。
- 不修改历史CSOAS、TCRA、TPCE、NewGuard、FloorBoost结果或矩阵；不运行scorer或读取truth完成G0。
- 所有代码测试先在`conda activate ssr-gpu`环境串行执行；项目改动必须Git提交；N607只能由一个专属runner以新run ID执行。

---

### Task 1: CCOC数学核心与D92端点暴露

**Files:**
- Create: `code/cvsrffi/stage2_d92_cross_class_offblock_consensus.py`
- Modify: `code/cvsrffi/stage2_d92_registration_balanced_covariance.py`
- Create: `tests/test_stage2_d92_cross_class_offblock_consensus.py`
- Modify: `tests/test_stage2_d92_registration_balanced_covariance.py`

**Interfaces:**
- Consumes: `build_registration_balanced_statistics(d42, transformed, targets, class_count, k_shot)`及`RegistrationBalancedStatistics`。
- Produces: `D92CCOCError`、`D92CCOCNumericalError`、`CrossClassOffblockConsensusStatistics`、`build_cross_class_offblock_consensus_statistics`、`compile_cross_class_offblock_consensus_affine`、`ccoc_inactive_receipt`。
- `RegistrationBalancedStatistics`新增只读`old_covariance: np.ndarray`和`new_covariance: np.ndarray`；现有`covariance`及audit语义不变。

- [ ] **Step 1: 写D92端点与手算rho的失败测试**

在`tests/test_stage2_d92_registration_balanced_covariance.py`断言stats的old/new端点shape均为`(288,288)`、均SPD，且`0.5*(old+new)`与现有`covariance`array-equal。新增核心测试，用K3、11类、288维fixture：每类canonical residual为`[-v,0,+v]`；相同off-block方向的组应得到literal`rho=1.0`，互相抵消/正交的组应clip为literal`0.0`。期望值不能调用生产rho helper计算。

```python
def test_ccoc_pairwise_consensus_has_literal_full_and_block_endpoints():
    stats = build_cross_class_offblock_consensus_statistics(
        fake_d42, rows, labels, class_count=11, k_shot=3
    )
    assert stats.old_rho == 1.0
    assert stats.new_rho == 0.0
    assert np.array_equal(
        stats.covariance,
        0.5 * stats.base.old_covariance
        + 0.5 * blockdiag(stats.base.new_covariance),
    )
```

- [ ] **Step 2: 运行RED**

Run:

```powershell
& 'F:\App\miniconda3\shell\condabin\conda-hook.ps1'; conda activate ssr-gpu; $env:PYTHONPATH='code'; python -m pytest -q tests/test_stage2_d92_cross_class_offblock_consensus.py tests/test_stage2_d92_registration_balanced_covariance.py
```

Expected: collection因`stage2_d92_cross_class_offblock_consensus`不存在而失败，或端点字段缺失而失败；不得以测试fixture错误作为RED。

- [ ] **Step 3: 最小实现端点与流式共识**

`RegistrationBalancedStatistics`保留现有字段并加入两个端点；builder直接保存已经计算的`old_covariance/new_covariance`。CCOC core按每类canonical float32 row bytes排序；float64均值、residual与scatter采用确定性顺序。每类执行两遍cross-block计算：第一遍只累加三个上三角块的Frobenius norm，第二遍归一后加到该组的三个upper accumulators；不得保存全部类的Q/u。

```python
@dataclass(frozen=True)
class CrossClassOffblockConsensusStatistics:
    base: RegistrationBalancedStatistics
    covariance: np.ndarray
    old_rho: float
    new_rho: float
    audit: dict[str, Any]

def build_cross_class_offblock_consensus_statistics(
    d42: Any,
    transformed: np.ndarray,
    targets: np.ndarray,
    *,
    class_count: int,
    k_shot: int,
) -> CrossClassOffblockConsensusStatistics:
    base = build_registration_balanced_statistics(
        d42,
        transformed,
        targets,
        class_count=class_count,
        k_shot=k_shot,
    )
    old_rho, old_audit = _stream_group_consensus(transformed, targets, range(d42.old_class_count), k_shot)
    new_rho, new_audit = _stream_group_consensus(transformed, targets, range(d42.old_class_count, class_count), k_shot)
    old_cov = _mix_full_and_blockdiag(base.old_covariance, old_rho)
    new_cov = _mix_full_and_blockdiag(base.new_covariance, new_rho)
    covariance = 0.5 * old_cov + 0.5 * new_cov
    _require_symmetric_positive_definite(covariance)
    return CrossClassOffblockConsensusStatistics(
        base=base,
        covariance=covariance,
        old_rho=old_rho,
        new_rho=new_rho,
        audit=_ccoc_statistics_audit(base, old_audit, new_audit, covariance),
    )

def compile_cross_class_offblock_consensus_affine(
    d42: Any,
    statistics: CrossClassOffblockConsensusStatistics,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    coefficients = np.linalg.solve(statistics.covariance, statistics.base.means.T).T
    intercept = -0.5 * np.sum(statistics.base.means * coefficients, axis=1)
    intercept -= np.log(float(statistics.base.means.shape[0]))
    return coefficients.astype(np.float32), intercept.astype(np.float32), _ccoc_compile_audit(statistics)
```

core active audit必须包含：`d92_ccoc_formula_revision='pairwise_cosine_v1'`、两个rho、组类数、offblock norm min/max、canonicalization字符串、对称性布尔、FULL端点复用、additional fit/block/loo/fisher/scan均0、dense solve=1、Cholesky字段、support MAC上界、`support_transient_bytes_upper_bound=334336`、persistent/query delta均0及七项query访问false。

- [ ] **Step 4: 添加数值、对称性与资源负测**

测试以下真实行为：任一类Q零范数抛`D92CCOCNumericalError`；nonfinite Q/rho和注入非SPD端点抛数值错误；row permutation得到bitwise equal covariance/affine/audit；old/new组内label permutation按逆置换后bitwise equal；交换old/new任务得到相同共享covariance；compile只执行一个`np.linalg.solve`语义且不调用BLOCK/LOO/Fisher；K10 receipt是`334336`而非数组`nbytes`猜测。

- [ ] **Step 5: 运行GREEN与邻接回归**

Run:

```powershell
& 'F:\App\miniconda3\shell\condabin\conda-hook.ps1'; conda activate ssr-gpu; $env:PYTHONPATH='code'; python -m pytest -q tests/test_stage2_d92_cross_class_offblock_consensus.py tests/test_stage2_d92_registration_balanced_covariance.py tests/test_stage2_d92_cauchy_scatter_oas.py
python -m py_compile code/cvsrffi/stage2_d92_cross_class_offblock_consensus.py code/cvsrffi/stage2_d92_registration_balanced_covariance.py
git diff --check
```

Expected: all PASS；现有D92/CSOAS行为无回归。

- [ ] **Step 6: 提交Task 1**

```powershell
git add code/cvsrffi/stage2_d92_cross_class_offblock_consensus.py code/cvsrffi/stage2_d92_registration_balanced_covariance.py tests/test_stage2_d92_cross_class_offblock_consensus.py tests/test_stage2_d92_registration_balanced_covariance.py
git commit -m "feat: add D92 CCOC covariance core"
```

### Task 2: Probe、Slim与Query生命周期接线

**Files:**
- Modify: `code/scripts/probe_d92_registration_balanced_covariance.py`
- Modify: `code/cvsrffi/stage2_d92_e0d_slim.py`
- Modify: `code/cvsrffi/stage2_d92_e0d_query_evaluation.py`
- Modify: `tests/test_probe_d92_registration_balanced_covariance.py`
- Modify: `tests/test_stage2_d92_e0d_slim.py`
- Modify: `tests/test_stage2_d92_e0d_query_evaluation.py`

**Interfaces:**
- Consumes Task 1公开API和core audit。
- Produces arm`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`、candidate`d92_e0_full_cross_class_offblock_consensus`、mode`ccoc_full`，以及query层`d92_e0d_ccoc_*`镜像receipt。
- 正常K>2的two-state total count=2、after actual count=1；core数值fallback的two-state total=3、after actual=2；codec whole-D42 retry必须按真实执行库存计数并标`G0_eligible=false`。
- `run_d92_e0d_query_evaluation`新增仅供G0调用的可选关键字`technical_support_receipt_sink: Callable[[Mapping[str, Any]], None] | None = None`；默认`None`时不产生额外工作或持久化字段，非`None`时只在registered after-state完成后回调最终D42 state与同一support的技术收据，不读取query或truth。

- [ ] **Step 1: 写未知mode、arm缺失与receipt缺失RED**

新增测试：probe以`registered_d_mode='ccoc_full'`不再报unknown；slim存在指定arm且`expected_total_component_fit_count(K10)=2`；query active receipt需完整传播两个rho、SPD、fit、资源与七项query false。篡改任一rho、`additional_covariance_fit_count`、canonicalization、solve count、state/MAC delta或query flag都必须拒绝。

```python
def test_ccoc_query_rejects_rho_or_fit_inventory_tamper(ccoc_result):
    ccoc_result.geometry_audit['final_covariance_audit']['d92_ccoc_old_rho'] = 1.5
    with pytest.raises(D92E0DQueryEvaluationError, match='CCOC'):
        _audit_d92_e0d_fit(ccoc_result, arm=ccoc_arm, scenario='leo_clear_weak', k_shot=10, old_count=6, class_count=11)
```

- [ ] **Step 2: 运行RED**

Run:

```powershell
& 'F:\App\miniconda3\shell\condabin\conda-hook.ps1'; conda activate ssr-gpu; $env:PYTHONPATH='code'; python -m pytest -q tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d92_e0d_slim.py tests/test_stage2_d92_e0d_query_evaluation.py -k ccoc
```

Expected: unknown mode、缺arm或缺receipt导致失败。

- [ ] **Step 3: 接入单次D81→D92 endpoint→CCOC FULL路径**

probe加入`ccoc_full`。K>2时只调用一次`translate_to_robust_centers`，把同一transformed rows送入Task 1 builder/compile；正常路径只追加一条FULL component record。`D92CCOCNumericalError`时追加一次失败attempt记录，再调用已有`full_fit`得到exact E0 reference；`D92CCOCError`和D92 registry错误包装为`D92ProbeError`继续抛。K≤2不调用CCOC builder，保持现有`d92_full_alias`。

```python
elif registered_d_mode == 'ccoc_full':
    selected_fit = ccoc_fit
    effective_d_mode = registered_d_mode
```

- [ ] **Step 4: 接入Slim严格receipt与库存**

新增arm spec；`expected_total_component_fit_count`把`ccoc_full`归入正常2。实现`_ccoc_receipt`：REG0 reason=`NOT_REGISTERED_STATE`，K≤2 reason=`K1_K2_EXACT_D92_FULL_ALIAS`且rho/量子均`None`；active要求两个rho finite且`0≤rho≤1`、candidate fit=1/ref=0、所有zero-access与资源/solve字段正确；numeric fallback要求active=false/fallback=true/ref fit=1且G0 false。

- [ ] **Step 5: 接入Query与codec边界**

Query增加`_CCOC_ARM_IDS`、字段白名单、独立数值关系复核和`d92_e0d_ccoc_*`映射。复核`rho`范围、两组类数、norm范围、Cholesky、fit/solve/scan、transient、persistent/query delta、query flags和state bytes/MAC。仅registered after-state int8 D42数值异常触发whole-D42 E0 retry；before-state或结构错误继续抛。codec fallback真实计数必须由实际调用产生，不复制固定假库存。可选`technical_support_receipt_sink`在正式D42 state产生后，以最终`log_diag`变换同一support，输出scene、canonical support/class handles、逐row跨组margin、三个block的`A_b`、最终state SHA及`scale1/scale2`；不得输出原始support feature，也不得改变fit、state或资源审计。

- [ ] **Step 6: 加K1/K2、fallback和真实D42非E0测试**

使用现有真实D42 codec fixture：K1/K2输出与D92 FULL alias state byte-exact；Q零范数导致exact E0 fallback；active合成support经D42后state与E0不同；结构性class registry漂移必须抛。测试还应证明对query feature/label内容的改变不会改变fit audit/state，只允许prediction改变。

- [ ] **Step 7: 运行GREEN与全邻接聚焦**

Run:

```powershell
& 'F:\App\miniconda3\shell\condabin\conda-hook.ps1'; conda activate ssr-gpu; $env:PYTHONPATH='code'; python -m pytest -q tests/test_stage2_d92_cross_class_offblock_consensus.py tests/test_stage2_d92_registration_balanced_covariance.py tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d92_e0d_slim.py tests/test_stage2_d92_e0d_query_evaluation.py
python -m py_compile code/scripts/probe_d92_registration_balanced_covariance.py code/cvsrffi/stage2_d92_e0d_slim.py code/cvsrffi/stage2_d92_e0d_query_evaluation.py
git diff --check
```

Expected: all PASS；CSOAS、TCRA、TPCE、NewGuard相邻测试不回归。

- [ ] **Step 8: 提交Task 2**

```powershell
git add code/scripts/probe_d92_registration_balanced_covariance.py code/cvsrffi/stage2_d92_e0d_slim.py code/cvsrffi/stage2_d92_e0d_query_evaluation.py tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d92_e0d_slim.py tests/test_stage2_d92_e0d_query_evaluation.py
git commit -m "feat: wire D92 CCOC through E0D"
```

### Task 3: G0技术收据、验证器与不可覆盖发布物

**Files:**
- Create: `code/cvsrffi/stage2_d92_ccoc_g0.py`
- Create: `code/scripts/run_d92_ccoc_g0.py`
- Create: `tests/test_stage2_d92_ccoc_g0.py`
- Create: `tests/test_run_d92_ccoc_g0.py`
- Create: `automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v1/report.md`
- Create: `automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v1/launch.sh`
- Create: `automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v1/DELIVERY_MANIFEST.txt`
- Modify: `code/SYNC_MANIFEST.txt`

**Interfaces:**
- Consumes sealed package arguments identical to`run_d92_e0d_prediction.py`，但只允许固定outer`rx_7_7__seed_713106__k_10__new_5`和arms`E0_FULL_ONLY`、`E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS`。
- Produces two immutable subroots`reference_e0/`与`candidate_ccoc/`及`g0_validation.json`；candidate资源独立测量，reference不计入candidate fit/wall/peak。
- `validate_ccoc_g0(reference_rows, candidate_rows)`按canonical support identity逐scene连接真实D42 state、support margins和scale，绝不读取truth/scorer。

- [ ] **Step 1: 写量子、身份和资源门RED**

用literal小fixture锁定：

```python
M = score_true - max_opposite_group_score
q = max_block(
    max_abs_support_coordinate_in_block
    * max(e0_scale1, e0_scale2, ccoc_scale1, ccoc_scale2)
)
```

验证`max_abs_delta_margin < q`、两个rho都在端点、state SHA相同、任一fallback、actual fit≠1、query flag true、wall/ratio/peak越界均拒绝；`max_abs_delta_margin == q`通过边界。reference/candidate canonical support SHA不相等必须拒绝。

- [ ] **Step 2: 运行RED**

Run:

```powershell
& 'F:\App\miniconda3\shell\condabin\conda-hook.ps1'; conda activate ssr-gpu; $env:PYTHONPATH='code'; python -m pytest -q tests/test_stage2_d92_ccoc_g0.py tests/test_run_d92_ccoc_g0.py
```

Expected: G0模块/CLI缺失而collection失败。

- [ ] **Step 3: 实现隔离双执行和support部署收据**

CLI只接受冻结sealed package路径、hash、ground路径/hash、两个新输出子root和device。它先执行E0 reference，再独立执行CCOC；两个执行必须使用同一support canonical identity和class registry。每个scene技术收据包含最终D42 coefficient/bias state SHA、`scale1/scale2`按块最大值、support块幅度、canonical row handle、逐row跨组margin和query访问布尔。验证器连接同handle后计算`max_j|Delta M_j|`与q；原始support feature不写盘。

```python
def validate_ccoc_g0(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    _require_same_support_identity(reference, candidate)
    margins = _paired_margin_delta_by_handle(reference, candidate)
    quantum = _maximum_cross_group_margin_quantum(reference, candidate)
    gates = _ccoc_g0_gates(reference, candidate, margins, quantum)
    return {
        "schema": "cvs.phase2.d92_ccoc.truth_free_g0_validation.v1",
        "max_cross_group_margin_change_abs": max(abs(value) for value in margins.values()),
        "cross_group_margin_quantum": quantum,
        "gates": gates,
        "pass": all(gates.values()),
    }
```

三个scene全部通过时marker=`D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS`；任何技术门失败非零退出且不运行scorer。

- [ ] **Step 4: 冻结report、launch与交付manifest**

report预登记：目标、唯一候选、科学commit、文件hash、固定outer/三scene、四包/seal/ground路径与SHA、普通N607环境、唯一命令、source/output/log路径、GPU0、expected artifacts、健康停止和`fresh_run_retry=false`。launch只做archive/hash/import closure、两个不可覆盖subroot的G0命令和marker验证；不得包含score/truth参数或读取accuracy/H/floor。

发布路径固定为：

```text
run_id=d92_e0_full_ccoc_g0_k10_20260813_v1
output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1
```

Task 3开始时先运行`git rev-parse --short=8 HEAD`得到实际8位科学commit，并把source固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_${scientific_commit8}_20260813_v1`；随后把展开后的绝对路径写入report、launch和DELIVERY_MANIFEST，发布物中不得保留环境变量或占位字符串。archive由同一Git HEAD精确封存，最终size/SHA/member count也必须写入三份发布物。

- [ ] **Step 5: 运行GREEN、CLI和交付静态检查**

Run:

```powershell
& 'F:\App\miniconda3\shell\condabin\conda-hook.ps1'; conda activate ssr-gpu; $env:PYTHONPATH='code'; python -m pytest -q tests/test_stage2_d92_ccoc_g0.py tests/test_run_d92_ccoc_g0.py
python -m py_compile code/cvsrffi/stage2_d92_ccoc_g0.py code/scripts/run_d92_ccoc_g0.py
python code/scripts/run_d92_ccoc_g0.py --help
git diff --check
```

另以系统tar核验archive无绝对路径、`..`或`code/code`，必需CCOC/E0D/entry文件齐全；以bash执行`bash -n launch.sh`。

- [ ] **Step 6: 提交Task 3**

```powershell
git add code/cvsrffi/stage2_d92_ccoc_g0.py code/scripts/run_d92_ccoc_g0.py tests/test_stage2_d92_ccoc_g0.py tests/test_run_d92_ccoc_g0.py automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v1 code/SYNC_MANIFEST.txt
git commit -m "chore: prepare D92 CCOC G0 release"
```

### Task 4: 独立审查、本地总验证与sole runner交接

**Files:**
- Modify: `analysis/d92_ccoc_traceability_20260813.md`
- Modify: `automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v1/report.md`
- Create after run: `automation_reports/CV-SincNet/d92_e0_full_ccoc_g0_k10_20260813_v1/runner_handoff.md`

**Interfaces:**
- Consumes Tasks 1–3 commits与冻结发布物。
- Produces independent`P0=0/P1=0`审查、完整本地验证记录，以及sole N607 runner handoff；primary不重复launch。

- [ ] **Step 1: 运行完整聚焦验证**

Run:

```powershell
& 'F:\App\miniconda3\shell\condabin\conda-hook.ps1'; conda activate ssr-gpu; $env:PYTHONPATH='code'; python -m pytest -q tests/test_stage2_d92_cross_class_offblock_consensus.py tests/test_stage2_d92_registration_balanced_covariance.py tests/test_probe_d92_registration_balanced_covariance.py tests/test_stage2_d92_e0d_slim.py tests/test_stage2_d92_e0d_query_evaluation.py tests/test_stage2_d92_ccoc_g0.py tests/test_run_d92_ccoc_g0.py
python -m py_compile code/cvsrffi/stage2_d92_cross_class_offblock_consensus.py code/cvsrffi/stage2_d92_registration_balanced_covariance.py code/scripts/probe_d92_registration_balanced_covariance.py code/cvsrffi/stage2_d92_e0d_slim.py code/cvsrffi/stage2_d92_e0d_query_evaluation.py code/cvsrffi/stage2_d92_ccoc_g0.py code/scripts/run_d92_ccoc_g0.py
git diff --check
git status -sb
```

Expected: all PASS，worktree clean。

- [ ] **Step 2: 独立P0/P1审查**

审查范围必须覆盖：公式与D92端点、canonical row/label/task对称性、numeric-vs-structural fallback、实际fit/D42库存、query/state/MAC、真实量子计算、G0双执行隔离、report/launch不可覆盖和无truth/scorer。只有`P0=0/P1=0`可发布；修复必须重新TDD和复审。

- [ ] **Step 3: 更新追溯与report为LOCAL_VERIFIED**

把CCOC-01至CCOC-12更新为实现/验证状态，记录commit、测试计数、review结论、archive/config/launch hash和精确local→remote mapping。不得预填G0结果。

- [ ] **Step 4: 交给唯一N607 runner**

runner只执行：普通账户direct preflight、远端三root ABSENT与GPU核验、顺序SCP及逐件SHA、唯一detached launch、PID/CWD/GPU/log健康检查、完整source/output/log取回、tree hash和SSH清理。不得改方法、阈值、矩阵、代码，不得重试同run，不得读取性能或运行scorer。

- [ ] **Step 5: 按G0结果自动裁决**

若marker和三scene全部硬门通过，更新report/trace并自动创建Hard9+K1机械计划；若任一scene fallback、rho端点全占、state E0、量子、fit/query/state或资源硬门失败，则标记该机制`REJECT_ROUTE`并停止CCOC，不把技术失败伪装为性能失败。纯发布工程故障最多以新run ID修复两轮。
