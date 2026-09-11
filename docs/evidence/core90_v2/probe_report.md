# CORE90 V2探针、真实source负对照与追赶入口审查

**当前结论：经验lag测量和只读状态隔离已完成工程验证；选中恢复头在固定梯度参考上的方向证据为`TRANSFER_FAILURE`，正式校正可用性为`UNKNOWN`。不得将本报告解释为source机制已可晋级、全分布最优恢复或目标性能改善。**

本worker未启动正式训练、未加载checkpoint、未操作远端、未commit/push。以下真实source阶段回放从scratch开始，直接构造指定阶段的功能上下文；不是从E1完整训练到E131或E200。

## 五项真实source候选

仅读取`E:/type10-7/local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl`，构建器核对L/U/V=6300/56700/27000。360条参考覆盖6TX×5RX×3day的全部90容器，每容器4条；没有V拟合或target选择。固定scratch模型、E131功能上下文，五项初始CE完全相同`2.7215490341`，差值范围为0。

|lr|steps|状态|fit后CE|normalized gap|用途|
|---:|---:|---|---:|---:|---|
|.0002|40|BUDGET_INCONCLUSIVE|2.6493778229|None|候选|
|.0002|120|BUDGET_INCONCLUSIVE|2.4725322723|None|候选|
|.002|40|RELIABLE_HIGH_GAP|1.7564097643|.3563963038|候选|
|.002|120|RELIABLE_HIGH_GAP|.1285113692|.9575292066|按预定source规则选中|
|.02|40|RELIABLE_HIGH_GAP|.0123057012|1.0004403925|旧学习率参考，预先排除选择|

旧.02/40在此scratch固定样本上没有失稳，不能把“失稳参考”写成观察到了失稳。每项保存完整每步轨迹与尺度。规则在拟合前写入：排除失败/未完成/不可靠，主候选中取同目标fit终点CE最低；不用monitor或target重选。

产物：[完整五候选](source_probe_development/development.json)、[预定规则](source_probe_development/selection_rule.json)、[90容器覆盖](source_probe_development/source_coverage.json)。

[冻结配置](source_probe_development/selected_probe_config.json)仅含`game_evidence_version=2/game_probe_lr=.002/game_probe_steps=120`三字段；[选择元数据](source_probe_development/selected_probe_metadata.json)单独保存。真实`parse_args --game_config_json`已读回`2/.002/120`。通用parser仍是40步，后续采用开发配置需要显式加载。

## 固定候选的方向迁移失败

原结果保留不覆盖。添加直接参考质量检查后，仅重放原选中`.002/120`，没有改seed、重fit90条或重新选候选来凑方向有效；fit后CE仍逐值相同`.1285113692`。

在90条梯度参考、18条当前U上下文上，原头参考CE=`2.7299330235`、恢复头CE=`4.5338282585`，恶化`1.8038952351`。这超过预冻结绝对数值容差`1e-6`，因此梯度`valid=False/status=TRANSFER_FAILURE/reason=reference_head_objective_worsened`；经验lag仍为`RELIABLE_HIGH_GAP`。

online/recovered对抗梯度cosine=`.1249379388`被保留但不作为可信控制方向；非对抗配对梯度最大误差仅`4.1723251e-7`。该真实E131 scratch teacher的U选中数为0，完整目标仍包含U熵项；不能声称此fixture验证了非零U伪标签CE。synthetic独立mask fixture覆盖全False→全True。

[固定候选guard判读](source_probe_development_guarded/development.json)总状态为`UNKNOWN`。它证明两类证据必须拆开，不支持正式校正可用性或晋级。

## 三支真实source状态负对照

`accept_core90_v2_source.py`在CPU/2threads分别执行ordinary、audit_only、no_action_controller三支，从相同scratch seed开始，在E1/40/41/79/80/130/131各执行1次真实CORE90更新。

14个配对步骤全部`first_difference=None`，独立读回`VERIFIED`。逐步exact比较模型参数/BN、parameter.grad、AdamW状态、EMA、prototype、scaler、solver、Python/NumPy/Torch CPU/CUDA RNG、独立satellite generator、完整StepContext缓存、样本ID和mask，没有失败后放宽容差。

**该负对照使用probe_steps=2、lr=.002、每容器2条，不是120步选中恢复头的完整state对照。**[真实source比较结果](source_negative_controls/acceptance.json)明确记录预算区别。若失败，脚本保存首个左右完整state；本轮没有差异，因此没有差异checkpoint。

## 运行时审计实现

`runtime_audit_v2.py:game_audit_v2`先以当前训练RNG在owned模型上预览actual context确定U mask，再将固定mask复制到代表参考。原ctx及全部调用者状态不变。lag通过pre-head hook捕获真实头RNG/模式/buffer，在完整main+sat批次重放；sat行CE权重为0但保留头Dropout/BN批次语义。

梯度分开记录TX CE、完整主视图labeled非对抗、完整labeled+satellite+U非对抗及完整当前训练目标；`adv_dom_logits`已包含GRL，不重复反号。每对照field记录8次显式backward。坏lag不进入恢复方向计算；可靠lag也必须通过上述90条参考质量检查。

## `head_catchup_v2`独立审查

审查当前`runtime.py:head_catchup_v2`，在本配置范围未发现P0/P1：owned模型以当前RNG和完整main+sat提取特征；live head保留训练态及2N批次，CE只监督main N行。legacy模式乘实际adv系数，separate模式头系数为1；features已detach，不错误加GRL反号。每步重放头RNG，外层恢复调用者RNG；非head AdamW矩/step不提交。

在已有AdamW状态的真实CORE90 synthetic CPU实例执行2次额外head step，非head参数/buffer/optimizer、RNG、ctx和模式差异数量为0，实际head步骤为2。[审查执行脚本](review_head_catchup.py)及[独立状态读回](review_head_catchup.json)保留证据。

边界：GameSolver清空了156个非head `.grad`缓存。因此该入口不是“所有非head对象逐位不动”；当前runtime主更新必定先zero_grad，未发现训练行为影响，不列P0/P1。首次本地审查脚本因repo父路径写错ImportError，修正脚本路径后实际执行通过，未修改生产实现。AMP/CUDA或其他架构仍不在此审查的数值验收范围内。

## 本轮验证命令与未完成项

全部使用`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8`及Windows原生环境。

- `-m pytest code/tests/test_game_tracking_evidence_v2.py code/tests/test_game_tracking_audit_control.py -q`：31项通过；objective_scale案例先因缺参数RED，再通过。
- `-m pytest code/tests/test_game_tracking_runtime_audit_v2.py -q`：最终4项通过，包含完整scope、同头配对、两种headscale等价和真实CE恶化禁止方向。transfer案例先因缺参考CE字段RED，再通过。
- candidate selector先缺模块RED，之后通过；failed低CE及旧lr参考不能替换有效主候选。
- 父Agent统一negative/coordinator的9项测试已通过；本worker未重复该批或父Agent的完整集成suite。
- `code/scripts/source_probe_v2_development.py --wisig-pkl ... --output local_artifacts/core90_v2/source_probe_development --device cpu --threads 2`：完成五候选。
- 同脚本`--replay-selected-from .../source_probe_development/development.json`：完成固定候选guard判读，原产物保留。
- `code/scripts/accept_core90_v2_source.py --wisig-pkl ... --output local_artifacts/core90_v2/source_negative_controls --device cpu --threads 2`：完成14个真实source逐步配对。

未完成：正式训练中的控制动作收益、多seed/target确认、120步selected probe全state三支回放、CUDA审计数值对照、在线optimizer状态副本诊断。不能从当前结果推导为通过。

## 初始Task A接口与单元证据（保留）

Scope: only `audit_evidence.py`, `source_audit.py`, and `test_game_tracking_evidence_v2.py`; parent owns runtime/data/controller/config integration. No formal training, remote action, commit, or push performed by this worker.

## API

- `make_evidence_v2(...)` builds independent lag/readout/gradient/capability fields; absent values remain `None`. `require_evidence_v2` rejects legacy v1 input.
- `fit_empirical_lag(online_head, features, labels, *, sample_weights, objective_scope, config=ProbeConfigV2(), fixed_head_state=None, source_role='train') -> AuditResult` measures a bounded weighted empirical head target. Features and labels detach internally, weights normalize after finite/nonnegative checks. Caller owns source permission, coverage, main-view provenance, head scale, and active-objective certification.
- `objective_scope='current_training_head_objective'` can be control-ready only when the numerical quality check passes and training stochastic semantics are fixed. `eval_clean_surrogate` is always diagnostic only.
- `fixed_head_state` accepts `cpu_rng`, optional `cuda_rng` (required for CUDA train-mode control readiness), and optional exact `module_training` name-to-boolean mapping. Each forward replays the same RNG and initial copied buffers. Mixed module train flags persist. This is a fixed stochastic realization, not expected future-mask loss.
- `ProbeConfigV2.objective_scale` reproduces the head target coefficient: differentiation uses scaled CE, trajectory records both unscaled `fit_ce` and `fit_objective`, and gradient norm is the scaled target norm. Zero scale returns `UNAVAILABLE` without optimizer steps. Nonnegative zero-weight satellite slots can remain in the complete head batch to preserve Dropout/BN batch semantics while receiving no direct CE weight.
- `cross_tx_readout(features, domain_labels, rx_labels, tx, groups, *, config, seed, standardize=False)` runs two swapped whole-TX folds. Both domain and RX get fresh linear/MLP heads; optional feature transform is fitted only on fit TX. Readout `fit_improvement_*` fields are fitting diagnostics, never empirical lag control inputs.
- `SourceAuditor.run` and `fit_probe` retain legacy v1 behavior for existing callers, explicitly labeled in docstrings/output schema. Parent runtime must choose v2 explicitly.

## Measurement and limitations

Every fit step plus endpoint records weighted CE, gradient norm, parameter norm, logit quantiles; fixed monitor checkpoints are observation only. Features record norm quantiles, including monitor-side norms. Selection is always the configured endpoint, never best monitor step. AdamW is cold-reset and only a copied head changes. Optimizer-state-copy diagnostics are not implemented.

High empirical gap certifies measured improvement, not a globally optimal recovery. Low gap requires completed minimum budget and stationary gradient or a fixed plateau criterion; zero improvement at an analytical optimum is allowed. Failures and incomplete/unsupported low-gap budgets emit nullable gaps. Cross-TX transfer failure leaves independently computed empirical lag untouched.

Defaults are explicitly `fixed_source_development_unvalidated`: lr=0.002, steps/min_steps=40, normalized high-gap threshold=0.1, gradient tolerance=1e-4, plateau tolerance=1e-5 over five updates, instability allowance=0.05+0.25*initial CE, transfer CE tolerance=0.01. These are implementation candidates, not calibrated scientific thresholds or measured optimal budgets.

No independent same-condition monitor is invented. The empirical estimate concerns supplied source-training features only. Whole-TX splitting does not itself prove physical acquisition independence; the caller's metadata contract remains necessary.

## Validation

RED: `C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 -m pytest code/tests/test_game_tracking_evidence_v2.py -q` failed during collection with missing `audit_evidence` before implementation.

GREEN: `C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 -m pytest code/tests/test_game_tracking_evidence_v2.py code/tests/test_game_tracking_audit_control.py -q` passed 30 tests. Coverage includes analytical zero-gap optimum; recoverable underfit; zero budget; nonfinite features; actual finite high-lr divergence; weighted CE parity; fixed Dropout and mixed module modes; BN/parameter/gradient/RNG isolation; actual fit improvement plus held-TX deterioration; monitor independence from recovered endpoint; two-fold coverage; explicit v1 rejection; existing legacy behavior.

Validation uses local synthetic CPU fixtures. It does not establish source numerical budget suitability, CUDA numerical parity, or control activation in the parent runtime.

Objective-scale follow-up RED: the new zero/inactive and half-scale gradient test failed with `ProbeConfigV2.__init__() got an unexpected keyword argument 'objective_scale'`. After implementation, the same combined command passed 31 tests with no warnings.
