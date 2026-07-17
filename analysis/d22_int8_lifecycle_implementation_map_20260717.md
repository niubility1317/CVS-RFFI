# D22地面int8锚+target support生命周期实现映射

日期：2026-07-17
任务边界：只读审查现有D21/M1–M6与int8 prototype工具；设计`SUPPORT_ONLY_NO_QUERY` screen；本轮不运行、不打开query、不产生prediction/score、不提交Git

## 1. 结论

现有组件已经覆盖D22约85%的安全与生命周期能力，可以实现一条很小的增量路线。无需复用M1–M5的query runner，也无需再写int8组件解析器、联合bundle校验器、新类注册器或逐query分类器。

唯一缺失的核心是：

1. 从共同封存的地面int8`center+rank3 domain offset+radius`组件导出不含样本的旧类先验统计；
2. 用target receiver的合法单一LEO_weak旧类support对地面中心进行逐类收缩；
3. 以class-balanced support folds将收缩候选与target-only基线作逐类非劣门；
4. 门通过后，把收缩后的旧类状态交给现有`register_new_classes()`完成真实新类注册；门失败则原子回退target-only状态。

推荐新增一个约150–220行的核心模块和一个薄runner。由于现有D21正式runner尚未把“已验证support context”暴露为公共对象，本轮不创建会复制大段私有安全逻辑的runner骨架；应先做一次小重构，再写D22 runner。这样比从M1–M6复制实验代码更小、更安全。

## 2. 最小可复用入口

|能力|现有文件与入口|D22用法|结论|
|---|---|---|---|
|共同封存ADV3B02+int8 bundle加载|`code/cvsrffi/phase1_adv3b02_deployment_bundle.py:832 load_formal_adv3b02_deployment_bundle()`|校验外部签名、detached seal、完整root allowlist、runtime/checkpoint parity、class-handle binding后，同时返回runtime与组件|直接复用，禁止另写宽松loader|
|int8中心|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py:1029 dequantized_center()`|得到6个旧类`160-D`地面中心，仅在内存临时FP32解量化|直接复用|
|int8 domain offset|同文件`:1033 reconstruct_domain()`|用`core+rank3 residual`逐域重建；不得持久化全量dense bank|直接复用|
|int8 radius|同文件`:1051 radius_for_domain()`|逐域得到6个旧类的离线P90 cosine-distance radius|直接复用|
|组件资源审计|同文件`:1060 resource_audit()`|继承真实numeric payload与reconstruction MAC，避免把NPZ文件大小误写成运行时状态|直接复用|
|目标support正式物化|`code/scripts/run_d21_support_only_lifecycle.py:681 run()`及其`_preopen_manifest/_require_target_joint_binding/_require_post_materialization_authority/_payload_rows/_old_reuse/_extract_z_id`|读取before/after enrollment-only support，验证旧support复用、唯一LEO_weak观测、密封receipt与同runtime绑定|复用，但应先抽公共context，避免D22继续导入私有函数|
|target-only旧类状态|`code/cvsrffi/stage2_prototype_lifecycle.py:689 fit_old_snapshot()`|生成D22-Z0基线、target empirical center/radius与support content hash|直接复用|
|真实新类注册|同文件`:1195 register_new_classes()`|append-only注册新类；复核旧support hash；锁定旧prototype/radius/score path；执行old→new intrusion、new radius与boundary support guard|直接复用|
|逐样本全注册类评分|同文件`:1458 score_one()`、`:1470 score_batch()`|最终正式阶段复用；screen阶段只在support folds上调用|直接复用|
|support-only候选门|`code/scripts/run_d21_support_only_lifecycle.py:342 _select_candidate()`、`:384 _evaluate_support_lifecycle()`|复用其“逐类accuracy与worst margin不得低于L0，再按floor/overall/margin/复杂度排序”的规则|逻辑复用；建议将门函数转为公共函数|
|严格无query输入schema|`local_artifacts/d21_m6_support_fold_lowrank/run_m6_support_fold_lowrank.py:54–120`|复用enrollment-only路径、manifest schema与成员全集exact allowlist、额外query/truth成员拒绝规则|证据结构复用，不复用其模型适配|
|query不可达receipt|同文件`:428–455`|复用显式false字段和observed input allowlist|直接复用字段集合|

### 不应复用的入口

- `local_artifacts/d21_m1_dual_state/`、`d21_m1b_soft_fusion/`、`d21_capsule_fast_adapt_dev_20260717/`及`d21_m5lite_norm_affine/`的runner都包含query读取或prediction/score执行面。其support loss可作机制参考，但不能成为D22 screen的依赖。
- `local_artifacts/m5_support_only_sparse_delta_k10_new5_20260717/run_m5.py`同样在后半段打开query与truth，不适合作为严格screen入口。
- M6的模型低秩适配、手工梯度与fold factor均与D22无关；仅复用fail-closed输入和审计结构。
- `phase1_int8_prototype_bundle.py`的v1 dense域×类组件已有更安全、更小的v2 center-lowrank-radius替代。D22正式路线应只接v2 joint bundle；历史v1只能走`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`特许分支且不得形成正式claim。

## 3. D22最小机制

### 3.1 不做target-support选域

D20几何审查已显示同类跨地面域方向高度相似，按少量target support选择某个地面域容易放大噪声。D22不把source receiver域当作target域Oracle，也不逐query选域。

对每个旧类`c`，先从密封组件得到：

\[
g_c=\operatorname{norm}(\text{center}_c),\qquad
p_{d,c}=\operatorname{norm}(\text{reconstruct\_domain}(d)_c).
\]

domain offset只用于估计地面先验不确定度：

\[
\delta_c=\operatorname{median}_d\left(1-p_{d,c}^{\top}g_c\right),\qquad
\bar r_c=\operatorname{median}_d r_{d,c},\qquad
u_c=\max(\bar r_c^2+\delta_c,\epsilon).
\]

这样同时使用center、offset和radius，但不会选一个看似最接近target的source receiver，也不会在query阶段遍历地面域。

### 3.2 target旧类收缩

由`fit_old_snapshot()`得到target-only旧类中心`s_c`与target radius`t_c`。预注册`kappa0=4`，与现有`LifecycleConfig.radius_shrink_offset=4`保持同一强度量级，不扩展新网格。令：

\[
q_c=\operatorname{clip}\left(\frac{\operatorname{median}_j u_j}{u_c},0.25,4\right),
\quad
\alpha_c=\frac{K}{K+\kappa_0q_c},
\]

\[
\hat p_c=\operatorname{norm}\left(\alpha_cs_c+(1-\alpha_c)g_c\right),
\quad
\hat r_c=\sqrt{\alpha_ct_c^2+(1-\alpha_c)\bar r_c^2}.
\]

`alpha`是target-support权重；地面先验越不稳定，`q_c`越小，target权重越高。所有量都只依赖密封Phase1聚合组件和当前target support。

首轮screen只有两个机制候选：

- `D22-Z0`：现有target-only `fit_old_snapshot()+register_new_classes()`；
- `D22-S1`：上述固定收缩旧状态+同一个`register_new_classes()`。

不扫描domain、alpha、beta或多套kappa。K=10用5个leave-two-rank class-balanced folds；每fold每类8训2验。S1必须在每个场景×fold满足：每个旧类accuracy不低于Z0、旧类worst margin不低于Z0；注册后还必须对全部old+new support满足每类accuracy、old floor、new floor与worst margin均不低于Z0。否则原子回退Z0并输出`NO_GO_SUPPORT_GATE`。

K=1无法做self-excluded fold，必须执行现有安全回退：radius/boundary关闭，D22-S1不得凭单条support晋升。K=5/K20沿用现有rank-held folds，不另定义query规则。

### 3.3 新类注册与防遗忘

地面bundle只绑定Phase1旧类，不能为新类制造“地面原型”。新类必须完全由目标接收机的真实LEO_weak support注册。D22将通过门的旧状态作为`PrototypeLifecycleState`，再调用现有`register_new_classes()`：

- 新类原型由target support的mean/medoid/robust-trim support CV选择；
- old prototype、old radius与old score columns注册后逐位锁定；
- 每个新类及全部新类组合都接受old-support intrusion guard；
- radius与稀疏boundary只有在all-registered support非劣时才激活。

现有`register_new_classes()`主要严格保护old路径；D22 runner仍需在外层增加“全部old+new support相对D22-Z0逐类非劣门”，避免只保护old却牺牲new fidelity。

## 4. 建议代码改动

### 4.1 先抽取公共verified support context

新增`code/cvsrffi/support_only_enrollment_context.py`，把D21 runner已经验证的以下逻辑移动为公共API：

```python
@dataclass(frozen=True)
class VerifiedSupportLifecycleContext:
    runtime: torch.jit.ScriptModule
    component: CenterLowRankPrototypeComponent
    formal_phase2_context: Mapping[str, Any]
    old_classes: tuple[str, ...]
    new_classes: tuple[str, ...]
    k_shot: int
    by_scenario: Mapping[str, VerifiedScenarioSupport]
    before_capsule_root_sha256: str
    before_receipt_sha256: str
    after_capsule_root_sha256: str
    after_receipt_sha256: str
    input_access_audit: Mapping[str, Any]

def load_verified_support_lifecycle_context(...) -> VerifiedSupportLifecycleContext:
    ...
```

该函数必须内部调用现有formal joint bundle loader和signed enrollment materializer，返回的`VerifiedScenarioSupport`只含`old_z_id/old_labels/new_z_id/new_labels`，不暴露raw路径、query入口或clean/cache构建参数。D21 runner改为调用它，行为不变；D22直接复用。

### 4.2 新增核心模块

新增`code/cvsrffi/stage2_int8_anchor_shrink_lifecycle.py`：

```python
@dataclass(frozen=True)
class GroundAnchorStatistics:
    center: np.ndarray          # [C_old,160], readonly
    median_radius: np.ndarray   # [C_old], readonly
    offset_dispersion: np.ndarray
    component_binding: Mapping[str, Any]

def derive_ground_anchor_statistics(
    component: CenterLowRankPrototypeComponent,
    expected_old_classes: Sequence[str],
) -> GroundAnchorStatistics: ...

def fit_shrunk_old_snapshot(
    target_only: PrototypeLifecycleState,
    old_support_z_id: np.ndarray,
    old_support_labels: Sequence[str],
    anchors: GroundAnchorStatistics,
    *, kappa0: float = 4.0,
) -> tuple[PrototypeLifecycleState, Mapping[str, Any]]: ...

def support_only_atomic_gate(
    baseline: PrototypeLifecycleState,
    candidate: PrototypeLifecycleState,
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
) -> Mapping[str, Any]: ...
```

`fit_shrunk_old_snapshot()`可调用`fit_old_snapshot()`得到合法target基线，再构造一个新的只读`PrototypeLifecycleState`。必须同步设置`old_prototype_snapshot`为收缩后的旧原型；不得修改Phase1组件；不得把解量化dense bank写入artifact。

### 4.3 薄runner

新增`code/scripts/run_d22_support_only_int8_lifecycle_screen.py`。CLI只接受：

- before/after enrollment-only root、seal、signed materialization policy及其外部固定hash；
- joint ADV3B02 bundle root、detached seal、signature envelope及完整外部固定hash；
- output和device。

runner流程固定为：`load_verified_support_lifecycle_context→derive anchors→Z0/S1 support folds→atomic gate→register new classes→all-support gate→write support-only evidence`。不得存在query、truth、scorer、prediction、role、quota或assignment参数。

输出只允许：

- `support_fold_log.jsonl`；
- `selector_lock.json`；
- `old_state_before_registration.json`；
- `state_after_registration.json`；
- `resource_audit.json`；
- `query_unreachable_proof.json`；
- `COMMIT.json`。

不得输出query token、prediction、score或任何解量化地面dense bank。

## 5. fail-closed输入schema

### 5.1 Phase1 joint bundle

必须由`load_formal_adv3b02_deployment_bundle()`验证：

- pinned Ed25519 authority、detached seal与signature envelope；
- exact root member allowlist；
- checkpoint lineage、runtime hash、component pre-sign root、class binding、parity receipt、generation/method/config/code lock及outer root；
- v2 component NPZ exact member allowlist；
- `component.class_registry == before registered old class handles`且顺序一致；
- `FEATURE_SCHEMA=ADV3B02:z_id:unit_l2:160:v1`、`RESIDUAL_RANK=3`；
- Phase2只读，`component_update_access=false`。

若走用户特许的历史组件screen，必须显式声明`mode=PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`，在打开support前固定checkpoint hash、组件hash及Phase1 TX→class-handle列映射；不得形成formal metric/matrix/deployment claim。正结果仍必须重建joint seal。

### 5.2 Target enrollment

before/after都必须是signed enrollment-only包，且：

- `schema/profile/registration_state`精确匹配；
- exact成员allowlist只含runtime/method lock/overlay provenance/3个LEO_weak support；
- 额外query/truth/scorer/apply-only/before-in-after路径、绝对路径、`..`、symlink/reparse逃逸全部拒绝；
- `receiver/seed/K/runtime/checkpoint lineage`一致；
- after registry必须是before old registry的严格append；
- before旧support与after旧support按物理root token、post-channel hash、标签与rank逐行一致；
- 每个物理样本只有一个LEO_weak观测，不跨3场景复用；
- NPZ运行时只打开IQ、class handle/index、唯一物理root及post-channel hash所需成员，不加载任何query成员。

### 5.3 显式不可达字段

`query_unreachable_proof.json`至少包含且全为false：

```text
query_access
query_iq_access
query_token_access
query_truth_opened
query_fit
query_calibration
query_selection
query_early_stop
query_rollback
query_candidate_ranking
truth_sidecar_access
phase2_query_role_oracle_access
phase2_query_true_batch_class_count_access
phase2_query_class_quota_access
phase2_query_batch_global_assignment
prediction_artifact_emitted
score_operation_available
```

并记录`observed_input_accesses == exact_input_allowlist`。manifest自报false不足以替代运行时访问审计。

## 6. 状态、MAC与Pareto预估

以当前v2组件`D=14、C_old=6、rank=3、P=160`及formal K=10/new5的11类状态估算。

### 6.1 Phase1地面组件

现有测试给出的精确numeric payload：

- direction：4278B；
- radius：96B；
- 合计：4374B；
- center-only reconstruction：0MAC；
- 单域全部6类重建：2880MAC；
- 全13个residual域一次性重建：37440MAC；
- persistent dense FP32 bank：0B。

D22在enrollment阶段一次性遍历offset并计算dispersion，额外旧类中心相似度约`14×6×160=13440MAC`；地面几何总计约50880MAC/场景，一次性完成，不进入query热路径。

### 6.2 Target生命周期状态

直接复用当前float32`PrototypeLifecycleState`，11类after状态的numeric payload约：

- 11类prototype：7040B；
- radius、active mask、support count：77B；
- 6类immutable old snapshot及radius/mask：3870B；
- 最多5个8维稀疏boundary：约300B；
- 合计约11287B，另加小量JSON metadata。

连同Phase1 numeric component约15661B，远低于256KiB。正式部署若新增target int8 head序列化，可把11类prototype+FP16 scale+radius/mask/count压到约1837B；与4374B地面组件合计约6211B，但这是后续需实现和验证的压缩口径，D22 screen不能先声称已有该产物。

### 6.3 每query head MAC

现有生命周期资源公式为：

\[
160C+2\sum_b|I_b|+5N_{radius}.
\]

11类、最多5个8维boundary、11类radius全开时约：

\[
1760+80+55=1895\ \text{MAC/query}.
\]

ground center/offset/radius已经折叠进旧类状态，query时不遍历14个地面域。相比K=10/new5单qKNN的`110×160=17600MAC/query`，head部分约降低9.3倍；相比其FP16逐support状态约35200B，D22当前float32生命周期numeric state约11287B，约降低67.9%。这些是结构估算，正式报告仍需测平均/P95延迟、峰值显存及完整bundle文件大小。

D22适配参数=0、epoch=0、optimizer step=0；只有support-fold中心、收缩和注册计算。

### 6.4 与新adapter资源上限的关系

D22-S1不是梯度adapter，不继承M5/M6诊断中的5epoch设置。若后续把D22旧类锚生命周期与模型adapter组合，必须采用当前统一上限：adapter参数不超过80000、训练不超过30epoch；若是sparse key-layer delta，optimizer step仍不超过50，优先使用SGD且momentum=0；checkpoint+int8组件+target状态+adapter delta总状态不超过256KiB。任何组合路线都必须重新报告参数、实际更新原参数量、epoch、optimizer step、FP16 patch、总状态与merge后新增MAC，不能因D22头部为0epoch而省略adapter审计。

## 7. 必需测试

新增：

1. `tests/test_stage2_int8_anchor_shrink_lifecycle.py`：center/offset/radius形状、有限值、class order、收缩权重范围、K=1回退、fold原子门、old snapshot注册后bitwise lock、candidate失败回退Z0；
2. `tests/test_run_d22_support_only_int8_lifecycle_screen.py`：CLI无query表面、额外query member负测、路径逃逸负测、联合bundle binding错配、before/after旧support不一致、observed allowlist、输出禁止prediction/score/dense bank；
3. 继续运行`tests/test_phase1_center_lowrank_prototype_bundle.py`、`tests/test_stage2_prototype_lifecycle.py`、`tests/test_run_d21_support_only_lifecycle.py`，证明D21行为未回归。

## 8. 实施顺序与停止条件

1. 抽取公共verified support context并让D21 tests保持全绿；
2. 实现纯NumPy的ground statistics与shrink函数，先跑单元测试；
3. 实现薄D22 runner和合成负测，但不打开真实support；
4. inventory确认唯一合法joint bundle及before/after enrollment材料后，才允许一次support-only screen；
5. 任一场景×fold出现旧类逐类accuracy或margin退化，或注册后new floor/H退化，即`NO_GO_SUPPORT_GATE`并停止；
6. 只有三场景support门全部通过，才允许重建正式method lock。query仍必须留到后续独立最终测试，不能用于回看或修改D22。

本轮没有创建runner骨架，也没有运行实验。原因不是组件不足，而是现有D21安全物化逻辑尚未形成公共context；直接骨架会复制私有安全函数并与正在进行的inventory结论产生漂移。上述两文件重构是当前最小且可维护的实现入口。
