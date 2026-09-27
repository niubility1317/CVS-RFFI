## Global Constraints

- 基准提交：`acf0a6407c4a2cd114851e9f60d0cd346e4b2c77`。
- 不修改历史证据目录；在隔离工作区建立新研究执行包 `experiments/ir_eg_v1/`。
- 保留全部源RX、clean/LEO物理配对、L/U曝光、DAOT与FastTrust身份路径、原生阶段和loss归约。
- U域头CE不向身份骨干传直接对抗梯度；U身份任务照常反向。
- 目标数据不参与训练、响应、选参、收敛、重跑或选模。
- 第一阶段 `kappa=0.25`、E21起、每步响应、CG最多2次、`reuse_origin_used_scales`，其余原生配置冻结。
- `kappa=0`直接调用旧完整EG，不运行新增随机/缓存/求解代码。
- 只有正式接受主步提交一次参数/优化器/外部持久状态。
- E200/44400是预算审计读数，不是科学收敛标准；远端长训练默认不启动。

## 1. 共同接口

`ir_types.py`应定义下列接口，字段不得在下游临时改名：

```python
TensorTuple = tuple[torch.Tensor, ...]
GradientTuple = tuple[torch.Tensor | None, ...]

@dataclass(frozen=True)
class ParamLayout:
    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    phi_indices: tuple[int, ...]
    psi_indices: tuple[int, ...]
    aliases: dict[str, str]
    signature: str

@dataclass(frozen=True)
class DomainCall:
    key: str
    physical_ids: tuple[str, ...]
    z_detached: torch.Tensor
    domain: torch.Tensor
    sample_weight: torch.Tensor  # 已含原生归约，不再次mean
    scaled_dropout_mask: torch.Tensor
    grl_multiplier: float

@dataclass(frozen=True)
class HeadBatch:
    calls: tuple[DomainCall, ...]
    layout_signature: str
    context_signature: str

@dataclass(frozen=True)
class ResponseMetric:
    diagonal: torch.Tensor
    sqrt_diagonal: torch.Tensor
    active_indices: torch.Tensor
    predictor_clip_coefficient: float

@dataclass(frozen=True)
class CGResult:
    solution: torch.Tensor  # 白化坐标u；不是delta_phi
    status: str
    iterations: int
    relative_residual: float
    reason: str | None

class ObjectivePacket:
    # 非对抗task张量与原生post-GRL图保留；头求解使用HeadBatch的detached输入。
    head_batch: HeadBatch
    phi_predictor: TensorTuple
    def assemble(self, phi_leaf: TensorTuple) -> torch.Tensor: ...
    def release(self) -> None: ...
```

`ResponseMetric.diagonal/sqrt_diagonal`只包含响应活动坐标；`active_indices`索引完整phi扁平向量。GGN输入/输出使用完整phi坐标，求解器负责scatter/gather，非活动增量严格为0。`CGResult.solution`固定表示白化解u，下游只乘一次sqrt(R)得到delta_phi。

`DomainCall.z_detached`不得携带骨干图；post-GRL图只保存在ObjectivePacket内部，不与detached字段混用。主求解涉及的额外ctx状态由明确的outer transaction拥有，而非任意字典全局缓存。

## Task 1：冻结基线、参数归属和loss ledger

**Files:** Create `ir_types.py`, `tests/test_ir_contract.py`, `acceptance/baseline_manifest.json`；在新执行包建立导入来源检查。

**Consumes:** 冻结代码、原生joint配置、运行环境、合法source contract。

**Produces:** `build_layout(model, optimizer) -> ParamLayout`；`audit_contract(model, optimizer, ctx) -> dict`。

- [ ] 写失败测试 `test_adv_head_not_dom_head`、`test_shared_parameter_dedup`、`test_import_origin_is_pinned`、`test_task_has_no_phi_gradient`、`test_weighted_LU_reduction_matches_native`、`test_no_target_fields_in_context`。验证adv_head仅属于phi、其余活动参数属于psi，L/U不被总batch mean重写。
- [ ] 运行 `python -m pytest tests/test_ir_contract.py -q`。实现前预期因缺失接口或明确断言失败；环境缺失导致的错误不能冒充有效红灯测试。
- [ ] 实现参数对象去重与name/shape签名，登记两项实际域头损失和所有原生task项。保存head梯度来源、native有效系数和sample IDs；导入来源不匹配时raise。
- [ ] 重跑该测试，并运行冻结包已有基线验收。关闭IR前后同一次seed/ctx的loss、原始梯度、参数、optimizer、RNG和持久状态应一致。
- [ ] 保存机器可读验收证据并提交独立变更：`test: lock native joint contract for ir-eg`。

## Task 2：原生域头随机函数重放

**Files:** Create `ir_head.py`, `tests/test_ir_head.py`；Modify新包`model_dual_cvsincnet.py`的受控捕获接口，不改默认forward行为。

**Consumes:** ParamLayout、真实head调用序列与Dropout前RNG。

**Produces:** `capture_head_calls(...) -> HeadBatch`；`head_logits(phi, call) -> Tensor`；`head_gradient(phi, batch) -> Tensor`。

- [ ] 写 `test_replay_logits_and_gradients`、`test_zero_activation_does_not_hide_mask`、`test_L_U_separate_call_rng`、`test_aux_LEO_call_not_in_domain_loss`、`test_replay_restores_caller_rng`、`test_unsupported_head_rejected`。
- [ ] 运行 `python -m pytest tests/test_ir_head.py -q` 验证测试确实针对未实现行为失败。
- [ ] 在原生Dropout前记录RNG，隔离重建缩放mask。纯head使用非原地ReLU及固定mask；保持原生head参数命名。除同构MLP外，首版拒绝其他结构。
- [ ] FP64小模型以`atol=1e-10, rtol=1e-8`比较logits/输入梯度/参数梯度；FP32以`atol=1e-6, rtol=1e-5`比较。RNG及call ledger必须完全相同。
- [ ] 保存包括p=0、正常dropout概率、零激活、L/U不同batch大小的证据，提交：`feat: add deterministic adversarial-head replay`。

## Task 3：真实AdamW预测对应的局部度量

**Files:** Create `ir_metric.py`, `tests/test_ir_metric.py`；Modify `game_tracking/solvers.py`只增加可验收的clip信息出口。

**Consumes:** 原点optimizer state、h0、真实预测clip系数、预测lr、虚拟step结果、ParamLayout。

**Produces:** `build_response_metric(optimizer_origin, predictor_record, h0, layout) -> ResponseMetric`。

- [ ] 写 `test_first_adam_step_metric`、`test_nonzero_history_and_decay`、`test_parameter_local_step_counter`、`test_clip_coefficient_used_once`、`test_zero_lr_excluded`、`test_none_gradient_not_zero`、`test_unsupported_optimizer_flags`。
- [ ] 运行 `python -m pytest tests/test_ir_metric.py -q` 确认红灯。
- [ ] 按设计规范R公式实现；beta/eps/state必须取真实值。暴露clip系数的改动不能改变原始梯度归约顺序或已接受参数。weight decay不加入GGN。
- [ ] 对冻结二阶矩、固定clip的独立仿射AdamW公式作有限差分验证。不得对完整动态AdamW有限差分后要求与R相等。再验证真实预测二阶矩与记录相等。
- [ ] 运行旧EG零改动回归并提交：`feat: derive response metric from isolated AdamW predictor`。

## Task 4：加权GGN与固定2次CG

**Files:** Extend `ir_head.py`；Create `ir_cg.py`, `tests/test_ir_curvature.py`, `tests/test_ir_cg.py`。

**Consumes:** phip、HeadBatch、ResponseMetric。

**Produces:** `ggn_matvec(phi, batch, vector) -> Tensor`；`solve_response(e, metric, ggn_operator, *, max_iterations=2, rtol=1e-3) -> CGResult`。

- [ ] 写小型线性softmax头的显式GGN测试；验证matvec、对称性、PSD、L/U归约、权重非平方。增加`test_nonlinear_GGN_not_declared_full_Hessian`。
- [ ] 写 `test_CG_two_steps_truncated_is_usable`、`test_zero_rhs`、`test_zero_h0_nonzero_hp`、`test_nonfinite_fallback_status`、`test_G_zero_equals_minus_R_e`，运行 `python -m pytest tests/test_ir_curvature.py tests/test_ir_cg.py -q` 确认红灯。
- [ ] 使用纯head JVP/VJP实现固定GGN乘积。白化系统A=I+sqrt(R)Gsqrt(R)，零初始化，最多2次CG，内积FP64累加。实现`zero_rhs/early_converged/truncated_usable/numerical_failure`四种status。
- [ ] 验证求解不会改变主模型参数、optimizer、RNG、head mask或任何buffer。2次未达到rtol不报converged、不自动fallback。
- [ ] 保存显式矩阵对照与求解状态测试，提交：`feat: implement bounded head-only GGN response solve`。
