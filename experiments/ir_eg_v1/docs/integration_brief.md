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

## Task 5：IR-ref与全状态事务

**Files:** Create `ir_solver.py`, `tests/test_ir_reference.py`, `tests/test_ir_transaction.py`；Modify `response_config.py`, `response_runtime.py`。

**Consumes:** 原生GameSolver、完整closure、metric与CG、HeadBatch、原点事务。

**Produces:** `IREGSolver.reference_step(closure, ctx) -> StepResult`；method registration `IR_EG`。

- [ ] 写`test_kappa_zero_direct_EG`、`test_reference_one_formal_commit`、`test_phir_not_directly_committed`、`test_LU_gradient_ownership`、`test_E21_start_without_TR_ramp`。
- [ ] 写参数化失败注入：原点后、虚拟step后、捕获后、CG内、正确器后、正式step后、EMA/normalizer commit中。运行 `python -m pytest tests/test_ir_reference.py tests/test_ir_transaction.py -q` 确认红灯。
- [ ] 实现规范5.1的重放参考算法，捕获图先释放再修改head。外层事务覆盖教师及原生ctx pending状态；回退使用同一batch与正确器语境，不能读loader下一个batch。
- [ ] 验证kappa=0所有参数、optimizer、RNG、buffer和外部状态bitwise对齐旧EG；响应开启时每步正式提交次数为1。CG截断正常应用，数值失败退回同一步EG，普通EG也失败则中止。
- [ ] 保留旧完整EG通道不变，保存失败注入报告，提交：`feat: add reference implicit-response EG transaction`。

## Task 6：ObjectivePacket及缓存正确器

**Files:** Create `ir_objective.py`, `tests/test_ir_packet.py`, `tests/test_ir_cached.py`；Modify新包`objective.py`、`dr_objective.py`、`legacy/objective.py`、`SSDG/train_ssdg.py`中的纯loss assembly接口。

**Consumes:** 原生task张量、原生post-GRL输入图、HeadBatch、phir数值、ParamLayout。

**Produces:** `build_objective_packet(ctx) -> ObjectivePacket`；`IREGSolver.cached_step(closure, ctx) -> StepResult`。

- [ ] 写 `test_replace_only_registered_adversarial_leaves`、`test_no_native_phi_gradient_leak`、`test_no_live_parameter_mutation`、`test_phi_leaf_gradient_mapping`、`test_U_GRL_zero_and_U_task_nonzero`、`test_scale_call_sequence_preserved`。
- [ ] 运行 `python -m pytest tests/test_ir_packet.py tests/test_ir_cached.py -q` 确认红灯。
- [ ] 提取纯loss assembly，保持原生加法顺序；替换已登记的L/U域头CE叶子而不是`old_total+new_H-old_H`。phi_response detach并clone成独立leaf；最终对psi与phi_leaf求导并按布局映射。
- [ ] 单步对比IR-ref/IR-cached的loss、逐参数梯度、实际AdamW位移、矩、RNG、buffers和所有持久状态。FP32单步`atol=1e-6, rtol=1e-5`，10步参数/矩允许`atol=1e-6, rtol=1e-4`；路由、ID、随机状态与计数必须完全相同。超限定位原因，不临时放宽容差。
- [ ] 验证只执行两次完整场前向/两次完整大参数梯度场，额外反向仅head；记录实际模型调用数而不只报告理论数。提交：`perf: reuse predictor task graph for head-only IR correction`。

## Task 8：BR历史预测与恢复

**Files:** Create `br_history.py`, `tests/test_ir_br_history.py`, `tests/test_ir_br_resume.py`；Extend `ir_solver.py`与配置schema。

**Consumes:** 上一接受主步最终原始梯度、当前原点head梯度、阶段signature、ParamLayout。

**Produces:** `BRHistory.predictor(nonhead_current_activity, h0, signature, accepted_index) -> GradientTuple`；`BRHistory.commit(final_raw_gradients, signature, accepted_index)`；完整state_dict。

- [ ] 写 `test_cache_is_previous_accepted_final_raw`、`test_history_age_one_not_eight`、`test_refresh_every_eight`、`test_boundaries_force_refresh`、`test_none_activity_requires_refresh`、`test_R1_equals_IR`、`test_kappa0_equals_BR_EG_not_full_EG`。
- [ ] 写 `test_resume_matches_continuous`、`test_rejected_step_does_not_update_cache`、`test_cache_layout_mismatch_fails`。运行 `python -m pytest tests/test_ir_br_history.py tests/test_ir_br_resume.py -q` 确认红灯。
- [ ] E1–20用完整EG；E21首次及规范8边界强制当前完整原点梯度。普通BR步保留原点必要前向，当前head梯度fresh；非head用上一接受步final raw。混合梯度先原生全局裁剪，再虚拟AdamW，R使用该真实预测的clip。
- [ ] 验证刷新周期1退化为IR，全历史字段在checkpoint中保存；普通BR步不因为捕获head重算整个骨干反向，内部必要导数另计。成本对比包含FP32历史缓存显存。
- [ ] 保存R1等价与续训报告，提交：`feat: add block-refreshed history predictor for IR-EG`。
