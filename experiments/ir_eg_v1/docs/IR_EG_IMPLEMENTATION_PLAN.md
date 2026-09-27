# IR-EG / BR-IR-EG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This document authorizes planning only; remote training requires the user's separate execution authorization.

**Goal:** 在冻结 DAOT＋FastTrust 联合底座上实现可核验的 IR-EG，再独立实现 BR-IR-EG，分别验证性能增量与计算节省。

**Architecture:** 复用原完整 EG 预测、事务与正式 AdamW 提交。域头响应使用实际原点/预测点梯度差、冻结二阶矩度量及固定2次CG的加权CE-GGN；先实现重放参考版，再实现域头缓存版。最后加入上一接受主步的非域头原始梯度历史。

**Tech Stack:** 既有 Python/PyTorch 环境；单GPU FP32；现有AdamW；pytest。不得自动升级依赖，不默认启用AMP/DDP/compile。

**Spec:** 同目录 `IR_EG_DESIGN_SPEC.md`。实施者须同时阅读两份文件。

**Status:** 未实施；以下命令、测试名和通过要求均为后续验收计划，不是已经执行的结果。

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

## Review Focus

1. `grad=None`、参数共享、分阶段激活、零学习率：保持真实优化器状态语义，不补零掩盖缺失。
2. Dropout调用顺序和零ReLU激活：掩码可复现，不能由输出除输入估计。
3. 原点normalizer及ctx pending状态：只有原点状态被提交，重复求解不能多推进一次。
4. 预测图建立后域头原地写入：参考版先释放图，缓存版使用独立phi_leaf。
5. CG预算耗尽与数值失败：前者允许截断解，后者回退；日志不得合并为同一个success。

## 0. 工作区、路径与交付物

在执行阶段先建立以冻结提交为基线的独立git工作区。新包结构：

```text
experiments/ir_eg_v1/
  code/
    model_dual_cvsincnet.py                 # 从冻结包继承；只加入受控head捕获入口
    SSDG/train_ssdg.py                     # 只提取RC4 loss assembly，不改原项
    cvsrffi/game_tracking/
      solvers.py                          # 保留原EG路径；暴露真实clip系数的等价接口
      state.py                            # 既有事务；必要扩展外层显式状态所有权
      legacy/objective.py                 # 提取label loss assembly，不改运算顺序
    cvsrffi/xuc_fusion/
      objective.py                        # 原闭包兼容；新增packet构造入口
      dr_objective.py                     # normalizer/RC4原生语义
      response_config.py                  # 注册IR/BR方法，不继承错误的旧方法调度
      response_runtime.py                 # 统一集成点
      ir_types.py                         # 类型、参数布局、call ledger
      ir_head.py                          # 掩码重放、纯head、head梯度/GGN
      ir_metric.py                        # AdamW局部R
      ir_cg.py                            # 固定预算白化CG
      ir_solver.py                        # IR-ref/IR-cached事务编排
      ir_objective.py                     # ObjectivePacket和纯loss重组
      ir_telemetry.py                      # 计数、隔离诊断、JSON记录
      br_history.py                       # 上一接受步最终原始梯度
    scripts/check_ir_acceptance.py         # CPU/CUDA受限验收；不加载target
  tests/test_ir_*.py
  configs/ir_eg_v1.json
  configs/br_ir_eg_v1.json
  configs/convergence_confirmation.json    # 只有完整冻结后才可启动相应确认实验
  docs/IR_EG_DESIGN_SPEC.md
  docs/IR_EG_IMPLEMENTATION_PLAN.md
  acceptance/
```

以上新增文件是拟建路径。现有文件从冻结执行包及其必要依赖复制，记录来源和hash；不能以同名根目录文件代替。全部命令在新包根目录执行，测试入口负责把新包code置于导入路径首位，并核验实际模块文件。

最小交付：实现diff、baseline manifest、梯度/loss ledger、测试报告、CPU/CUDA验收记录、运行配置、遥测schema、成本报告、实验记录。不得只交付“测试通过”文字。

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

## Task 7：遥测、机制探针及不干扰验收

**Files:** Create `ir_telemetry.py`, `scripts/check_ir_acceptance.py`, `tests/test_ir_diagnostics.py`, `tests/test_ir_reporting.py`。

**Consumes:** StepResult、原点/预测/正式状态摘要、合法source train/V只读接口。

**Produces:** 每步JSONL、阶段审计表、探针JSON、profiling CSV、状态对齐报告。

- [ ] 写`test_diagnostics_do_not_change_update`、`test_negative_recovery_gain_not_clamped`、`test_missing_metrics_are_null`、`test_executed_not_equal_configured`、`test_no_target_access_in_acceptance`、`test_actual_clip_displacement_logged`。
- [ ] 运行 `python -m pytest tests/test_ir_diagnostics.py tests/test_ir_reporting.py -q` 确认红灯。
- [ ] 实现规范9的字段。每1000个接受步与既有离散阶段边界执行重诊断；副本域头10次拟合，源训练拟合/V独立监测，严禁回写。实际delta与kappa缩放后的delta分别计算非线性响应残差。
- [ ] 对诊断开/关运行同一受限轨迹，所有训练状态完全一致。CPU基础验收命令 `python code/scripts/check_ir_acceptance.py --device cpu --output acceptance/cpu`；CUDA命令同入口`--device cuda`，仅在设备可用且授权的受限验收中运行。
- [ ] 报告明确分开旧测试、新测试、未运行测试；提交：`test: audit IR response execution and noninterfering telemetry`。

## Task 8：BR历史预测与恢复

**Files:** Create `br_history.py`, `tests/test_ir_br_history.py`, `tests/test_ir_br_resume.py`；Extend `ir_solver.py`与配置schema。

**Consumes:** 上一接受主步最终原始梯度、当前原点head梯度、阶段signature、ParamLayout。

**Produces:** `BRHistory.predictor(nonhead_current_activity, h0, signature, accepted_index) -> GradientTuple`；`BRHistory.commit(final_raw_gradients, signature, accepted_index)`；完整state_dict。

- [ ] 写 `test_cache_is_previous_accepted_final_raw`、`test_history_age_one_not_eight`、`test_refresh_every_eight`、`test_boundaries_force_refresh`、`test_none_activity_requires_refresh`、`test_R1_equals_IR`、`test_kappa0_equals_BR_EG_not_full_EG`。
- [ ] 写 `test_resume_matches_continuous`、`test_rejected_step_does_not_update_cache`、`test_cache_layout_mismatch_fails`。运行 `python -m pytest tests/test_ir_br_history.py tests/test_ir_br_resume.py -q` 确认红灯。
- [ ] E1–20用完整EG；E21首次及规范8边界强制当前完整原点梯度。普通BR步保留原点必要前向，当前head梯度fresh；非head用上一接受步final raw。混合梯度先原生全局裁剪，再虚拟AdamW，R使用该真实预测的clip。
- [ ] 验证刷新周期1退化为IR，全历史字段在checkpoint中保存；普通BR步不因为捕获head重算整个骨干反向，内部必要导数另计。成本对比包含FP32历史缓存显存。
- [ ] 保存R1等价与续训报告，提交：`feat: add block-refreshed history predictor for IR-EG`。

## Task 9：实验矩阵、收敛协议和报告

**Files:** Create `configs/ir_eg_v1.json`, `configs/br_ir_eg_v1.json`, `configs/convergence_confirmation.json`与`docs/experiment_protocol.md`、`tests/test_ir_experiment_manifest.py`。

**Consumes:** 已通过验收的baseline/IR/BR实现、冻结source contract、原生配置、既有收敛策略。

**Produces:** launch=false完整矩阵；历史预算审计和科学收敛确认两个独立配置族；独立prediction/scorer合同。

- [ ] 写 `test_all_methods_native_DR_enabled`、`test_equal_physical_exposure`、`test_no_target_selection`、`test_budget_audit_not_convergence`、`test_unapproved_runs_not_launched`、`test_encoder_off_keeps_head_supervision`。
- [ ] 运行 `python -m pytest tests/test_ir_experiment_manifest.py -q` 确认红灯。
- [ ] 准备性能矩阵SIM/EG/IR；机制消融IR_G0、规范10.1精确定义的OR-EG原点特征响应控制；效率矩阵BR_EG/BR_IR。先限定source筛查，再冻结候选作392005/392006/392007配对确认。所有data/augmentation/eval随机流单独登记，不以model seed替代。
- [ ] 明确相同曝光成本与相同source收敛成本。固定预算到点记录而不自动宣布收敛；未冻结完整收敛manifest的配置不得启动确认训练。目标预测在冻结后生成，真值独立连接。
- [ ] 提交预注册方案与未执行状态：`docs: preregister IR-EG performance and BR efficiency experiments`。不得在本任务中擅自启动或恢复远端长训练。

## 2. 额外验收算例

以下是拟写测试，不是本次训练结果：

1. 纯旋转线性场：完整EG与IR的更新矩阵按设计推导匹配；不只检查loss。
2. 带强任务曲率的反例：能够重现强响应未必更稳，不为所有kappa写错误的单调改进断言。
3. 二次域头目标：G为精确Hessian时，kappa=1的充分精度小系统解匹配线性隐式方程；固定2次CG仍按近似状态报告。
4. 非对称L/U：U只改变head直接梯度，仍能改变域头对theta的响应；不能硬设交叉Jacobian负转置关系。
5. 相同位置的随机重放：h0-hp严格为零或在预设数值容差内，未固定掩码的负测试必须能被检测。
6. 图生命期：开启autograd anomaly检测，禁止存活图期间修改参数；正确器完成后显存不随主步线性累积。

## 3. 完成标准与暂停标准

开发完成必须同时满足：旧EG回归、IR数学与事务测试、ref/cached对齐、CPU验收、实际CUDA验收（未运行则明确缺项）、配置与遥测完整。通过开发验收不等于性能成功。

算法是否值得继续由冻结的源侧机制探针、联合训练性能、稳定性和成本共同决定。若IR没有实际响应、响应主要来自随机不一致、或收益只能由重写裁剪解释，不得宣称对手响应机制成立。若BR成本没有降低或关键性能下降，不晋升默认版本。

停止/暂停研究候选的原因须区分：技术故障、数值不稳定、机制证据不足、性能不达标、预算安全上限。不得用跳过坏batch、修改seed、临时增大kappa、扩大CG次数或减少U曝光掩盖失败。

最终报告必须列出：修改文件、基准hash、算法近似条件、每种子结果、全部消融与失败、实际动作次数、总成本、未完成验证以及是否晋升。未验证前默认不替换现有强基线。
