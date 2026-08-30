# ADV3B02-BiCAD-XR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变旧ADV3B02默认行为和部署推理图的前提下，实现报告定义的BiCAD-XR训练机制、D0–F3候选矩阵、诊断artifact和严格Phase1评估闭合。

**Architecture:** 新功能集中在`code/cvsrffi/phase1_bicad_xr/`，通过`BiCADXRConfig`和`BiCADXRTrainer`组合现有`DualCVSincNetDisentangle`输出。旧模型与`train_ssdg.py`只增加显式构造参数、feature出口和`phase1_method=bicad_xr`钩子；默认ADV3B02路径不调用新package。

**Tech Stack:** Python3.10、PyTorch、pytest、现有CV-SincNet WiSig/ManySig数据入口、`ConcatSatChannelAugment`、Git。

**Spec:** `docs/superpowers/specs/2026-08-30-adv3b02-bicad-xr-design.md`

## Global Constraints

- Phase1只读取source `L_s/U_s/V_cal/V_select`；禁止Phase2、target receiver、support、query和truth路径。
- 正式星地增强固定为`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`和三种`LEO_WEAK`课程；禁止把`mixed_orbit`单前向带入本方法。
- 旧ADV3B02默认CLI、checkpoint构造和推理路径必须保持不变。
- V1关闭FastTrust、pseudo-label、CSD、HCF transport、26D content LODO、HDRO、proxy unknown、open-world feature loss、Fishr、generic MixUp和MixStyle。
- 所有类、receiver、day和channel使用同一公式；禁止具体ID专属分支或阈值。
- TDD强制执行：每个新行为必须先有可观察的RED，再写production code并得到GREEN。
- Windows测试使用已激活`ssr-gpu`环境；不得使用`pwsh`。
- 本计划不启动N607正式性能矩阵；只允许实现完成后的真实checkpoint无query smoke。

## File Map

| 文件 | 职责 |
|---|---|
| `phase1_bicad_xr/config.py` | 冻结候选、阶段、损失权重与冲突开关 |
| `phase1_bicad_xr/heads.py` | class-conditional factorized adversarial heads和条件映射 |
| `phase1_bicad_xr/losses.py` | conditional cross-cov、pair、margin与tail损失 |
| `phase1_bicad_xr/sampler.py` | TX/RX平衡主batch和缺cell可mask结构化episode |
| `phase1_bicad_xr/xdc.py` | ridge donor、质量权重、query exchange与KD |
| `phase1_bicad_xr/tangent.py` | 类条件receiver中心、SVD基、factual/worst shift |
| `phase1_bicad_xr/gradients.py` | 梯度比控制、shared-stem firewall和D6投影 |
| `phase1_bicad_xr/metrics.py` | probe、迁移矩阵、tail、资源与artifact schema |
| `phase1_bicad_xr/trainer.py` | Stage0–4路由、稀疏机制调用、总损失与checkpoint runtime |
| `launch_phase1_bicad_xr_matrix_20260830.py` | source-LORO计划、worker、严格评估与闭合 |
| `code/tests/phase1_bicad_xr/` | 聚焦协议、单元、集成和launcher测试 |

---

### Task 1: 冻结配置、候选注册表和阶段调度

**Files:**
- Create: `code/cvsrffi/phase1_bicad_xr/__init__.py`
- Create: `code/cvsrffi/phase1_bicad_xr/config.py`
- Create: `code/tests/phase1_bicad_xr/test_config.py`

**Interfaces:**
- Produces: `BiCADXRConfig`、`BiCADXRStage`、`candidate_config(candidate_id,overrides=None)`、`stage_for_update(update,total_updates)`、`candidate_diff(left,right)`。
- `BiCADXRConfig`必须包含所有开关和精确权重；后续任务只读该dataclass，不自行解释candidate ID。

- [ ] **Step 1: 写RED配置测试**

```python
def test_v1_alias_is_d5_plus_sparse_xdc_and_tail_only():
    cfg = candidate_config("ADV3B02-BiCAD-XDC-V1")
    assert cfg.factorized_domains and cfg.conditional_cdan
    assert cfg.zdom_tx_adversary and cfg.conditional_xcov
    assert cfg.gradient_firewall and cfg.sparse_xdc and cfg.margin_tail
    assert not cfg.task_protected_gradient
    assert not cfg.xdc_kd and not cfg.paired_satellite
    assert not cfg.receiver_tangent and not cfg.swad

def test_forbidden_legacy_features_fail_closed():
    with pytest.raises(ValueError, match="incompatible"):
        candidate_config("D5", overrides={"use_fasttrust": True})

@pytest.mark.parametrize(
    ("update", "stage"),
    [(1,"stage0"),(500,"stage0"),(501,"stage1"),(1750,"stage1"),
     (1751,"stage2"),(3500,"stage2"),(3501,"stage3"),(4500,"stage3"),
     (4501,"stage4"),(5000,"stage4")],
)
def test_five_stage_boundaries(update, stage):
    assert stage_for_update(update, 5000).name == stage
```

- [ ] **Step 2: 运行RED并确认因模块不存在失败**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_config.py -q`

Expected: collection error naming `cvsrffi.phase1_bicad_xr.config`.

- [ ] **Step 3: 实现最小配置API**

```python
@dataclass(frozen=True)
class BiCADXRConfig:
    candidate_id: str
    factorized_domains: bool = False
    conditional_cdan: bool = False
    zdom_tx_adversary: bool = False
    conditional_xcov: bool = False
    gradient_firewall: bool = False
    task_protected_gradient: bool = False
    sparse_xdc: bool = False
    xdc_kd: bool = False
    paired_satellite: bool = False
    margin_tail: bool = False
    receiver_tangent: str = "off"
    swad: bool = False
    batch_size: int = 96
    xdc_interval: int = 4
    pair_interval: int = 4
    lambda_sat_cls: float = 0.68
    lambda_cond_xcov: float = 0.02
    gradient_firewall_scale: float = 0.05

def stage_for_update(update: int, total_updates: int) -> BiCADXRStage:
    if not 1 <= update <= total_updates:
        raise ValueError("update must be in [1,total_updates]")
    progress = update / total_updates
    boundaries = ((0.10,"stage0"),(0.35,"stage1"),(0.70,"stage2"),(0.90,"stage3"),(1.0,"stage4"))
    return next(BiCADXRStage(name) for limit, name in boundaries if progress <= limit)
```

- [ ] **Step 4: GREEN并验证D0–F3相邻差异**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_config.py -q`

Expected: PASS；每个相邻候选只有规格允许的主要差异。

- [ ] **Step 5: 提交**

Run: `git add code/cvsrffi/phase1_bicad_xr/__init__.py code/cvsrffi/phase1_bicad_xr/config.py code/tests/phase1_bicad_xr/test_config.py && git commit -m feat:bicad-xr-config`

### Task 2: 类条件多因素头、双向对抗和基础损失

**Files:**
- Create: `code/cvsrffi/phase1_bicad_xr/heads.py`
- Create: `code/cvsrffi/phase1_bicad_xr/losses.py`
- Create: `code/tests/phase1_bicad_xr/test_heads.py`
- Create: `code/tests/phase1_bicad_xr/test_losses.py`

**Interfaces:**
- Consumes: `BiCADXRConfig`。
- Produces: `conditional_outer(z_id,tx,num_classes)`、`FactorizedAdversarialHeads.forward(...)`、`conditional_cross_covariance(...)`、`paired_satellite_loss(...)`、`classification_margin(...)`。

- [ ] **Step 1: 写RED条件映射与梯度测试**

```python
def test_conditional_outer_uses_true_one_hot():
    z = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
    y = torch.tensor([0,1])
    out = conditional_outer(z, y, num_classes=2)
    assert out.tolist() == [[1.,0.,2.,0.],[0.,3.,0.,4.]]
    out.sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))

def test_conditional_outer_rejects_missing_labels():
    with pytest.raises(ValueError, match="TX labels"):
        conditional_outer(torch.randn(4,160), None, 6)
```

- [ ] **Step 2: 写RED因素化头和cross-cov测试**

```python
def test_factorized_heads_return_three_identity_adversaries_and_four_environment_heads():
    heads = FactorizedAdversarialHeads(160,6,4,3,4)
    out = heads(torch.randn(8,160), torch.randn(8,160), torch.arange(8)%6, grl_identity=.2, grl_tx=.08)
    assert set(out) == {"id_receiver","id_day","id_channel","dom_receiver","dom_day","dom_channel","dom_tx"}

def test_conditional_cross_covariance_is_zero_without_valid_tx_group():
    zid = torch.randn(3,4,requires_grad=True)
    loss = conditional_cross_covariance(zid,torch.randn(3,5),torch.tensor([0,1,2]))
    assert loss.item() == 0.0
    loss.backward()
    assert zid.grad is not None
```

- [ ] **Step 3: 运行RED**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_heads.py code/tests/phase1_bicad_xr/test_losses.py -q`

Expected: imports/functions missing.

- [ ] **Step 4: 实现最小模块**

```python
def conditional_outer(z_id, tx, num_classes):
    if tx is None:
        raise ValueError("TX labels are required for conditional CDAN")
    one_hot = F.one_hot(tx.long(), num_classes=num_classes).to(z_id.dtype)
    return torch.einsum("bd,bc->bdc", z_id, one_hot).reshape(z_id.size(0), -1)

def conditional_cross_covariance(z_id, z_dom, tx):
    losses = []
    for class_id in torch.unique(tx):
        mask = tx == class_id
        if int(mask.sum()) < 2:
            continue
        a = z_id[mask] - z_id[mask].mean(0, keepdim=True)
        b = z_dom[mask] - z_dom[mask].mean(0, keepdim=True)
        cov = a.T @ b / (int(mask.sum()) - 1)
        losses.append(cov.square().sum() / (z_id.size(1) * z_dom.size(1)))
    return torch.stack(losses).mean() if losses else z_id.sum() * 0.0
```

- [ ] **Step 5: GREEN并做有限值反向测试**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_heads.py code/tests/phase1_bicad_xr/test_losses.py -q`

Expected: PASS，无NaN/Inf。

- [ ] **Step 6: 提交**

Run: `git add code/cvsrffi/phase1_bicad_xr/heads.py code/cvsrffi/phase1_bicad_xr/losses.py code/tests/phase1_bicad_xr/test_heads.py code/tests/phase1_bicad_xr/test_losses.py && git commit -m feat:bicad-xr-adversarial-heads`

### Task 3: TX/RX平衡采样与结构化XDC episode

**Files:**
- Create: `code/cvsrffi/phase1_bicad_xr/sampler.py`
- Create: `code/tests/phase1_bicad_xr/test_sampler.py`

**Interfaces:**
- Produces: `BalancedIndexPool`、`StructuredEpisode(indices,tx,receiver,day,valid_cells)`、`build_structured_episode(...)`。
- `indices`只引用真实物理样本；`valid_cells`为`[C,R]`布尔mask。

- [ ] **Step 1: 写RED缺cell和不重复测试**

```python
def test_structured_episode_masks_missing_cells_without_duplication():
    tx = [0,0,1,1,1]
    rx = [0,1,0,0,1]
    ep = build_structured_episode(tx,rx,day=[0]*5,samples_per_cell=2,generator=torch.Generator().manual_seed(7))
    assert ep.valid_cells.tolist() == [[False,False],[True,False]]
    assert len(ep.indices) == len(set(ep.indices))

def test_episode_never_fills_a_missing_tx_rx_cell_from_another_tx():
    ep = build_structured_episode([0,0],[0,0],[0,0],samples_per_cell=2,generator=torch.Generator().manual_seed(1))
    assert ep.tx.tolist() == [0,0]
    assert ep.receiver.tolist() == [0,0]
```

- [ ] **Step 2: 运行RED**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_sampler.py -q`

Expected: module/function missing.

- [ ] **Step 3: 实现真实cell采样**

Implementation rule: 对每个`(tx,receiver)`建立索引池；只从数量不少于`samples_per_cell`的cell无放回抽样。cell不足时整体mask，不重复单个样本，也不从其他TX或receiver补齐。

- [ ] **Step 4: GREEN并增加确定性测试**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_sampler.py -q`

Expected: PASS；相同generator seed得到相同indices。

- [ ] **Step 5: 提交**

Run: `git add code/cvsrffi/phase1_bicad_xr/sampler.py code/tests/phase1_bicad_xr/test_sampler.py && git commit -m feat:bicad-xr-structured-sampler`

### Task 4: XDC ridge exchange与公共头蒸馏

**Files:**
- Create: `code/cvsrffi/phase1_bicad_xr/xdc.py`
- Create: `code/tests/phase1_bicad_xr/test_xdc.py`

**Interfaces:**
- Consumes: 结构化episode后的`z_id,tx,receiver`和公共TX logits。
- Produces: `fit_receiver_donors(...) -> DonorBank`、`xdc_losses(...) -> XDCLossOutput`、`donor_query_matrix(...)`。

- [ ] **Step 1: 写RED稳定ridge与donor过滤测试**

```python
def test_ridge_uses_solve_and_produces_finite_weights():
    z = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.]])
    y = torch.tensor([0,1,0,1])
    bank = fit_receiver_donors(z,y,torch.tensor([0,0,0,0]),num_classes=2,ridge=1e-2)
    assert torch.isfinite(bank.weights[0]).all()

def test_low_coverage_donor_is_skipped():
    bank = fit_receiver_donors(torch.randn(3,4),torch.tensor([0,0,0]),torch.zeros(3,dtype=torch.long),num_classes=2,ridge=1e-2)
    assert bank.valid_receivers.numel() == 0
```

- [ ] **Step 2: 写RED donor停止梯度与KD测试**

```python
def test_query_features_receive_gradient_but_donor_weights_do_not():
    output = xdc_losses(z_id,tx,receiver,public_logits,num_classes=2,temperature=2.0)
    output.total.backward()
    assert z_id.grad is not None
    assert all(weight.grad is None for weight in output.detached_donor_weights)
```

- [ ] **Step 3: 运行RED**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_xdc.py -q`

Expected: module/functions missing.

- [ ] **Step 4: 实现稳定求解和统一质量权重**

```python
gram = z @ z.T
alpha = torch.linalg.solve(gram + ridge * torch.eye(n,device=z.device,dtype=z.dtype), one_hot)
weight = (z.T @ alpha).detach()
quality = accuracy * margin.clamp_min(0.0) / torch.log1p(condition_number)
```

非有限condition、类别覆盖不足2类、自身support准确率低于0.25的donor跳过。无有效跨receiver donor时返回与`z_id`连接的零损失并记录`skip_reason`。

- [ ] **Step 5: GREEN并验证迁移矩阵mask**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_xdc.py -q`

Expected: PASS；矩阵未评估cell为NaN且不参与均值。

- [ ] **Step 6: 提交**

Run: `git add code/cvsrffi/phase1_bicad_xr/xdc.py code/tests/phase1_bicad_xr/test_xdc.py && git commit -m feat:bicad-xr-xdc`

### Task 5: margin-tail、receiver tangent和梯度控制

**Files:**
- Create: `code/cvsrffi/phase1_bicad_xr/tangent.py`
- Create: `code/cvsrffi/phase1_bicad_xr/gradients.py`
- Create: `code/tests/phase1_bicad_xr/test_tail.py`
- Create: `code/tests/phase1_bicad_xr/test_tangent.py`
- Create: `code/tests/phase1_bicad_xr/test_gradients.py`
- Modify: `code/cvsrffi/phase1_bicad_xr/losses.py`

**Interfaces:**
- Produces: `GroupedMarginTail`、`ReceiverTangentBank`、`project_conflicting_gradient`、`GradientRatioController`、`scale_parameter_gradients`。

- [ ] **Step 1: 写RED margin和三层CVaR测试**

```python
def test_margin_tail_weights_only_group_risks():
    tail = GroupedMarginTail(cvar_fraction=.2,weights=(.6,.3,.1),ema=.9)
    loss, audit = tail(logits,tx,receiver,day,channel)
    assert set(audit.group_risks) == {"tx_rx","tx_rx_day","tx_rx_channel"}
    assert abs(sum(audit.component_weights.values())-1.0) < 1e-12
```

- [ ] **Step 2: 写RED tangent source-only和F1/F2测试**

```python
def test_tangent_basis_uses_only_observed_source_centers():
    bank = ReceiverTangentBank(num_classes=2,num_receivers=2,dim=3,rank=2)
    bank.update(z,tx,rx)
    basis = bank.basis()
    assert basis.shape[0] == 3 and basis.shape[1] <= 2

def test_worst_shift_does_not_reduce_requested_attack_loss():
    shifted = bank.worst_direction(z,tx,classifier,rho=.1)
    assert margin_risk(classifier(shifted),tx) >= margin_risk(classifier(z),tx)-1e-6
```

- [ ] **Step 3: 写RED梯度投影测试**

```python
def test_projection_removes_negative_tx_component():
    gy = torch.tensor([1.,0.])
    ga = torch.tensor([-1.,1.])
    projected = project_conflicting_gradient(gy,ga)
    assert torch.dot(projected,gy) >= -1e-7

def test_non_conflicting_gradient_is_unchanged():
    gy=torch.tensor([1.,0.]); ga=torch.tensor([1.,1.])
    assert torch.equal(project_conflicting_gradient(gy,ga),ga)
```

- [ ] **Step 4: 运行RED**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_tail.py code/tests/phase1_bicad_xr/test_tangent.py code/tests/phase1_bicad_xr/test_gradients.py -q`

Expected: missing APIs.

- [ ] **Step 5: 实现最小状态与有限值保护**

Implementation rules: EMA状态使用`detach()`更新；SVD输入做层次收缩但不读取heldout receiver；F2只对切向系数求一次梯度；非有限SVD/梯度直接抛异常；投影只处理显式传入的参数列表。

- [ ] **Step 6: GREEN**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_tail.py code/tests/phase1_bicad_xr/test_tangent.py code/tests/phase1_bicad_xr/test_gradients.py -q`

Expected: PASS。

- [ ] **Step 7: 提交**

Run: `git add code/cvsrffi/phase1_bicad_xr/losses.py code/cvsrffi/phase1_bicad_xr/tangent.py code/cvsrffi/phase1_bicad_xr/gradients.py code/tests/phase1_bicad_xr/test_tail.py code/tests/phase1_bicad_xr/test_tangent.py code/tests/phase1_bicad_xr/test_gradients.py && git commit -m feat:bicad-xr-tail-tangent-gradients`

### Task 6: 双骨干构造与BiCAD-XR训练器集成

**Files:**
- Modify: `code/model_dual_cvsincnet.py`
- Modify: `code/post_stage_common.py`
- Create: `code/cvsrffi/phase1_bicad_xr/trainer.py`
- Create: `code/tests/phase1_bicad_xr/test_model_integration.py`
- Create: `code/tests/phase1_bicad_xr/test_trainer.py`

**Interfaces:**
- Consumes: Tasks1–5全部公共API。
- Produces: `BiCADXRBatch`、`BiCADXRTrainOutput`、`BiCADXRTrainer.compute_step(...)`、checkpoint runtime字典。

- [ ] **Step 1: 写RED旧模型兼容测试**

```python
def test_default_dual_model_state_dict_is_unchanged_when_bicad_disabled():
    old = build_dual_model(6,12)
    new = build_dual_model(6,12,bicad_xr=False)
    assert old.state_dict().keys() == new.state_dict().keys()

def test_bicad_model_exports_shared_and_branch_features_without_changing_tx_logits():
    model = build_dual_model(6,12,bicad_xr=True)
    out = model(x,y,return_aux=True)
    assert {"z_id","z_dom","shared_features","identity_features","domain_features"} <= set(out)
```

- [ ] **Step 2: 写RED阶段路由和稀疏调用测试**

```python
def test_stage0_has_no_grl_xdc_tail_or_tangent():
    out = trainer.compute_step(batch,update=1,total_updates=5000)
    assert out.audit["stage"] == "stage0"
    assert out.audit["grl_identity"] == 0.0
    assert not out.audit["xdc_called"] and not out.audit["tail_called"]

def test_xdc_runs_every_four_steps_only_after_stage2():
    assert not trainer.compute_step(batch,1748,5000).audit["xdc_called"]
    assert trainer.compute_step(batch,1752,5000).audit["xdc_called"]

def test_swad_updates_only_for_f3_in_stage4():
    f3 = make_trainer("F3")
    f2 = make_trainer("F2")
    assert not f3.compute_step(batch,4500,5000).audit["swad_updated"]
    assert f3.compute_step(batch,4504,5000).audit["swad_updated"]
    assert not f2.compute_step(batch,4504,5000).audit["swad_updated"]
```

- [ ] **Step 3: 运行RED**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_model_integration.py code/tests/phase1_bicad_xr/test_trainer.py -q`

Expected: new constructor/trainer missing.

- [ ] **Step 4: 添加最小模型出口和构造参数**

`build_dual_model(...,bicad_xr=False)`默认关闭。`post_stage_common.build_baseline_model`只在`model_args.phase1_method=="bicad_xr"`时传入新开关。训练头不得进入`return_aux=False`快速推理路径。

- [ ] **Step 5: 实现训练器总损失路由**

`BiCADXRTrainer.compute_step`必须返回各分量原始值、加权值、是否调用、有效样本/组/donor数和skip reason。Stage4条件DANN系数为峰值0.6，shared stem LR scale为0.1。pair只在E3打开并复用concat成对输出。F3在Stage4维护source-LORO低风险窗口参数平均，F2及其他候选不得创建或更新SWAD状态。

- [ ] **Step 6: GREEN并运行旧双骨干回归**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_model_integration.py code/tests/phase1_bicad_xr/test_trainer.py -q`

Run: `python -m pytest code/tests/phase1_hcfdg/test_model.py code/tests/test_phase1_adv3b03_src5_day123_seedscan.py -q`

Expected: 全部PASS。

- [ ] **Step 7: 提交**

Run: `git add code/model_dual_cvsincnet.py code/post_stage_common.py code/cvsrffi/phase1_bicad_xr/trainer.py code/tests/phase1_bicad_xr/test_model_integration.py code/tests/phase1_bicad_xr/test_trainer.py && git commit -m feat:integrate-bicad-xr-dual-backbone`

### Task 7: 显式SSDG入口、协议负测和checkpoint runtime

**Files:**
- Modify: `code/SSDG/train_ssdg.py`
- Create: `code/tests/phase1_bicad_xr/test_ssdg_entry.py`
- Create: `code/tests/phase1_bicad_xr/test_protocol.py`

**Interfaces:**
- Consumes: `BiCADXRTrainer`。
- Produces: `--phase1_method bicad_xr`入口、`bicad_xr_runtime` checkpoint字段、旧路径零行为变化。

- [ ] **Step 1: 写RED CLI与协议测试**

```python
def test_bicad_entry_forces_concat_leo_weak_contract():
    args = parse(["--phase1_method","bicad_xr","--candidate_id","D5"])
    resolved = resolve_bicad_protocol(args)
    assert resolved.use_concat_sat_channel_aug
    assert resolved.concat_sat_ce_only
    assert resolved.concat_sat_ce_weight == pytest.approx(.68)
    assert resolved.concat_sat_start_epoch == 80
    assert resolved.sat_train_scenarios == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"

@pytest.mark.parametrize("flag", ["--use_fasttrust","--use_mixstyle","--sat_train_scenario=mixed_orbit"])
def test_bicad_rejects_incompatible_flags(flag):
    with pytest.raises(ValueError,match="BiCAD-XR"):
        parse_and_resolve(["--phase1_method","bicad_xr",flag])
```

- [ ] **Step 2: 写RED source-only参数表面测试**

测试parser和launcher均不存在`target_rx`、`phase2`、`support`、`query`或`truth`输入；运行时报告固定`target_access=false`。

- [ ] **Step 3: 运行RED**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_ssdg_entry.py code/tests/phase1_bicad_xr/test_protocol.py -q`

Expected: parser/route missing.

- [ ] **Step 4: 实现显式路由**

旧训练循环的BiCAD入口只做四件事：冻结协议配置、构造`BiCADXRTrainer`、在有标签主步调用`compute_step`、把audit写入现有metrics。非BiCAD路径不导入或构造训练器。

- [ ] **Step 5: GREEN并验证旧CLI帮助输出**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_ssdg_entry.py code/tests/phase1_bicad_xr/test_protocol.py -q`

Run: `python code/SSDG/train_ssdg.py --help`

Expected: PASS且帮助可生成。

- [ ] **Step 6: 提交**

Run: `git add code/SSDG/train_ssdg.py code/tests/phase1_bicad_xr/test_ssdg_entry.py code/tests/phase1_bicad_xr/test_protocol.py && git commit -m feat:add-bicad-xr-ssdg-entry`

### Task 8: 诊断artifact、launcher和四场景闭合

**Files:**
- Create: `code/cvsrffi/phase1_bicad_xr/metrics.py`
- Create: `code/scripts/launch_phase1_bicad_xr_matrix_20260830.py`
- Create: `code/tests/phase1_bicad_xr/test_metrics.py`
- Create: `code/tests/phase1_bicad_xr/test_launcher.py`

**Interfaces:**
- Produces: `build_plan(...)`、`validate_artifact_closure(...)`、`evaluate_final_checkpoint(...)`、`BiCADXRMetricStore`。

- [ ] **Step 1: 写RED计划矩阵测试**

```python
def test_quick_plan_is_two_folds_three_seeds_and_no_target():
    rows = build_plan(stage="quick",candidates=("D0","D5","ADV3B02-BiCAD-XDC-V1"))
    assert {(r.fold,r.seed) for r in rows} == {(f,s) for f in (1,8) for s in (392001,392002,392003)}
    assert all(r.optimizer_updates == 5000 and r.target_access is False for r in rows)
```

- [ ] **Step 2: 写RED artifact闭合测试**

```python
def test_closure_requires_clean_and_each_leo_scenario(tmp_path):
    write_complete_training_artifacts(tmp_path)
    for scene in ("clean","leo_clear_weak","leo_low_elev_weak"):
        write_eval(tmp_path,scene)
    result = validate_artifact_closure(tmp_path)
    assert not result["complete"]
    assert result["missing"] == ["leo_rain_weak"]
```

- [ ] **Step 3: 写RED诊断schema测试**

schema必须包含条件receiver probe、`z_dom` TX probe、donor→query矩阵、pair指标或N/A、Q0.1 margin、最差组合、梯度比、projection触发率、吞吐、显存、GPU-hours和额外前向比例。

- [ ] **Step 4: 运行RED**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_metrics.py code/tests/phase1_bicad_xr/test_launcher.py -q`

Expected: modules/functions missing.

- [ ] **Step 5: 实现计划、严格重建与闭合**

launcher每row使用不可覆盖目录，checkpoint runtime必须精确匹配candidate/fold/seed/updates/source receivers/train days。`evaluate_final_checkpoint`严格加载后分别调用现有clean和三个LEO弱评估入口；任一missing/unexpected/shape mismatch或场景缺失均不能写`ARTIFACTS_COMPLETE`。

- [ ] **Step 6: GREEN和plan dry-run**

Run: `python -m pytest code/tests/phase1_bicad_xr/test_metrics.py code/tests/phase1_bicad_xr/test_launcher.py -q`

Run: `python code/scripts/launch_phase1_bicad_xr_matrix_20260830.py --stage quick --dry-run --run-id phase1_bicad_xr_dryrun_20260830`

Expected: 18行plan，0个target参数，未创建训练进程。

- [ ] **Step 7: 提交**

Run: `git add code/cvsrffi/phase1_bicad_xr/metrics.py code/scripts/launch_phase1_bicad_xr_matrix_20260830.py code/tests/phase1_bicad_xr/test_metrics.py code/tests/phase1_bicad_xr/test_launcher.py && git commit -m feat:add-bicad-xr-launcher-artifacts`

### Task 9: 全量验证、真实checkpoint smoke、追踪表和发布

**Files:**
- Modify: `analysis/phase1_bicad_xr_traceability.md`
- Create: `automation_reports/CV-SincNet/phase1_bicad_xr_implementation_20260830/report.md`

**Interfaces:**
- Consumes: Tasks1–8全部实现和测试。
- Produces: 本地验证报告、更新后的追踪状态、正式Git发布证据。

- [ ] **Step 1: 运行语法和聚焦套件**

Run: `python -m py_compile code/cvsrffi/phase1_bicad_xr/*.py code/scripts/launch_phase1_bicad_xr_matrix_20260830.py`

Run: `python -m pytest code/tests/phase1_bicad_xr -q`

Expected: 全部PASS。

- [ ] **Step 2: 运行相邻回归套件**

Run: `python -m pytest code/tests/phase1_hcfdg code/tests/test_phase1_adv3b03_src5_day123_seedscan.py -q`

Expected: 全部PASS且旧checkpoint构造测试无变化。

- [ ] **Step 3: 执行真实ADV3B02 checkpoint无query smoke**

使用`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`严格重建，运行最小BiCAD训练步并分别完成clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`小评估。验证`missing=[]`、`unexpected=[]`、`shape_mismatch=[]`、`target_access=false`和四场景artifact齐全。

- [ ] **Step 4: 一次独立P0/P1审查**

审查范围仅限会让真实实验跑错、越权、覆盖输出、不能启动、不能闭合prediction或影响旧ADV3B02的问题。发现P0/P1时修复并最多做一次定点复审；P2不阻断。

- [ ] **Step 5: 更新追踪表与实现报告**

每个`implemented`项必须写实际文件和测试；每个`verified`项必须写命令结果。P13继续标记`deferred`，不得伪报同packet配对完成。报告区分`LOCAL_VERIFIED`与尚未运行的N607性能矩阵。

- [ ] **Step 6: 最终质量检查**

Run: `git diff --check`

Run: `git status -sb`

Expected: 只包含本计划文件；无缓存、checkpoint、日志或数据集进入stage。

- [ ] **Step 7: 精确提交和推送**

Run: `git add -- analysis/phase1_bicad_xr_traceability.md automation_reports/CV-SincNet/phase1_bicad_xr_implementation_20260830/report.md && git commit -m docs:publish-bicad-xr-implementation`

Run: `git push --set-upstream origin HEAD`（若已有upstream则`git push`）

Run: `git ls-remote origin refs/heads/codex/phase1-bicad-xr-20260830`

Expected: 远端OID与`git rev-parse HEAD`完全一致。

## Execution Order and Luna Ownership

1.主对话先执行Task1，冻结所有公共接口。
2.Luna可分别承担Task2、Task3、Task4、Task5的机械实现，但共享分支上必须串行提交；每个Luna只修改任务列出的文件，不得修改共享入口。
3.Task6和Task7由主对话集成，因为它们修改现有双骨干和大型训练入口。
4.Luna可承担Task8的launcher/metrics机械实现；主对话复核计划矩阵和协议边界。
5.Task9由主对话完成验证、独立审查协调、追踪对齐和Git发布。
