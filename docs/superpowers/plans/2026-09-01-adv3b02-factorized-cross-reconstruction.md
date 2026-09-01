# ADV3B02-FCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏现有ADV3B02关闭态兼容性的前提下，实现报告定义的三因子、物理顺序、可交叉验证的Phase1因子化交叉重构候选`ADV3B02-FCR`。

**Architecture:** 现有ADV3B02继续输出`z_id_raw`作为matched control；显式FCR分支从同一物理片段的clean/LEO配对中提取时序内容`z_s`、激励条件化发射机响应`z_f=[z_f_id,z_tx_state]`和低容量结构化nuisance`z_n`。Decoder固定遵循`G_s→T_zf→C_zn`，输出复数条件均值和有界方差，并通过latent cycle、定向指纹移植、三轴干预和冻结物理特征验证因子角色。

**Tech Stack:** Python 3、PyTorch、complex64线性代数、pytest、现有CV-SincNet/WiSig/LEO_WEAK训练管线、Git。

**Spec:** `docs/superpowers/specs/2026-09-01-adv3b02-factorized-cross-reconstruction-design.md`

## Global Constraints

- 科学语义以`E:\type10-7\项目.md`版本2026-08-30和上述Spec为准；实现追踪表为`docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`。
- 实现目标树固定为`E:\type10-7\github_publish\CVS-RFFI-repo`；不得直接编辑`E:\type10-7\code`运行副本或N607文件。
- `use_fcr=false`时不得实例化FCR参数；旧ADV3B02`state_dict`键、logits、`z_id`、checkpoint严格加载和训练行为必须保持不变。
- FCR采用方案A相对链路语义；`z_n^leo`解释相对clean的新增复合nuisance，不声明恢复纯星地信道或绝对TX CFO。
- Decoder固定为`D=C_zn o T_zf o G_s`；禁止普通latent concat Decoder、逐采样skip、与输入等长的自由`z_n`和目标波形旁路。
- `L_s/U_s/V=0.07/0.63/0.30`；`U_s`不得读取TX真值，`V`不得反向传播或更新持久状态。
- 第一版不使用硬伪标签锚定`U_s`的`z_f`，不把FastTrust、SAT-Anchor、CRRA或另一份ECRS设计擅自拼入核心FCR。
- 保留`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、普通ADV3B02的`lambda_sat_cons=0`和现有LEO_WEAK三段日程；FCR损失只在`phase1_method=adv3b02_fcr`时启用。
- 训练预算默认200epoch；最终分别评测clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。
- 所有本地代码测试先激活`ssr-gpu`；本计划本身不运行训练、不连接query、不启动N607实验。
- 每个任务只stage列出的文件；不得使用`git add -A`。每次提交自动push并独立核对远端分支OID。
- 独立P0/P1审查只在全部本地聚焦验证和一次真实checkpoint无query smoke之后执行一次；不得增加逐任务审批、额外seal或重复审查。

## File Structure

新增模块按责任拆分：

```text
code/cvsrffi/phase1_fcr_types.py           共享dataclass、配置和复数IQ转换
code/cvsrffi/phase1_fcr_interventions.py   配对元数据、能力审计和三轴pair索引
code/cvsrffi/phase1_fcr_canonicalizer.py   粗CFO/相位/增益估计与解析规范化
code/cvsrffi/phase1_fcr_factors.py         内容token、内容生成和z_f编码
code/cvsrffi/phase1_fcr_fingerprint.py     激励条件化TX响应算子
code/cvsrffi/phase1_fcr_nuisance.py        低容量结构化z_n
code/cvsrffi/phase1_fcr_decoder.py         物理顺序概率Decoder
code/cvsrffi/phase1_fcr_physics.py         冻结R_fp和Fisher门控
code/cvsrffi/phase1_fcr_transplant.py      同TX/跨TX/drop-f反事实验证
code/cvsrffi/phase1_fcr_losses.py          完整FCR损失组合
code/cvsrffi/phase1_fcr_schedule.py        E1-200四阶段权重和梯度路由
code/cvsrffi/phase1_fcr_diagnostics.py     独立probe、latent和资源诊断
```

现有大文件只做适配：`code/model_dual_cvsincnet.py`负责开关和输出契约，`code/train.py`负责调用，`code/dataset_wisig.py`与`code/baseline_origin_sat_view.py`负责配对元数据，`code/cvsrffi/checkpoint.py`负责持久化。

---

### Task 1: 锁定关闭态兼容并建立共享类型

**Files:**
- Create: `code/cvsrffi/phase1_fcr_types.py`
- Modify: `code/model_dual_cvsincnet.py:507-571`
- Test: `code/tests/test_phase1_fcr_model_contract.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: 现有`build_dual_model(...)`和`DualCVSincNetDisentangle`构造函数。
- Produces: `FCRConfig`、`FCRPairBatch`、`FCRFactorOutput`、`FCRDecodeOutput`、`FCRLossOutput`；模型参数`use_fcr: bool=False`与`fcr_config: Optional[FCRConfig]=None`。

- [ ] **Step 1: 写关闭态兼容失败测试**

```python
def test_fcr_off_preserves_legacy_state_dict_and_outputs():
    torch.manual_seed(7)
    legacy = build_dual_model(num_classes=6, num_domains=5)
    torch.manual_seed(7)
    candidate = build_dual_model(num_classes=6, num_domains=5, use_fcr=False)
    candidate.load_state_dict(legacy.state_dict(), strict=True)
    assert not any(key.startswith("fcr.") for key in candidate.state_dict())
    x = torch.randn(2, 2, 256)
    legacy.eval()
    candidate.eval()
    with torch.no_grad():
        old = legacy(x, return_aux=True)
        new = candidate(x, return_aux=True)
    torch.testing.assert_close(new["tx_logits"], old["tx_logits"])
    torch.testing.assert_close(new["z_id"], old["z_id"])
```

- [ ] **Step 2: 运行测试确认接口尚不存在**

Run in an activated environment:

```text
conda activate ssr-gpu
python -m pytest code/tests/test_phase1_fcr_model_contract.py::test_fcr_off_preserves_legacy_state_dict_and_outputs -v
```

Expected: FAIL，`build_dual_model`不接受`use_fcr`。

- [ ] **Step 3: 建立共享类型和显式配置**

```python
@dataclass(frozen=True)
class FCRConfig:
    input_len: int = 256
    content_stride: int = 4
    content_dim: int = 32
    tx_state_dim: int = 16
    channel_dim: int = 16
    receiver_dim: int = 8
    sync_dim: int = 6
    gain_dim: int = 3
    variance_floor: float = 1e-4
    variance_ceiling: float = 1.0

@dataclass
class FCRPairBatch:
    clean_iq: torch.Tensor
    leo_iq: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    receiver_id: torch.Tensor
    day_id: torch.Tensor
    nuisance: torch.Tensor
    nuisance_valid: torch.Tensor
    physical_sample_id: tuple[str, ...]
    pair_id: tuple[str, ...]
    clean_crop_offset: torch.Tensor
    leo_crop_offset: torch.Tensor
    nuisance_pair_index: torch.Tensor
    content_pair_index: torch.Tensor
    fingerprint_pair_index: torch.Tensor
    pair_valid_mask: dict[str, torch.Tensor]

@dataclass
class FCRFactorOutput:
    z_s: torch.Tensor
    z_f_id: torch.Tensor
    z_tx_state: torch.Tensor
    z_n_parts: dict[str, torch.Tensor]
    s_hat: torch.Tensor
    content_confidence: torch.Tensor
    response_coef: Optional[torch.Tensor] = None
    response_quality: Optional[dict[str, torch.Tensor]] = None

@dataclass
class FCRDecodeOutput:
    mu_iq: torch.Tensor
    log_variance: torch.Tensor
    delta_f: torch.Tensor

@dataclass
class FCRLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    metrics: dict[str, float]
```

上述类型是后续任务的唯一共享输出契约。

- [ ] **Step 4: 仅在`use_fcr=True`时保留配置，不实例化尚未实现的模块**

```python
self.use_fcr = bool(use_fcr)
self.fcr_config = fcr_config if self.use_fcr else None
self.fcr = None
```

- [ ] **Step 5: 运行契约测试**

Run: `python -m pytest code/tests/test_phase1_fcr_model_contract.py -v`

Expected: PASS。

- [ ] **Step 6: 更新追踪表并提交**

将FCR-25更新为`implemented`；关闭态测试通过后更新为`verified`。

```text
git add code/cvsrffi/phase1_fcr_types.py code/model_dual_cvsincnet.py code/tests/test_phase1_fcr_model_contract.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-contract
```

### Task 2: 建立同步clean/LEO元数据和三轴干预能力审计

**Files:**
- Modify: `code/dataset_wisig.py:100-243,286-306`
- Modify: `code/baseline_origin_sat_view.py:44-74,276-356`
- Modify: `code/cvsrffi/tensors.py:24-44`
- Create: `code/cvsrffi/phase1_fcr_interventions.py`
- Create: `code/scripts/audit_phase1_fcr_interventions.py`
- Test: `code/tests/test_phase1_fcr_pairing.py`
- Test: `code/tests/test_phase1_fcr_interventions.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: WiSig`meta={base_index,tx_i,rx_i,day_i,eq_i,sig_i}`、`SatViewTransform`和`FCRConfig`。
- Produces: `sanitize_fcr_meta(meta, label_visible)`、`build_physical_sample_id(meta)`、`InterventionCapability`和`InterventionCubeBatchBuilder.build(...) -> FCRPairBatch`。

- [ ] **Step 1: 写无标签真值不可达和同步配对失败测试**

```python
def test_unlabeled_role_does_not_expose_true_tx():
    _, y, _, meta = hidden_role_dataset[0]
    assert y == -1
    assert "true_tx_i" not in meta

def test_clean_leo_pair_keeps_physical_id_and_crop():
    pair = builder.build(clean, leo_view, y, d, meta)
    assert pair.clean_iq.shape == pair.leo_iq.shape
    assert pair.pair_id == pair.physical_sample_id
    torch.testing.assert_close(pair.clean_crop_offset, pair.leo_crop_offset)
```

- [ ] **Step 2: 运行新测试并确认当前`true_tx_i`泄漏和pair字段缺失**

Run: `python -m pytest code/tests/test_phase1_fcr_pairing.py -v`

Expected: FAIL；当前`WiSigRoleDataset`把`true_tx_i`写入隐藏标签meta，`SatViewTransform`也没有pair字段。

- [ ] **Step 3: 从无标签角色删除真实TX字段并补齐稳定ID**

```python
physical_sample_id = (
    f"tx{it.tx_i}:rx{it.rx_i}:day{it.day_i}:eq{it.eq_i}:sig{it.sig_i}"
)
meta.update({
    "physical_sample_id": physical_sample_id,
    "crop_offset": int(crop_offset),
    "label_visible": bool(y >= 0),
})
if not self.tx_label_visible:
    y = -1
```

不得再写入`meta["true_tx_i"]`。

- [ ] **Step 4: 扩展卫星视图返回配对元数据但保持旧调用兼容**

```python
@dataclass
class SatViewTransform:
    # existing fields remain unchanged
    pair_id: Optional[tuple[str, ...]] = None
    physical_sample_id: Optional[tuple[str, ...]] = None
    crop_offset: Optional[torch.Tensor] = None
```

`transform(..., batch_meta=None)`新增可选参数；未传meta时保持当前返回语义。传入meta时LEO视图继承clean的ID和crop，不二次裁剪。

- [ ] **Step 5: 实现三类pair索引和明确失效行为**

```python
@dataclass(frozen=True)
class InterventionCapability:
    nuisance_pair: bool
    content_pair: bool
    fingerprint_pair: bool
    reason: dict[str, str]

def invalid_indices(batch_size: int, device) -> torch.Tensor:
    return torch.full((batch_size,), -1, dtype=torch.long, device=device)
```

Nuisance Pair固定为clean/LEO同位置；Content Pair只使用同一物理记录的不同有效窗口；Fingerprint Pair只使用显式公共preamble区间且匹配receiver/day/view/激励bin。条件不足时返回`-1`和`valid=false`，不能回退到随机pair。

- [ ] **Step 6: 实现能力审计脚本**

脚本只读WiSig索引和样本长度，输出JSON中的`common_preamble_configured/content_window_pairs/fingerprint_pair_candidates/reasons`。当公共preamble未配置或原始样本无第二窗口时，对应能力为false，退出码仍为0，因为这是科学能力结论，不是命令故障。

- [ ] **Step 7: 运行聚焦测试和只读审计**

Run:

```text
python -m pytest code/tests/test_phase1_fcr_pairing.py code/tests/test_phase1_fcr_interventions.py code/tests/test_baseline_origin_sat_view.py -v
python code/scripts/audit_phase1_fcr_interventions.py --dataset wisig --output local_artifacts/fcr_intervention_audit.json
```

Expected: 测试PASS；审计明确报告三类能力，不生成训练数据。

- [ ] **Step 8: 按审计证据更新追踪表并提交**

FCR-01通过同步pair测试后为`verified`。FCR-13只有三类pair均有真实证据才为`verified`；缺少Fingerprint Pair时标为`blocked`并保留原因。

```text
git add code/dataset_wisig.py code/baseline_origin_sat_view.py code/cvsrffi/tensors.py code/cvsrffi/phase1_fcr_interventions.py code/scripts/audit_phase1_fcr_interventions.py code/tests/test_phase1_fcr_pairing.py code/tests/test_phase1_fcr_interventions.py code/tests/test_baseline_origin_sat_view.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-intervention-pairs
```

### Task 3: 实现保守Canonicalizer与时序内容因子

**Files:**
- Create: `code/cvsrffi/phase1_fcr_canonicalizer.py`
- Create: `code/cvsrffi/phase1_fcr_factors.py`
- Test: `code/tests/test_phase1_fcr_canonicalizer.py`
- Test: `code/tests/test_phase1_fcr_content.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: `iq: FloatTensor[B,2,256]`和`FCRConfig`。
- Produces: `CanonicalOutput(canonical_iq,eta_hat,residual_iq,quality)`；`ContentOutput(z_s,s_hat,content_confidence)`，其中`z_s:[B,64,32]`、`s_hat:[B,256] complex64`。

- [ ] **Step 1: 写粗nuisance恢复和细残差保持失败测试**

```python
def test_canonicalizer_removes_common_cfo_phase_gain_only():
    perturbed = apply_common_gain_phase_cfo(clean, gain=1.4, phase=0.5, cfo_norm=0.03)
    out = canonicalizer(perturbed)
    assert complex_nmse(out.canonical_iq, clean) < complex_nmse(perturbed, clean)
    assert out.eta_hat.shape == (clean.size(0), 3)

def test_tx_residual_is_not_zeroed():
    out = canonicalizer(clean_with_iq_imbalance)
    assert out.residual_iq.square().mean() > 0
```

- [ ] **Step 2: 实现解析逆变换**

```python
phase = phase0[:, None] + omega[:, None] * sample_index[None, :]
canonical = complex_iq * torch.exp(-1j * phase) / gain[:, None].clamp_min(1e-4)
residual = complex_iq - canonical
```

Canonicalizer只输出`log_gain/phase0/omega`，不含自由FIR、共轭IQ消除或高容量网络。

- [ ] **Step 3: 写时序token、masked prediction和梯度隔离失败测试**

```python
content = content_model(canonical_iq)
assert content.z_s.shape == (batch, 64, 32)
assert content.s_hat.shape == (batch, 256)
assert torch.all((content.content_confidence >= 0) & (content.content_confidence <= 1))
tx_head = nn.Linear(32, num_classes)
tx_ce = F.cross_entropy(tx_head(content.z_s.detach().mean(dim=1)), labels)
tx_ce.backward()
assert all(p.grad is None for p in content_model.parameters())
```

- [ ] **Step 4: 实现`ContentSequenceEncoder`与`ContentGenerator`**

编码器用stride-4局部卷积产生64个token，不做全局池化；生成器用受限上采样恢复复数`\hat s`。暴露`detach_identity_input=True`，使TX CE默认不能进入内容分支；masked reconstruction仍可反向传播。

- [ ] **Step 5: 运行聚焦测试**

Run: `python -m pytest code/tests/test_phase1_fcr_canonicalizer.py code/tests/test_phase1_fcr_content.py -v`

Expected: PASS，输出有限且复数/实数形状一致。

- [ ] **Step 6: 更新FCR-03/FCR-15并提交**

```text
git add code/cvsrffi/phase1_fcr_canonicalizer.py code/cvsrffi/phase1_fcr_factors.py code/tests/test_phase1_fcr_canonicalizer.py code/tests/test_phase1_fcr_content.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-content-factors
```

### Task 4: 实现激励条件化发射机响应算子

**Files:**
- Modify: `code/cvsrffi/phase1_fcr_factors.py`
- Create: `code/cvsrffi/phase1_fcr_fingerprint.py`
- Test: `code/tests/test_phase1_fcr_fingerprint.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: `id_feature_raw:[B,160]`、`canonical_iq:[B,2,256]`、`residual_iq:[B,2,256]`、`s_hat:[B,256] complex64`。
- Produces: `z_f_id:[B,160]`、`z_tx_state:[B,16]`、`delta_f:[B,256] complex64`、`response_coef`和`response_quality`。

- [ ] **Step 1: 写身份/状态拆分、相位等变和残差容量失败测试**

```python
factor = encoder(id_feature_raw, canonical_iq, residual_iq, excitation)
assert factor.z_f_id.shape == (batch, 160)
assert factor.z_tx_state.shape == (batch, 16)
phase_rot = torch.exp(1j * torch.tensor(0.4))
rotated = operator(s_hat * phase_rot, factor)
base = operator(s_hat, factor) * phase_rot
torch.testing.assert_close(rotated.delta_f, base.delta_f, atol=1e-4, rtol=1e-4)
assert (rotated.delta_f.norm(dim=1) / s_hat.norm(dim=1)).max() <= operator.residual_ratio_max + 1e-5
```

- [ ] **Step 2: 实现`excitation_features`和固定物理基**

```python
def excitation_features(s: torch.Tensor) -> torch.Tensor:
    amp = s.abs()
    slew = torch.diff(amp, dim=-1, prepend=amp[..., :1])
    return torch.stack((amp, amp.square(), amp.pow(3), slew), dim=-1)

def fixed_response_basis(s: torch.Tensor) -> torch.Tensor:
    delayed = torch.roll(s, shifts=1, dims=-1)
    return torch.stack((s, s.conj(), s * s.abs().square(), delayed * delayed.abs().square()), dim=-1)
```

完整实现增加报告要求的PA直接项、IQ共轭项、memory项和slew项；basis是固定函数，不能变成自由MLP。

- [ ] **Step 3: 实现`FingerprintFactorEncoder`**

现有ADV3B02身份主干作为`E_f`的底层特征抽取器。编码器将`id_feature_raw`与规范化残差/激励摘要组合，输出单位化`z_f_id`和独立`z_tx_state`；跨天一致性只作用于`z_f_id`。

- [ ] **Step 4: 实现受限`ExcitationConditionedFingerprintOperator`**

```python
delta_physical = torch.einsum("btn,bn->bt", basis, response_coef)
delta_small = bounded_residual(excitation, z_tx_state)
delta = limit_energy_and_bandwidth(delta_physical + delta_small, s_hat, residual_ratio_max)
```

`bounded_residual`使用低rank、短感受野网络，不能直接读取原始IQ。

- [ ] **Step 5: 运行测试并更新追踪表**

Run: `python -m pytest code/tests/test_phase1_fcr_fingerprint.py -v`

Expected: PASS；FCR-04/FCR-05为`verified`。

- [ ] **Step 6: 提交**

```text
git add code/cvsrffi/phase1_fcr_factors.py code/cvsrffi/phase1_fcr_fingerprint.py code/tests/test_phase1_fcr_fingerprint.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-fingerprint-operator
```

### Task 5: 实现结构化nuisance和物理顺序概率Decoder

**Files:**
- Create: `code/cvsrffi/phase1_fcr_nuisance.py`
- Create: `code/cvsrffi/phase1_fcr_decoder.py`
- Test: `code/tests/test_phase1_fcr_nuisance.py`
- Test: `code/tests/test_phase1_fcr_decoder.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: `x:[B,2,256]`、`eta_hat:[B,3]`、`s_hat:[B,256] complex64`、`delta_f:[B,256] complex64`。
- Produces: `NuisanceOutput(z_ch,z_rx,z_sync,z_gain,eta_pred)`和`FCRDecodeOutput(mu_iq,log_variance,delta_f)`。

- [ ] **Step 1: 写低容量、无skip和TX不可预测失败测试**

```python
out = nuisance_encoder(x, eta_hat)
assert out.z_ch.shape == (batch, 16)
assert out.z_rx.shape == (batch, 8)
assert out.z_sync.shape == (batch, 6)
assert out.z_gain.shape == (batch, 3)
latent_size = sum(v[0].numel() for v in (out.z_ch, out.z_rx, out.z_sync, out.z_gain))
assert latent_size < x[0].numel()
assert not any("skip" in name for name, _ in nuisance_encoder.named_modules())
```

- [ ] **Step 2: 实现结构化参数头**

`z_sync`显式预测公共相位、CFO、Doppler rate、STO和SFO；`z_gain`预测AGC/幅度；`z_ch`产生短低rank通道响应；`z_rx`只产生受限RX residual。所有头共享低容量统计编码器，不返回时序latent。

- [ ] **Step 3: 写Decoder调用顺序和方差逃逸失败测试**

```python
decoded = decoder(content.s_hat, fingerprint.delta_f, nuisance)
assert decoded.mu_iq.shape == (batch, 2, 256)
variance = decoded.log_variance.exp()
assert variance.min() >= config.variance_floor
assert variance.max() <= config.variance_ceiling
assert decoder.call_trace == ("content", "fingerprint", "channel_receiver")
```

- [ ] **Step 4: 实现`PhysicsOrderedDecoder`**

```python
u_hat = s_hat + delta_f
linked = apply_short_channel(u_hat, nuisance.z_ch)
linked = apply_rx_residual(linked, nuisance.z_rx)
mu = apply_sync_and_gain(linked, nuisance.z_sync, nuisance.z_gain)
log_variance = bounded_variance_head(nuisance)
```

Decoder不接收原始`x`，因此目标波形不能绕过latent路径。

- [ ] **Step 5: 运行聚焦测试并更新FCR-02/FCR-06/FCR-07**

Run: `python -m pytest code/tests/test_phase1_fcr_nuisance.py code/tests/test_phase1_fcr_decoder.py -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```text
git add code/cvsrffi/phase1_fcr_nuisance.py code/cvsrffi/phase1_fcr_decoder.py code/tests/test_phase1_fcr_nuisance.py code/tests/test_phase1_fcr_decoder.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-physical-decoder
```

### Task 6: 实现冻结物理特征、Fisher门控和重构误差

**Files:**
- Create: `code/cvsrffi/phase1_fcr_physics.py`
- Create: `code/cvsrffi/phase1_fcr_losses.py`
- Test: `code/tests/test_phase1_fcr_physics.py`
- Test: `code/tests/test_phase1_fcr_reconstruction_losses.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: 真实IQ、`FCRDecodeOutput`、`\hat s`、响应基Gram矩阵和SNR/激励统计。
- Produces: `FingerprintFeatures`、`FisherGateOutput(block_weights,quality)`、`heteroscedastic_complex_nll`、`mrstft_loss`、`phase_increment_loss`和`physical_feature_loss`。

- [ ] **Step 1: 写冻结特征和低激励门控失败测试**

```python
bank = FrozenFingerprintFeatureBank()
assert sum(p.numel() for p in bank.parameters()) == 0
low_papr_gate = gate(low_papr_signal, gram, snr_db)
high_papr_gate = gate(high_papr_signal, gram, snr_db)
assert low_papr_gate.block_weights["pa"] < high_papr_gate.block_weights["pa"]
```

- [ ] **Step 2: 实现固定`R_fp`和Fisher门控**

特征集合固定包含IQ非圆性/椭圆、AM/AM、AM/PM、memory residual、谱肩、局部相位噪声PSD、幅度条件残差和循环平稳摘要。门控使用Gram block effective rank、激励覆盖、PAPR、SNR和噪声地板；所有gate质量输入stop-gradient。

- [ ] **Step 3: 写NLL、MRSTFT和相位环绕失败测试**

```python
assert heteroscedastic_complex_nll(target, perfect_mu, bounded_logvar) < nll_bad
assert mrstft_loss(target, target) < 1e-6
wrapped_a = torch.polar(torch.ones(1, 32), torch.full((1, 32), math.pi - 1e-3))
wrapped_b = torch.polar(torch.ones(1, 32), torch.full((1, 32), -math.pi + 1e-3))
assert phase_increment_loss(wrapped_a, wrapped_b) < 1e-2
```

- [ ] **Step 4: 实现四类重构项**

`heteroscedastic_complex_nll`在方差上下界内计算；MRSTFT使用三组窗口并为低能量bin设置噪声地板；相位项比较归一化共轭乘积并用目标幅度加权；物理特征项逐block乘Fisher gate。

- [ ] **Step 5: 运行测试并更新FCR-08/FCR-09/FCR-19**

Run: `python -m pytest code/tests/test_phase1_fcr_physics.py code/tests/test_phase1_fcr_reconstruction_losses.py -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```text
git add code/cvsrffi/phase1_fcr_physics.py code/cvsrffi/phase1_fcr_losses.py code/tests/test_phase1_fcr_physics.py code/tests/test_phase1_fcr_reconstruction_losses.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-physical-losses
```

### Task 7: 实现shared、swap、latent-cycle、nuisance监督和防塌缩

**Files:**
- Modify: `code/cvsrffi/phase1_fcr_losses.py`
- Test: `code/tests/test_phase1_fcr_cross_losses.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: clean/LEO两套`FCRFactorOutput`、交叉`FCRDecodeOutput`、`FCRPairBatch`。
- Produces: `self_loss`、`swap_loss`、`shared_loss`、`latent_cycle_loss`、`eta_loss`、`factor_loss`及逐项metrics。

- [ ] **Step 1: 写双向swap和cycle来源绑定失败测试**

```python
losses = compute_cross_losses(clean_factors, leo_factors, clean_to_leo, leo_to_clean, pair)
assert set(losses) >= {"self", "swap", "shared", "latent_cycle", "eta", "factor"}
assert losses["swap_clean_to_leo"].requires_grad
assert losses["swap_leo_to_clean"].requires_grad
```

- [ ] **Step 2: 实现共享一致性和防塌缩**

```python
shared = symmetric_stopgrad_distance(clean.z_s, leo.z_s)
shared = shared + symmetric_stopgrad_distance(clean.z_f_id, leo.z_f_id)
anti_collapse = variance_floor_loss(z) + off_diagonal_covariance_loss(z)
```

`z_s`按token比较，`z_f_id`按单位球特征比较；常数表征负测必须产生正损失。

- [ ] **Step 3: 实现latent cross-cycle**

重新编码clean→LEO生成结果，恢复clean的`z_s/z_f`和LEO的`z_n`；LEO→clean反向同理。所有目标latent使用`detach()`。

- [ ] **Step 4: 实现nuisance参数监督和因子泄漏抑制**

`L_eta`只在`nuisance_valid`上回归已知字段。`L_factor`组合cross-covariance、条件domain混淆和训练外probe接口；不直接最大化clean/LEO`z_n`距离。

- [ ] **Step 5: 运行测试并更新FCR-10/FCR-11/FCR-12/FCR-16/FCR-17**

Run: `python -m pytest code/tests/test_phase1_fcr_cross_losses.py -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```text
git add code/cvsrffi/phase1_fcr_losses.py code/tests/test_phase1_fcr_cross_losses.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-cross-cycle-losses
```

### Task 8: 实现定向指纹移植和改进necessity

**Files:**
- Create: `code/cvsrffi/phase1_fcr_transplant.py`
- Modify: `code/cvsrffi/phase1_fcr_losses.py`
- Test: `code/tests/test_phase1_fcr_transplant.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: `FCRPairBatch.fingerprint_pair_index`、源/目标`FCRFactorOutput`、冻结身份判别器和FCR重编码函数。
- Produces: `TransplantOutput(iq,target_logits,reencoded)`和`L_target_id/L_preserve_s/L_preserve_n/L_same_f/L_drop_f`。

- [ ] **Step 1: 写无有效pair零损失和三角验证失败测试**

```python
empty = transplant_loss(pair_with_no_fingerprint_pairs, factors, frozen_classifier)
assert empty.active_pairs == 0
assert empty.total.item() == 0.0

valid = transplant_loss(pair_with_valid_pairs, factors, frozen_classifier)
assert set(valid.components) == {"target_id", "preserve_s", "preserve_n", "same_f", "drop_f"}
```

- [ ] **Step 2: 实现独立冻结身份分类器约束**

冻结分类器参数且不与Decoder共同更新；跨TX输出必须预测目标TX，同TX交换必须保持原身份。分类器读取生成IQ的正常ADV3B02身份forward，不读取移植pair真值以外的信息。

- [ ] **Step 3: 实现内容、nuisance和指纹重编码保持**

```python
transplanted = decode(source.z_s, target.z_f, source.z_n)
re = encode(transplanted.mu_iq)
loss_preserve_s = distance(re.z_s, source.z_s.detach())
loss_preserve_n = distance(re.z_n, source.z_n.detach())
loss_target_f = distance(re.z_f_id, target.z_f_id.detach())
```

- [ ] **Step 4: 实现drop-f和Decoder防作弊路由**

drop-f使用零向量或batch均值替换`z_f`，要求Fisher门控后的指纹残差误差增加。necessity阶段提供`freeze_decoder=True`，只更新`E_f/G_f`；正确路径误差作为stop-gradient参考。

- [ ] **Step 5: 运行测试并更新FCR-18**

Run: `python -m pytest code/tests/test_phase1_fcr_transplant.py -v`

Expected: PASS；随机shuffle gap不在验收条件中。

- [ ] **Step 6: 提交**

```text
git add code/cvsrffi/phase1_fcr_transplant.py code/cvsrffi/phase1_fcr_losses.py code/tests/test_phase1_fcr_transplant.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-directed-transplant
```

### Task 9: 将完整FCR接入ADV3B02模型并定义输出schema

**Files:**
- Modify: `code/model_dual_cvsincnet.py:507-1070`
- Modify: `code/cvsrffi/phase1_fcr_types.py`
- Test: `code/tests/test_phase1_fcr_model_contract.py`
- Test: `code/tests/test_phase1_fcr_forward.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: Tasks3-8的模块；现有`id_feature_raw=z_id`和原始IQ。
- Produces: `ADV3B02FactorizedCrossReconstruction.forward(x, pair_context=None, return_aux=True)`及模型输出`z_id_raw/z_f_id/z_tx_state/z_s/z_n/fcr_decode/fcr_quality`。

- [ ] **Step 1: 写开启态输出、单视图推理和关闭态回归测试**

```python
model = build_dual_model(num_classes=6, num_domains=5, use_fcr=True)
out = model(torch.randn(2, 2, 256), return_aux=True)
assert out["z_id_raw"].shape == (2, 160)
assert out["z_f_id"].shape == (2, 160)
assert out["z_s"].shape[:2] == (2, 64)
assert out["feature_schema"] == "ADV3B02:FCR:z_f_id:unit_l2:160:v1"
assert "clean_companion" not in inspect.signature(model.forward).parameters
```

- [ ] **Step 2: 实现FCR聚合模块**

```python
class ADV3B02FactorizedCrossReconstruction(nn.Module):
    def forward(self, x, id_feature_raw, *, pair_context=None):
        canonical = self.canonicalizer(x)
        content = self.content(canonical.canonical_iq)
        fingerprint = self.fingerprint(id_feature_raw, canonical, content)
        nuisance = self.nuisance(x, canonical.eta_hat)
        decoded = self.decoder(content.s_hat, fingerprint.delta_f, nuisance)
        return {
            "canonical": canonical,
            "content": content,
            "fingerprint": fingerprint,
            "nuisance": nuisance,
            "decoded": decoded,
        }
```

- [ ] **Step 3: 在`DualCVSincNetDisentangle`中只做可选接线**

`self.fcr=ADV3B02FactorizedCrossReconstruction(...) if use_fcr else None`。开启时保留旧`z_id`别名不变，新增`z_f_id`供候选feature key显式选择；不得静默用`z_f_id`覆盖旧`z_id`。

- [ ] **Step 4: 验证forward/backward有限值和关闭态逐元素一致**

Run: `python -m pytest code/tests/test_phase1_fcr_model_contract.py code/tests/test_phase1_fcr_forward.py -v`

Expected: PASS。

- [ ] **Step 5: 更新FCR-02至FCR-07的端到端可达状态并提交**

```text
git add code/model_dual_cvsincnet.py code/cvsrffi/phase1_fcr_types.py code/tests/test_phase1_fcr_model_contract.py code/tests/test_phase1_fcr_forward.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:wire-FCR-into-ADV3B02
```

### Task 10: 接入四阶段训练、数据权限和完整总损失

**Files:**
- Create: `code/cvsrffi/phase1_fcr_schedule.py`
- Modify: `code/train.py:678-790,2451-2719,3000-3691`
- Modify: `code/cvsrffi/schedule.py:299-307`
- Test: `code/tests/test_phase1_fcr_schedule.py`
- Test: `code/tests/test_phase1_fcr_gradient_routing.py`
- Test: `code/tests/test_phase1_fcr_unlabeled_boundary.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: epoch、`FCRPairBatch`、clean/LEO forward结果和命令行lambda。
- Produces: `FCRStageState`、参数冻结状态、`FCRLossOutput`及训练日志字段。

- [ ] **Step 1: 写E1-200阶段矩阵失败测试**

```python
assert stage_for_epoch(1).active == {"id", "self", "eta"}
assert {"swap", "shared", "latent_cycle"} <= stage_for_epoch(41).active
assert {"transplant", "intervention"} <= stage_for_epoch(91).active
assert stage_for_epoch(151).reconstruction_scale < stage_for_epoch(90).reconstruction_scale
```

- [ ] **Step 2: 实现独立FCR日程**

```python
@dataclass(frozen=True)
class FCRStageState:
    name: str
    active: frozenset[str]
    scales: dict[str, float]
    freeze_decoder_for_necessity: bool
```

E41-90对swap/shared/cycle线性ramp；E91-150按optimizer step交替真实组合与necessity更新；E151-200降低raw reconstruction scale。

- [ ] **Step 3: 写`L_s/U_s/V`梯度边界失败测试**

```python
assert unlabeled_pair.labels.eq(-1).all()
assert unlabeled_losses["id"].item() == 0.0
assert unlabeled_losses["transplant"].item() == 0.0
assert unlabeled_losses["swap"].requires_grad
assert validation_step.updated_persistent_state is False
```

- [ ] **Step 4: 增加显式CLI并阻止普通ADV3B02误启用**

增加`--phase1_method adv3b02_fcr`、`--use_fcr`和八个lambda。解析规则要求只有`phase1_method=adv3b02_fcr`才能设置`use_fcr=true`；普通ADV3B02继续保持所有FCR lambda为0。

- [ ] **Step 5: 在训练循环接入pair forward和总损失**

```python
loss_total = loss_adv3b02
if model_raw.use_fcr:
    fcr_out = compute_fcr_losses(clean_out, leo_out, pair_batch, stage_state)
    loss_total = loss_total + fcr_out.total
```

所有需标签项使用`label_mask`；`U_s`只允许self/swap/shared/cycle/eta/phys。卫星辅助CE仍从E80开始，FCR日程不修改原`lambda_sat_cls`。

- [ ] **Step 6: 运行聚焦测试**

Run:

```text
python -m pytest code/tests/test_phase1_fcr_schedule.py code/tests/test_phase1_fcr_gradient_routing.py code/tests/test_phase1_fcr_unlabeled_boundary.py -v
```

Expected: PASS。

- [ ] **Step 7: 更新FCR-14/FCR-16/FCR-20/FCR-21/FCR-24并提交**

```text
git add code/cvsrffi/phase1_fcr_schedule.py code/cvsrffi/schedule.py code/train.py code/tests/test_phase1_fcr_schedule.py code/tests/test_phase1_fcr_gradient_routing.py code/tests/test_phase1_fcr_unlabeled_boundary.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:add-FCR-training-schedule
```

### Task 11: 闭合checkpoint、诊断、正式入口和本地验证

**Files:**
- Create: `code/cvsrffi/phase1_fcr_diagnostics.py`
- Modify: `code/cvsrffi/checkpoint.py:57-75`
- Modify: `code/train.py:3931-4093`
- Create: `code/scripts/launch_phase1_adv3b02_fcr_20260901.sh`
- Create: `code/tests/test_phase1_fcr_checkpoint.py`
- Create: `code/tests/test_phase1_fcr_diagnostics.py`
- Create: `code/tests/test_phase1_adv3b02_fcr_launcher.py`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: 完整FCR模型、训练metrics、现有`save_checkpoint(...)`和Phase1评测器。
- Produces: `fcr_bundle`、`ADV3B02:FCR:z_f_id:unit_l2:160:v1`、独立probe指标、R0-R8入口和四种最终评测配置。

- [ ] **Step 1: 写checkpoint往返和单LEO推理失败测试**

```python
save_checkpoint(path, model=model, optimizer=None, scheduler=None, scaler=None,
                epoch=3, args=args, split_info={}, stats={})
payload = torch.load(path, map_location="cpu")
assert payload["fcr_bundle"]["feature_schema"] == "ADV3B02:FCR:z_f_id:unit_l2:160:v1"
loaded = build_dual_model(num_classes=6, num_domains=5, use_fcr=True)
loaded.load_state_dict(payload["model"], strict=True)
loaded.eval()
torch.testing.assert_close(loaded(leo_iq, return_aux=True)["z_f_id"], reference_z_f)
```

- [ ] **Step 2: 扩展checkpoint payload**

```python
payload["fcr_bundle"] = state_model.export_fcr_bundle() if state_model.use_fcr else None
```

bundle保存FCR配置、模块权重、固定物理基版本、normalization、Fisher gate参数、nuisance schema和feature schema。旧checkpoint缺少`fcr_bundle`时仍可严格加载`use_fcr=false`模型。

- [ ] **Step 3: 写独立probe和诊断字段失败测试**

```python
required = {
    "zf_tx_probe", "zf_domain_probe", "zn_domain_probe", "zn_tx_probe",
    "zs_content_probe", "clean_leo_zf_distance", "same_tx_zf_distance",
    "drop_f_residual_gap", "transplant_target_id", "transplant_preserve_s",
    "transplant_preserve_n", "gram_condition", "effective_rank",
    "fisher_coverage", "train_time_s", "peak_vram_mb", "latency_ms",
}
assert required <= set(metrics)
```

- [ ] **Step 4: 实现训练外probe和诊断聚合**

probe只读取detach后的冻结artifact，不向模型反向传播。无法严格构造的pair指标写`N/A`并记录原因，不写0。

- [ ] **Step 5: 写launcher dry-run测试并创建R0-R8入口**

launcher固定`ADV3B02`主干、`phase1_method=adv3b02_fcr`、E200、三段LEO_WEAK日程、E80卫星CE和clean/三场景最终评测。R0-R8通过显式`--fcr_ablation_row`选择，不能隐式跳级。

- [ ] **Step 6: 运行全部FCR聚焦测试**

Run:

```text
python -m pytest code/tests/test_phase1_fcr_*.py code/tests/test_phase1_adv3b02_fcr_launcher.py code/tests/test_baseline_origin_sat_view.py -v
```

Expected: PASS。

- [ ] **Step 7: 运行一次真实checkpoint无query smoke**

在`ssr-gpu`中加载真实ADV3B02 checkpoint和一个source clean/LEO配对batch，执行forward、backward、checkpoint保存—加载和单LEO推理。smoke不得连接Phase2 query、target标签或truth scorer。

- [ ] **Step 8: 执行唯一一次独立P0/P1审查**

审查范围只包括会导致真实实验跑错、越权、覆盖输出、误杀进程、不能启动或不能产生合法prediction的问题。若发现P0/P1，修复后只对原问题定点复审一次；P2不阻断。

- [ ] **Step 9: 更新全部实现型追踪项并提交推送**

FCR-22/FCR-23/FCR-26只有在诊断、launcher和checkpoint往返均通过后标为`verified`。科学结果型条目保持`implemented`，直到真实实验返回证据。

```text
git add code/cvsrffi/phase1_fcr_diagnostics.py code/cvsrffi/checkpoint.py code/train.py code/scripts/launch_phase1_adv3b02_fcr_20260901.sh code/tests/test_phase1_fcr_checkpoint.py code/tests/test_phase1_fcr_diagnostics.py code/tests/test_phase1_adv3b02_fcr_launcher.py docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md
git commit -m feat:close-FCR-local-implementation
```

- [ ] **Step 10: 独立核对远端OID**

```text
git rev-parse HEAD
git ls-remote origin refs/heads/work/cvs-active
```

Expected: 两个完整OID相同；不同则发布状态为`FAILED`或`UNKNOWN`，保留本地提交，禁止force-push。

### Task 12: 按最小实验流程发布首个N607证伪矩阵

**Files:**
- Create: `automation_reports/CV-SincNet/<immutable-run-id>/report.md`
- Create: `release_archives/<immutable-release>.tar.gz`
- Modify: `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`

**Interfaces:**
- Consumes: 已推送且远端OID一致的实现提交、R0-R8冻结矩阵、真实checkpoint smoke和一次P0/P1审查结果。
- Produces: 单seed、source-only、同row的R0-R8最小可证伪证据；不自动扩大到多seed或完整125。

- [ ] **Step 1: 建立最小预登记报告**

只写候选/矩阵、Git commit、命令、环境/CWD、输入输出路径、GPU、直接技术停止规则和预期artifact。不得加入额外hash、seal、receipt或审批。

- [ ] **Step 2: 同步Git实现到运行副本并创建单一release归档**

只同步本次已提交文件，读回后打包。release只做一次本地到远端SHA比较，不计算成员SHA。

- [ ] **Step 3: 执行N607直接SSH preflight和远端编译**

先运行`tools\n607_ssh_preflight.ps1`规定的只读preflight；直接N607失败且仅TCP路径不可用时才走已验证lab bridge。远端命令保持短时、离散，任务完成即断开。

- [ ] **Step 4: 启动R0-R8单seed矩阵并做一次归属检查**

核对PID、CWD、cmdline、GPU、run root和日志增长。低性能不得停止；只按预注册协议/安全/执行故障规则停止精确run-owned进程树。

- [ ] **Step 5: 完成Phase1四种评测和独立评分**

每个完成训练的row分别保存clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`指标。prediction完整后独立scorer连接truth，同row报告身份性能、latent诊断、指纹移植、floor和资源成本。

- [ ] **Step 6: 更新追踪表和正式报告并提交推送**

只有真实artifact支持的科学条目转为`verified`。Fingerprint Pair能力为`blocked`时，报告必须收缩声明，不得写成完整三轴可辨识。

## Coverage Map

| Tasks | Traceability IDs |
|---|---|
| Task1 | FCR-25 |
| Task2 | FCR-01、FCR-13、FCR-20 |
| Task3 | FCR-03、FCR-15 |
| Task4 | FCR-04、FCR-05 |
| Task5 | FCR-02、FCR-06、FCR-07 |
| Task6 | FCR-08、FCR-09、FCR-19 |
| Task7 | FCR-10、FCR-11、FCR-12、FCR-16、FCR-17 |
| Task8 | FCR-18 |
| Task9 | FCR-02至FCR-07的端到端模型可达性、FCR-25 |
| Task10 | FCR-14、FCR-16、FCR-20、FCR-21、FCR-24 |
| Task11 | FCR-22、FCR-23、FCR-25、FCR-26 |
| Task12 | FCR-22、FCR-23、FCR-24的真实实验验证 |

## Self-review

- Spec coverage: FCR-01至FCR-26均映射到Task1-Task12；数据配对、三因子、物理Decoder、噪声统计、完整损失、训练、诊断、checkpoint和N607证据均有可达路径。
- Placeholder scan: 没有未决占位或空泛的“补测试/补错误处理”步骤；每个代码任务包含具体接口、失败测试、实现动作、验证命令和提交文件。
- Type consistency: `FCRPairBatch`贯穿Task2、7、8、10；`FCRFactorOutput`贯穿Task3-10；`FCRDecodeOutput`贯穿Task5-10；`z_f_id`固定为ADV3B02身份维度160，feature schema保持一致。
- Scope: 各模块顺序依赖同一训练路径，不构成可独立交付的多个项目，因此保留为一个计划。Task12是实现后的证据阶段，不会提前启动。
- Strictness: 目标是当前报告的严格设计一致，不是ECRS局部近似。无法构造的Fingerprint Pair必须进入`blocked`并收缩声明。
