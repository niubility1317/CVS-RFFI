# ADV3B02-ECRS-V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 严格按设计稿把`ADV3B02-ECRS-V1`作为并行局部系统辨识器接入现有ADV3B02，并保持关闭ECRS时旧模型完全兼容。

**Architecture:** 现有ADV3B02主干继续产生160维`z_id_raw`；并行ECRS路径按`NuisanceEstimator→AnalyticCanonicalizer→ContentEstimator→固定复数响应基→可微加权岭回归→固定锚点编码`产生64维`z_resp`，再以`rho_max=0.25`的质量门控残差投影回160维并单位化。训练严格复用clean+LEO配对与`L_s+U_s`边界，推理只需单条LEO IQ。

**Tech Stack:** Python、PyTorch、complex64线性代数、pytest、现有CV-SincNet训练器与WiSig数据管线。

**Spec:** `E:\codex\home\attachments\8053de7d-3bda-4fd5-bd38-a5a186bad7d0\pasted-text.txt`

## Global Constraints

- 唯一方法规格是上述设计稿及`docs/CVS_PHASE1_ADV3B02_ECRS_V1_TRACE_20260901.md`；不得新增设计稿之外的识别分支、自由MLP响应基或query适配。
- `use_ecrs=false`时不得实例化ECRS参数，旧ADV3B02`state_dict`键、logits、`z_id`和严格checkpoint加载必须逐项兼容。
- V1保留原PA分支、现有160维身份主干和clean+LEO训练，不允许以响应分支替换主干。
- Canonicalizer只执行CFO、公共相位和标量增益解析归一化；V1禁止自由RX-IQ纠正与高容量FIR。
- 固定复数相位等变样条基、有效响应维度`K=28`、response embedding为64维、`rho_max=0.25`、岭回归使用complex64。
- `L_s/U_s/V=0.07/0.63/0.30`；`U_s`不得读取TX真值，`V`不得反向传播或更新持久状态。
- 继续使用`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`；E1–40为`leo_clear_weak,p=0.30`，E41–90为`leo_low_elev_weak,leo_rain_weak,p=0.60`，E91–200为三场景并集`p=0.80`。
- 最终评测分别保留clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`；不得用聚合场景替代。
- 任何N607动作都在本地Git实现、聚焦验证和一次P0/P1审查后进行；本计划本身不启动实验。

---

### Task 1: 锁定旧ADV3B02兼容基线与ECRS开关

**Files:**
- Modify: `code/model_dual_cvsincnet.py:390`
- Modify: `code/train.py:2109`
- Test: `code/tests/test_ecrs_model_contract.py`

**Interfaces:**
- Consumes: 现有`build_dual_model(...)`和`DualCVSincNetDisentangle.forward(...)`。
- Produces: `use_ecrs: bool=False`及`ecrs_config: Optional[dict]=None`；关闭时行为完全不变。

- [ ] **Step 1: 写失败测试锁定关闭态兼容性**

```python
def test_ecrs_off_preserves_legacy_state_and_outputs():
    legacy = build_dual_model(num_classes=6, num_domains=5, use_ecrs=False)
    candidate = build_dual_model(num_classes=6, num_domains=5, use_ecrs=False)
    candidate.load_state_dict(legacy.state_dict(), strict=True)
    assert not any(key.startswith("ecrs") for key in candidate.state_dict())
```

- [ ] **Step 2: 运行测试并确认当前接口不接受ECRS参数**

Run: `pytest code/tests/test_ecrs_model_contract.py::test_ecrs_off_preserves_legacy_state_and_outputs -v`

Expected: FAIL，原因是`use_ecrs`尚未定义。

- [ ] **Step 3: 仅在`use_ecrs=True`时实例化响应模块**

```python
self.use_ecrs = bool(use_ecrs)
self.ecrs = ResponseSurfaceBranch(...) if self.use_ecrs else None
```

- [ ] **Step 4: 验证旧checkpoint严格加载、关闭态logits和`z_id`一致**

Run: `pytest code/tests/test_ecrs_model_contract.py -v`

Expected: PASS。

- [ ] **Step 5: 只提交本任务文件**

```text
git add code/model_dual_cvsincnet.py code/train.py code/tests/test_ecrs_model_contract.py
git commit -m "feat: add opt-in ECRS model route"
```

### Task 2: 接入同步clean/LEO配对元数据

**Files:**
- Modify: `code/dataset_wisig.py:218`
- Modify: `code/baseline_origin_sat_view.py:13`
- Modify: `code/cvsrffi/tensors.py:18`
- Modify: `code/train.py:678`
- Test: `code/tests/test_ecrs_pair_metadata.py`
- Test: `code/tests/test_baseline_origin_sat_view.py`

**Interfaces:**
- Consumes: WiSig的`base_index/tx_i/rx_i/day_i/eq_i/sig_i`元数据与现有卫星增强器。
- Produces: `physical_sample_id`、`pair_id`、`view_type`、`label_mask`、`receiver_id`、`day_id`、`crop_offset`、`synchronized_crop`、`sat_meta`和clean/sat mask。

- [ ] **Step 1: 写配对身份与同步crop失败测试**

```python
assert clean_meta["pair_id"] == leo_meta["pair_id"]
assert clean_meta["physical_sample_id"] == leo_meta["physical_sample_id"]
assert clean_meta["crop_offset"] == leo_meta["crop_offset"]
assert clean_meta["view_type"] == "clean"
assert leo_meta["view_type"] == "leo"
```

- [ ] **Step 2: 运行现有卫星视图测试和新失败测试**

Run: `pytest code/tests/test_baseline_origin_sat_view.py code/tests/test_ecrs_pair_metadata.py -v`

Expected: 新测试FAIL，旧测试PASS。

- [ ] **Step 3: 从WiSig索引构造稳定物理样本ID并透传meta**

```python
physical_sample_id = f"tx{it.tx_i}:rx{it.rx_i}:day{it.day_i}:eq{it.eq_i}:sig{it.sig_i}"
```

- [ ] **Step 4: 增强器返回配对batch，不改变旧`transform/expand`调用者**

`use_ecrs=True`时增加配对包装；旧路径保持原返回类型。ECRS配对包装用同一已裁剪clean IQ生成LEO视图，因此不产生不同crop。

- [ ] **Step 5: 验证`U_s`的`label_mask=False`且真实TX不可进入训练张量**

Run: `pytest code/tests/test_ecrs_pair_metadata.py -v`

Expected: PASS。

- [ ] **Step 6: 提交本任务**

```text
git add code/dataset_wisig.py code/baseline_origin_sat_view.py code/cvsrffi/tensors.py code/train.py code/tests/test_ecrs_pair_metadata.py code/tests/test_baseline_origin_sat_view.py
git commit -m "feat: preserve paired clean LEO metadata"
```

### Task 3: 实现保守规范化器与内容估计器

**Files:**
- Modify: `code/model_dual_cvsincnet.py:100`
- Test: `code/tests/test_ecrs_canonicalizer.py`

**Interfaces:**
- Consumes: `x: FloatTensor[B,2,256]`。
- Produces: `canonical_iq: FloatTensor[B,2,256]`、`s_hat: ComplexTensor[B,256]`、`content_confidence: FloatTensor[B,256]`、`nuisance_coef: FloatTensor[B,3]`和质量量。

- [ ] **Step 1: 写CFO、相位、增益合成恢复与公共相位等变失败测试**

```python
assert canonical_nmse(clean, canonicalized) < raw_nmse(clean, perturbed)
assert torch.all((content_confidence >= 0) & (content_confidence <= 1))
```

- [ ] **Step 2: 实现`NuisanceEstimator`和`AnalyticCanonicalizer`**

`NuisanceEstimator`只输出归一化CFO、公共相位和log-gain；`AnalyticCanonicalizer`用复数乘法执行逆变换，不包含共轭IQ支路或任意FIR。

- [ ] **Step 3: 实现低容量`ContentEstimator`**

输入、输出均保持复数时序形状；主TX分类梯度默认detach，只接受masked reconstruction与clean/LEO内容一致性。

- [ ] **Step 4: 验证有限值、形状、梯度隔离和单视图运行**

Run: `pytest code/tests/test_ecrs_canonicalizer.py -v`

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```text
git add code/model_dual_cvsincnet.py code/tests/test_ecrs_canonicalizer.py
git commit -m "feat: add conservative ECRS canonicalization"
```

### Task 4: 实现固定响应基、可辨识性与可微岭回归

**Files:**
- Modify: `code/model_dual_cvsincnet.py`
- Test: `code/tests/test_ecrs_weighted_ridge.py`
- Test: `code/tests/test_ecrs_response_basis.py`

**Interfaces:**
- Consumes: `s_hat: ComplexTensor[B,N]`、`canonical_u: ComplexTensor[B,N]`、`content_confidence: FloatTensor[B,N]`。
- Produces: `resp_coef: ComplexTensor[B,28]`、`resp_cov_diag: FloatTensor[B,28]`、`resp_quality: dict`、`ridge_info: IntTensor[B]`。

- [ ] **Step 1: 写固定复数相位等变基测试**

```python
phi = basis(s_hat)
phi_rot = basis(s_hat * torch.exp(1j * psi))
torch.testing.assert_close(phi_rot, phi * torch.exp(1j * psi), rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: 写加权岭已知系数恢复与退化矩阵回退测试**

测试正常Cholesky、10倍岭二次Cholesky、增广QR/`lstsq`三条路径；断言全程不调用`torch.inverse`。

- [ ] **Step 3: 实现固定`ResponseBasis`与受限`NuisanceEstimator`字典**

V1固定有效维度28，显式包含PA直接、IQ共轭、cross-memory和slew项；不创建可学习`U`。

- [ ] **Step 4: 实现逐采样权重与块状可辨识性收缩**

```python
lambda_block = alpha_lambda * gram_trace_over_k / (q_block + eps)
weight = weight.clamp_min(0.05)
weight = weight / (weight.mean(dim=-1, keepdim=True) + 1e-6)
```

- [ ] **Step 5: 实现`WeightedRidgeLayer`及fallback rate输出**

所有线性代数在autocast关闭区使用FP32/complex64；输出Gram谱、条件数、effective rank、effective sample size和覆盖度。

- [ ] **Step 6: 运行聚焦测试**

Run: `pytest code/tests/test_ecrs_weighted_ridge.py code/tests/test_ecrs_response_basis.py -v`

Expected: PASS。

- [ ] **Step 7: 提交本任务**

```text
git add code/model_dual_cvsincnet.py code/tests/test_ecrs_weighted_ridge.py code/tests/test_ecrs_response_basis.py
git commit -m "feat: add differentiable ECRS identification"
```

### Task 5: 实现锚点曲面编码与受限残差融合

**Files:**
- Modify: `code/model_dual_cvsincnet.py`
- Test: `code/tests/test_ecrs_model_contract.py`

**Interfaces:**
- Consumes: `resp_coef`、协方差、覆盖度和旧`z_id_raw: FloatTensor[B,160]`。
- Produces: `resp_anchor`、`z_resp: FloatTensor[B,64]`、`z_id_fused: FloatTensor[B,160]`与`rho_resp`。

- [ ] **Step 1: 写固定锚点、门控上下界和单位范数失败测试**

```python
assert out["z_resp"].shape == (batch, 64)
assert out["z_id_fused"].shape == (batch, 160)
assert torch.all((out["rho_resp"] >= 0) & (out["rho_resp"] <= 0.25))
torch.testing.assert_close(out["z_id_fused"].norm(dim=1), torch.ones(batch), atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: 实现`SurfaceAnchorEncoder`和`ResponseFusionGate`**

gate输入只包含detach后的`log_condition/effective_rank/N_eff/coverage/NMSE/SNR/covariance`，不得读取标签、类别logit或原始幅度直方图。

- [ ] **Step 3: 使用原CosFace head对`z_id_fused`分类并保留raw auxiliary CE**

新增`z_id_raw/z_resp/z_id_fused`，兼容别名`z_id=z_id_fused`仅在ECRS开启时生效；`tx_logits_raw`继续由旧`z_id_raw`产生。

- [ ] **Step 4: 验证设计稿第30节全部输出键**

Run: `pytest code/tests/test_ecrs_model_contract.py -v`

Expected: PASS。

- [ ] **Step 5: 提交本任务**

```text
git add code/model_dual_cvsincnet.py code/tests/test_ecrs_model_contract.py
git commit -m "feat: fuse bounded ECRS identity evidence"
```

### Task 6: 实现设计稿规定的响应损失

**Files:**
- Modify: `code/cvsrffi/losses.py`
- Test: `code/tests/test_ecrs_losses.py`

**Interfaces:**
- Consumes: clean/LEO配对输出、幅度分层split、标签mask、receiver/day和曲面质量。
- Produces: `response_split_fit_loss`、`response_pair_cross_prediction_loss`、`response_surface_distance`、`same_tx_cross_response_loss`、`different_tx_response_ranking_loss`、`basis_gauge_loss`和`response_gate_calibration_loss`。

- [ ] **Step 1: 为每个设计函数写独立失败测试**

覆盖A→B/B→A、clean→LEO/LEO→clean、同TX跨receiver正样本、匹配域与激励覆盖的异TX负样本、低可靠锚点降权和gate rescue/harm。

- [ ] **Step 2: 实现幅度分层50/50 split-fit**

相同包内两半尽量覆盖相同幅度区；分割只基于激励幅度，不读取TX真值。

- [ ] **Step 3: 实现双向cross-response与曲面距离**

主比较对象是固定锚点函数值或跨内容残差预测，不把自由基原始系数距离作为主损失。

- [ ] **Step 4: 实现有标签判别与gate校准**

同TX正样本要求跨receiver/day；异TX负样本匹配receiver、day、view、激励直方图和SNR区间。

- [ ] **Step 5: 运行聚焦测试**

Run: `pytest code/tests/test_ecrs_losses.py -v`

Expected: PASS。

- [ ] **Step 6: 提交本任务**

```text
git add code/cvsrffi/losses.py code/tests/test_ecrs_losses.py
git commit -m "feat: add ECRS response objectives"
```

### Task 7: 按报告Stage0–Stage6接入训练日程和梯度分流

**Files:**
- Modify: `code/cvsrffi/schedule.py`
- Modify: `code/train.py`
- Test: `code/tests/test_ecrs_schedule.py`
- Test: `code/tests/test_ecrs_gradient_routing.py`

**Interfaces:**
- Consumes: epoch、ECRS配置、`L_s/U_s`batch和模型aux输出。
- Produces: 每阶段损失scale、参数冻结状态和诊断日志。

- [ ] **Step 1: 写Stage0–Stage6启用矩阵测试**

```python
assert stage2["canonical"] and not stage2["diff_tx"]
assert stage3["split_fit"] and stage3["pair_cross"]
assert stage4["resp_cls"] and stage4["same_tx_cross"]
assert not v1_default["learnable_basis"]
```

- [ ] **Step 2: 在`cvsrffi/schedule.py`增加独立ECRS日程，不重写旧ADV3B02日程**

Stage2训练canonical/content；Stage3固定基自监督；Stage4加入有标签响应判别并把`rho`从0升至0.2；Stage5学习基入口默认关闭；Stage6只在Teacher稳定后接FastTrust类别结构。

- [ ] **Step 3: 在训练循环接入`L_s+U_s`响应自监督**

`U_s`只进入canonical、content、split-fit与pair-cross；所有需TX标签的loss必须由`label_mask`屏蔽。

- [ ] **Step 4: 保留原ADV3B02总损失并增加raw CE**

```python
loss_total = loss_adv3b02 + loss_ecrs
loss_id = loss_cosface_fused + 0.3 * loss_cosface_raw + alpha_resp * loss_resp_ce
```

`alpha_resp`只在报告范围`0.10–0.25`内配置。

- [ ] **Step 5: 验证梯度隔离**

初期TX CE不能更新ContentEstimator；拟合loss不能通过把`W`压到0作弊；`z_dom`不直接进入TX分类器或响应分类器。

- [ ] **Step 6: 运行聚焦测试**

Run: `pytest code/tests/test_ecrs_schedule.py code/tests/test_ecrs_gradient_routing.py -v`

Expected: PASS。

- [ ] **Step 7: 提交本任务**

```text
git add code/cvsrffi/schedule.py code/train.py code/tests/test_ecrs_schedule.py code/tests/test_ecrs_gradient_routing.py
git commit -m "feat: stage ECRS training gradients"
```

### Task 8: 升级checkpoint与单视图部署schema

**Files:**
- Modify: `code/cvsrffi/checkpoint.py`
- Modify: `code/train.py`
- Test: `code/tests/test_ecrs_checkpoint.py`

**Interfaces:**
- Consumes: ECRS模型、固定basis、`M_ref`、anchor grid、归一化统计、响应原型与协方差。
- Produces: `ecrs_bundle`和`ADV3B02:ECRS:z_fused:unit_l2:160:v1`。

- [ ] **Step 1: 写保存—加载—单视图推理失败测试**

断言bundle字段齐全、schema精确匹配、加载后同一LEO IQ的`z_id_fused`一致，且推理函数不接受clean伴随输入。

- [ ] **Step 2: 扩展checkpoint payload**

```python
payload["ecrs_bundle"] = model.export_ecrs_bundle() if model.use_ecrs else None
payload["feature_schema"] = "ADV3B02:ECRS:z_fused:unit_l2:160:v1" if model.use_ecrs else legacy_schema
```

- [ ] **Step 3: 验证旧checkpoint无ECRS字段仍可严格加载旧模型**

Run: `pytest code/tests/test_ecrs_checkpoint.py code/tests/test_ecrs_model_contract.py -v`

Expected: PASS。

- [ ] **Step 4: 提交本任务**

```text
git add code/cvsrffi/checkpoint.py code/train.py code/tests/test_ecrs_checkpoint.py
git commit -m "feat: persist ECRS deployment schema"
```

### Task 9: 补齐诊断、负对照与V1正式入口

**Files:**
- Modify: `code/train.py`
- Create: `code/scripts/launch_phase1_adv3b02_ecrs_v1_20260901.sh`
- Create: `code/tests/test_phase1_adv3b02_ecrs_v1_launcher.py`
- Create: `code/tests/test_ecrs_negative_controls.py`

**Interfaces:**
- Consumes: ECRS训练统计与现有Phase1评测器。
- Produces: 设计稿第27节诊断、R0–R8逐级入口和三种LEO弱场景最终评测。

- [ ] **Step 1: 写launcher dry-run失败测试**

断言入口固定ADV3B02主干、ECRS-V1、`K=28`、64维响应、`rho_max=0.25`、E200、三段LEO_WEAK日程和四种最终测试。

- [ ] **Step 2: 写设计稿第28节负对照**

包括激励/残差打乱、仅质量特征TX分类、clean/LEO pair打乱、固定MP/固定样条/学习样条比较，以及原始系数/白化系数/锚点曲面对照。学习基对照只属于R9，默认不进入V1训练。

- [ ] **Step 3: 记录第27节全部诊断**

记录response NMSE、same/diff prediction ratio、Gram条件数、effective rank、覆盖度、ridge fallback、独立TX/RX probe、gate rescue/harm/net gain及曲面导出数据。

- [ ] **Step 4: 保留R0–R11递进关系**

正式V1先运行R0–R8；只有same-TX cross-response优于different-TX、clean/LEO surface distance下降且LEO mean/floor/strict UDU稳定提升后，才进入R9–R11。

- [ ] **Step 5: 运行本地聚焦验证**

Run: `pytest code/tests/test_ecrs_*.py code/tests/test_phase1_adv3b02_ecrs_v1_launcher.py code/tests/test_baseline_origin_sat_view.py -v`

Expected: PASS。

- [ ] **Step 6: 运行一次真实checkpoint无query smoke**

使用`ssr-gpu`环境、真实ADV3B02 checkpoint、单个clean/LEO配对batch；验证forward、backward、checkpoint往返和单LEO推理，不连接Phase2 query。

- [ ] **Step 7: 提交并推送实现**

精确stage本次代码、测试与launcher，提交后自动push，并独立核对远端分支OID等于本地`HEAD`。

### Task 10: 按最小实验工作流发布首个N607证伪矩阵

**Files:**
- Create: `automation_reports/CV-SincNet/<immutable-run-id>/report.md`
- Create: `release_archives/<immutable-release>.tar.gz`

**Interfaces:**
- Consumes: 已通过本地聚焦测试、真实checkpoint smoke和一次P0/P1审查的提交。
- Produces: R0→R8单seed最小同row证据；不自动扩大到多seed/完整125。

- [ ] **Step 1: 在报告写入最小预登记**

只记录候选/矩阵、commit、命令、环境/CWD、输入输出路径、GPU、停止规则和预期artifact。

- [ ] **Step 2: 进行一次N607资源/路径preflight、一次release归档SHA对比和一次远端编译**

不得增加成员hash、seal、receipt链或重复审查。

- [ ] **Step 3: 启动后只做一次PID/CWD/cmdline/GPU/log增长绑定检查**

低性能不得停止实验；只按预注册系统技术失败规则处理。

- [ ] **Step 4: prediction完成后由独立scorer连接truth**

同row报告clean、三种LEO弱场景、LEO mean、LEO floor、strict UDU、response cross-pred、gate rescue/harm和资源成本。

- [ ] **Step 5: 只按设计稿判定是否进入后续项**

若V1未同时满足响应可辨识性和身份指标稳定提升，保留负结果并停止在R8；不得直接跳到共享低秩基、反事实移植或Phase2响应注册。

## Self-review

- Spec coverage: ECRS-01至ECRS-22均映射到Task1–Task10。
- Placeholder scan: 未发现未决占位语；设计稿规定的后续项明确标为deferred且有进入条件。
- Type consistency: `z_id_raw[160]→z_resp[64]→P[64→160]→z_id_fused[160]`，`resp_coef/resp_cov_diag[28]`，schema和checkpoint字段一致。
- Strictness: 这是严格设计一致性计划，不是近似PA分支方案；唯一工程路径映射是把设计稿的`schedule.py`落到当前真实文件`code/cvsrffi/schedule.py`。
