# ADV3B02-MUSE-SSDG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持ADV3B02主干、开放集边界和部署接口兼容的前提下，实现从Epoch 1联合利用`U_s`、具有多证据H/M/L路由和星地信道训练闭环的`ADV3B02_MUSE_SSDG_E200`。

**Architecture:** 将纯算法逻辑集中在新的`cvsrffi.muse_ssdg`模块中，训练入口只负责提取现有模型输出、调用MUSE模块和组装损失。训练期头和分类prototype使用独立state，不写入部署模型state；现有开放集geometry继续只接收`L_s/V_cal`。独立launcher冻结M0-M3同协议矩阵，并在训练完成后强制执行clean和三个`leo_weak`场景测试。

**Tech Stack:** Python 3、PyTorch、pytest、Git Bash、现有CV-SincNet/SSDG训练与LEO评测模块。

**Spec:** `docs/superpowers/specs/2026-08-19-adv3b02-muse-ssdg-design.md`

## Global Constraints

- 数据角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，物理样本ID互斥。
- 训练和选择不得访问target receiver；`U_s`的TX标签不得进入损失、路由、阈值、prototype或checkpoint选择。
- `U_s`不得生成proxy unknown，不得更新开放集prototype、半径、能量或尾部边界。
- checkpoint选择固定为`final_only`。
- 每个训练完成候选必须分别完成clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`测试。
- 所有测试命令在Git Bash中调用`/c/Users/lh594/.conda/envs/ssr-gpu/python.exe`，不得调用PowerShell。
- 先写失败测试并确认按预期失败，再写生产代码；每个任务只stage本任务文件并提交、push、核对远端OID。

---

### Task 1: MUSE日程与纯数据类型

**Files:**
- Create: `code/cvsrffi/muse_ssdg.py`
- Create: `code/tests/test_muse_ssdg_schedule.py`

**Interfaces:**
- Produces: `MUSEScheduleState`、`MUSEConfig`、`muse_schedule_for_epoch(epoch: int, config: MUSEConfig) -> MUSEScheduleState`。
- `MUSEScheduleState`字段固定为`stage`、`ema_decay`、`lambda_u`、`p_sat`、`grl_lambda`、`proto_momentum`、`pseudo_enabled`、`candidate_enabled`、`freeze_statistics`。

- [ ] **Step 1: 写日程边界失败测试**

```python
from cvsrffi.muse_ssdg import MUSEConfig, muse_schedule_for_epoch


def test_muse_schedule_matches_five_training_segments():
    cfg = MUSEConfig()
    assert muse_schedule_for_epoch(1, cfg).stage == "S1"
    assert not muse_schedule_for_epoch(16, cfg).pseudo_enabled
    assert muse_schedule_for_epoch(17, cfg).stage == "S2A"
    assert muse_schedule_for_epoch(40, cfg).p_sat == 0.25
    assert muse_schedule_for_epoch(41, cfg).candidate_enabled
    assert muse_schedule_for_epoch(69, cfg).ema_decay == 0.999
    assert muse_schedule_for_epoch(161, cfg).stage == "S3B"
    assert muse_schedule_for_epoch(181, cfg).freeze_statistics
    assert muse_schedule_for_epoch(200, cfg).lambda_u == 0.25
```

- [ ] **Step 2: 确认测试因模块不存在而失败**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_schedule.py -q`

Expected: FAIL，错误指向`cvsrffi.muse_ssdg`或约定接口尚不存在。

- [ ] **Step 3: 实现不可变配置与分段日程**

```python
@dataclass(frozen=True)
class MUSEConfig:
    s2a_start: int = 17
    s2b_start: int = 41
    s3a_start: int = 69
    s3b_start: int = 161
    s3c_start: int = 181
    final_epoch: int = 200
    lambda_u_full: float = 0.60
    lambda_u_consolidate: float = 0.25
    p_sat_s2a_end: float = 0.25
    p_sat_full: float = 0.50
    grl_min: float = 0.02
    grl_max: float = 0.10


@dataclass(frozen=True)
class MUSEScheduleState:
    stage: str
    ema_decay: float
    lambda_u: float
    p_sat: float
    grl_lambda: float
    proto_momentum: float
    pseudo_enabled: bool
    candidate_enabled: bool
    freeze_statistics: bool
```

对`epoch<1`和`epoch>200`抛出`ValueError`；线性ramp包含两端，所有概率限制在`[0,1]`。

- [ ] **Step 4: 运行日程测试并确认通过**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_schedule.py -q`

Expected: PASS。

- [ ] **Step 5: 提交日程模块**

```bash
git add code/cvsrffi/muse_ssdg.py code/tests/test_muse_ssdg_schedule.py
git commit -m "feat: add MUSE SSDG training schedule"
```

### Task 2: 多证据融合、可靠度与H/M/L路由

**Files:**
- Modify: `code/cvsrffi/muse_ssdg.py`
- Create: `code/tests/test_muse_ssdg_routing.py`

**Interfaces:**
- Consumes: `MUSEConfig`。
- Produces: `geometric_fuse_probabilities(probabilities, weights)`、`align_source_domain_prior(prob, domain_prior, global_prior, gamma, ratio_clip)`、`js_head_disagreement(probabilities)`、`compute_muse_reliability(...)`、`route_muse_reliability(reliability, high_threshold, low_threshold)`。
- `route_muse_reliability`返回`MUSERoute(high, mid, low)`，三个bool mask必须互斥且并集覆盖全部样本。

- [ ] **Step 1: 写融合与路由失败测试**

```python
def test_three_head_fusion_is_normalized_and_routing_is_a_partition():
    p0 = torch.tensor([[0.80, 0.15, 0.05], [0.40, 0.35, 0.25]])
    p1 = torch.tensor([[0.75, 0.20, 0.05], [0.38, 0.37, 0.25]])
    p2 = torch.tensor([[0.85, 0.10, 0.05], [0.34, 0.33, 0.33]])
    fused = geometric_fuse_probabilities([p0, p1, p2], [0.50, 0.25, 0.25])
    assert torch.allclose(fused.sum(1), torch.ones(2), atol=1e-6)
    reliability = torch.tensor([0.91, 0.52, 0.18])
    route = route_muse_reliability(reliability, high_threshold=0.80, low_threshold=0.30)
    stacked = torch.stack([route.high, route.mid, route.low]).int().sum(0)
    assert stacked.tolist() == [1, 1, 1]
```

- [ ] **Step 2: 确认测试因接口缺失而失败**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_routing.py -q`

Expected: FAIL，错误指向尚未实现的融合或路由函数。

- [ ] **Step 3: 实现数值稳定的融合和可靠度**

实现要求：先对输入概率执行`nan_to_num`和`clamp_min(1e-8)`；几何融合在log空间加权；权重和必须大于0；JS分歧使用各头相对均值分布的KL平均；先验校正比截断到`[0.5,2.0]`；可靠度输入包括confidence、margin、JS、prototype distance和stability，输出限制在`[0,1]`。

- [ ] **Step 4: 加入异常输入与单调性测试并运行**

```python
def test_reliability_decreases_when_head_disagreement_increases():
    stable = compute_muse_reliability(
        confidence=torch.tensor([0.9]), margin=torch.tensor([0.5]),
        js=torch.tensor([0.01]), proto_distance=torch.tensor([0.1]),
        stability=torch.tensor([1.0]),
    )
    disputed = compute_muse_reliability(
        confidence=torch.tensor([0.9]), margin=torch.tensor([0.5]),
        js=torch.tensor([0.30]), proto_distance=torch.tensor([0.1]),
        stability=torch.tensor([1.0]),
    )
    assert stable.item() > disputed.item()
```

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_routing.py -q`

Expected: PASS。

- [ ] **Step 5: 提交融合与路由**

```bash
git add code/cvsrffi/muse_ssdg.py code/tests/test_muse_ssdg_routing.py
git commit -m "feat: add MUSE evidence routing"
```

### Task 3: H/M/L损失、时间稳定memory与分类prototype

**Files:**
- Modify: `code/cvsrffi/muse_ssdg.py`
- Create: `code/tests/test_muse_ssdg_losses.py`
- Create: `code/tests/test_muse_ssdg_memory.py`

**Interfaces:**
- Produces: `weighted_soft_cross_entropy(student_logits, teacher_prob, weights, mask)`、`candidate_set_mask(prob, mass=0.75, max_classes=3)`、`candidate_set_cross_entropy(logits, candidate_mask, weights, sample_mask)`、`MUSETemporalMemory.observe(keys, predictions, confidence, epoch)`、`MUSEClassificationPrototypeBank.observe(features, pseudo, domains, high_mask, stable_mask, unlabeled_weight)`。
- Memory和prototype类均提供`state_dict()`与`load_state_dict()`；冻结后`observe`不得改变state。

- [ ] **Step 1: 写低置信候选集和无身份梯度失败测试**

```python
def test_candidate_set_caps_at_three_and_rejects_unreachable_mass():
    prob = torch.tensor([[0.40, 0.30, 0.20, 0.10], [0.24, 0.23, 0.22, 0.16, 0.15]])
    mask, active = candidate_set_mask(prob, mass=0.75, max_classes=3)
    assert mask[0].sum().item() == 3
    assert active.tolist() == [True, False]


def test_inactive_low_confidence_row_has_zero_identity_gradient():
    logits = torch.tensor([[0.2, 0.1, 0.0, -0.1]], requires_grad=True)
    candidate = torch.zeros_like(logits, dtype=torch.bool)
    loss = candidate_set_cross_entropy(
        logits, candidate, torch.ones(1), torch.tensor([False])
    )
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_losses.py code/tests/test_muse_ssdg_memory.py -q`

Expected: FAIL，错误只来自新接口缺失。

- [ ] **Step 3: 实现按有效权重归一化的损失**

分母使用`sum(weights[mask]).clamp_min(1e-8)`；空mask返回`logits.sum()*0.0`；candidate loss定义为候选集合概率和的负对数，不对候选集合内强制均匀标签；任何未标注损失接口均不接受真实TX标签参数。

- [ ] **Step 4: 实现稳定memory与分类prototype约束**

稳定key固定为`(rx_i,day_i,eq_i,sig_i,base_index)`；连续相同预测达到3次才返回stable。未标注prototype权重默认`0.075`且构造时验证位于`[0.05,0.10]`；只有`high_mask & stable_mask`进入更新；`freeze()`后memory和prototype都不可变。

- [ ] **Step 5: 运行两组测试并确认通过**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_losses.py code/tests/test_muse_ssdg_memory.py -q`

Expected: PASS。

- [ ] **Step 6: 提交损失与状态组件**

```bash
git add code/cvsrffi/muse_ssdg.py code/tests/test_muse_ssdg_losses.py code/tests/test_muse_ssdg_memory.py
git commit -m "feat: add MUSE tri-state losses and memory"
```

### Task 4: 训练期局部头、自监督头和扰动回归头

**Files:**
- Modify: `code/cvsrffi/muse_ssdg.py`
- Create: `code/tests/test_muse_ssdg_training_heads.py`

**Interfaces:**
- Produces: `MUSETrainingHeads(z_id_dim, z_dom_dim, num_classes, num_domains, nuisance_dim)`。
- 方法固定为`local_prob(z_id, domains)`、`self_supervised_loss(z_id_a, z_id_b)`、`nuisance_loss(z_dom, targets, valid_mask)`、`training_state_dict()`。
- `deployment_state_dict()`必须返回空字典，防止训练期头进入Phase2 bundle。

- [ ] **Step 1: 写shape、有限值和部署隔离失败测试**

```python
def test_training_heads_are_finite_and_not_deployable():
    heads = MUSETrainingHeads(160, 32, 6, 20, 6)
    zid = torch.randn(8, 160)
    zdom = torch.randn(8, 32)
    domains = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    local = heads.local_prob(zid, domains)
    assert local.shape == (8, 6)
    assert torch.isfinite(local).all()
    loss = heads.nuisance_loss(zdom, torch.randn(8, 6), torch.ones(8, dtype=torch.bool))
    assert torch.isfinite(loss)
    assert heads.deployment_state_dict() == {}
```

- [ ] **Step 2: 确认测试因训练头缺失而失败**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_training_heads.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现低秩domain局部头和两个辅助头**

局部头采用共享`160->rank`投影和每domain的低秩分类增量，不复制完整分类器；自监督头采用两层projection/prediction MLP和stop-gradient对称负余弦；扰动头从`z_dom`输出固定6维并使用masked smooth-L1。空valid mask返回零损失。

- [ ] **Step 4: 运行测试并确认通过**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_training_heads.py -q`

Expected: PASS。

- [ ] **Step 5: 提交训练期头**

```bash
git add code/cvsrffi/muse_ssdg.py code/tests/test_muse_ssdg_training_heads.py
git commit -m "feat: add MUSE training-only heads"
```

### Task 5: 将MUSE接入现有SSDG训练循环

**Files:**
- Modify: `code/SSDG/train_ssdg.py`
- Create: `code/tests/test_muse_ssdg_train_integration.py`

**Interfaces:**
- Consumes: Tasks 1-4的所有公共接口。
- Produces: parser开关`--use_muse_ssdg`、四级`--muse_level {M0,M1,M2,M3}`和完整MUSE超参数；训练checkpoint字段`muse_training_heads`、`muse_temporal_memory`、`muse_classification_prototypes`、`muse_schedule_state`。
- 训练循环以`unlabeled_loader`长度定义MUSE epoch；`labeled_loader`耗尽后循环。

- [ ] **Step 1: 写源码级协议负测与最小训练集成失败测试**

```python
def test_muse_unlabeled_path_never_passes_y_u_to_identity_losses():
    text = Path("code/SSDG/train_ssdg.py").read_text(encoding="utf-8")
    muse_block = text[text.index("def _compute_muse_unlabeled_losses"):]
    signature = muse_block.split("\n", 1)[0]
    assert "y_u" not in signature
    assert "proxy_unknown" not in muse_block.split("def ", 2)[0]


def test_muse_parser_defaults_to_final_only_and_joint_epoch():
    args = build_arg_parser().parse_args(["--output_dir", "out", "--use_muse_ssdg", "true"])
    assert args.checkpoint_selection == "final_only"
    assert args.muse_epoch_basis == "unlabeled_loader"
```

- [ ] **Step 2: 确认测试因MUSE尚未接入而失败**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_train_integration.py -q`

Expected: FAIL，错误指向helper或parser参数缺失。

- [ ] **Step 3: 增加parser、初始化和optimizer参数组**

启用MUSE时创建EMA、训练期头、memory和分类prototype；训练期头参数加入AdamW。M0保持现有路径；M1只启用全程domain/GRL/self/nuisance；M2增加融合和H/M/L；M3再增加卫星学生、跨receiver对齐和分类prototype更新。

- [ ] **Step 4: 重构epoch迭代和未标注损失helper**

新增`_compute_muse_unlabeled_losses(...)`，参数只包含`x_u`、metadata、teacher/student输出、MUSE state和模拟器metadata，不包含`y_u`。训练外诊断将`y_u`比较移到`torch.no_grad()`且输出值不得影响loss、阈值、memory、选择或停止。S1也必须取得`U_s`batch；S3C调用memory和prototype的`freeze()`。

- [ ] **Step 5: 添加开放集防污染断言**

MUSE路径调用现有开放集损失时只传`L_s`张量；若任何API尝试以`dataset_role="U_s"`更新开放集geometry，抛出`RuntimeError("MUSE_PROTOCOL_U_S_OPEN_GEOMETRY_FORBIDDEN")`。

- [ ] **Step 6: 运行集成测试和相邻回归测试**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_train_integration.py code/tests/test_meta_ssl_pseudo_gate.py code/tests/test_concat_sat_channel_aug.py code/tests/test_phase1_p1_protocol.py -q`

Expected: PASS。

- [ ] **Step 7: 提交训练循环集成**

```bash
git add code/SSDG/train_ssdg.py code/tests/test_muse_ssdg_train_integration.py
git commit -m "feat: integrate MUSE into SSDG training"
```

### Task 6: 确定性卫星学生、遥测与checkpoint恢复

**Files:**
- Modify: `code/cvsrffi/muse_ssdg.py`
- Modify: `code/SSDG/train_ssdg.py`
- Create: `code/tests/test_muse_ssdg_satellite.py`
- Create: `code/tests/test_muse_ssdg_checkpoint.py`

**Interfaces:**
- Produces: `stable_sample_keys(extra)`、`select_satellite_student_mask(keys, epoch, probability, seed)`。
- 每epoch遥测新增`muse/high_ratio`、`muse/mid_ratio`、`muse/low_ratio`、`muse/effective_weight`、`muse/head_js`、`muse/proto_update_weight`、`muse/pseudo_precision_diagnostic`。

- [ ] **Step 1: 写确定性选择和checkpoint round-trip失败测试**

```python
def test_satellite_choice_is_stable_for_sample_and_epoch():
    keys = [(1, 2, 3, 4, 5), (1, 2, 3, 4, 6)]
    first = select_satellite_student_mask(keys, epoch=41, probability=0.5, seed=392002)
    second = select_satellite_student_mask(list(reversed(keys)), epoch=41, probability=0.5, seed=392002)
    assert first.tolist() == list(reversed(second.tolist()))
```

checkpoint测试构造训练头、memory和prototype，保存后加载到新实例，断言state、稳定streak和prototype计数完全一致。

- [ ] **Step 2: 运行测试并确认失败**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_satellite.py code/tests/test_muse_ssdg_checkpoint.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现稳定hash、遥测和恢复字段**

hash输入使用UTF-8编码的`seed|epoch|rx|day|eq|sig|base_index`并取SHA-256前8字节，不使用Python进程随机hash。checkpoint保存训练期state用于恢复；final deployment model仍只使用`model`字段。

- [ ] **Step 4: 运行新测试与遥测回归测试**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_muse_ssdg_satellite.py code/tests/test_muse_ssdg_checkpoint.py code/tests/test_ssdg_telemetry.py -q`

Expected: PASS。

- [ ] **Step 5: 提交卫星与恢复闭环**

```bash
git add code/cvsrffi/muse_ssdg.py code/SSDG/train_ssdg.py code/tests/test_muse_ssdg_satellite.py code/tests/test_muse_ssdg_checkpoint.py
git commit -m "feat: close MUSE satellite and checkpoint state"
```

### Task 7: M0-M3 launcher与强制四场景评测

**Files:**
- Create: `code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`
- Create: `code/tests/test_phase1_muse_launcher.py`
- Create: `automation_reports/CV-SincNet/phase1_adv3b02_muse_ssdg_20260819/report.md`

**Interfaces:**
- Launcher接受`--dry-run`和`--only=M0,M1,M2,M3`。
- 每个候选输出根包含`train.log`、`config.json`、`final_ssdg.pth`、`eval_clean.log`、`eval_leo_clear_weak.log`、`eval_leo_low_elev_weak.log`、`eval_leo_rain_weak.log`和逐场景metrics JSON。

- [ ] **Step 1: 写launcher静态失败测试**

```python
def test_launcher_freezes_protocol_and_all_required_evaluations():
    text = Path("code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh").read_text(encoding="utf-8")
    for token in ("--labeled_ratio 0.07", "--unlabeled_ratio 0.63",
                  "--source_cal_ratio 0.15", "--source_select_ratio 0.15",
                  "--checkpoint_selection final_only", "--epochs 200"):
        assert token in text
    for scenario in ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        assert scenario in text
    assert "ARTIFACTS_COMPLETE" in text
```

- [ ] **Step 2: 确认测试因launcher不存在而失败**

Run: `/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_phase1_muse_launcher.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现不可覆盖launcher和四级配置**

M0-M3固定seed`392002`、200 epoch和同一数据split。launcher在输出根已存在时退出；训练结束后严格定位`final_ssdg.pth`，依次运行clean和三个LEO评测；只有四份metrics和日志均存在且非空时写入`ARTIFACTS_COMPLETE`。任何评测技术失败保留训练输出并记录`EVAL_FAILED_<SCENARIO>`。

- [ ] **Step 4: 执行本地dry-run与shell语法检查**

Run: `bash -n code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh`

Run: `bash code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh --dry-run --only=M3`

Expected: shell语法通过；dry-run打印唯一run root、M3训练命令和四条评测命令，不创建正式训练输出。

- [ ] **Step 5: 更新最小预登记报告并运行测试**

报告只记录候选矩阵、commit、命令、环境/CWD、输入输出、GPU、停止规则和预期artifact。运行：

`/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest code/tests/test_phase1_muse_launcher.py -q`

Expected: PASS。

- [ ] **Step 6: 提交launcher与预登记报告**

```bash
git add code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh code/tests/test_phase1_muse_launcher.py automation_reports/CV-SincNet/phase1_adv3b02_muse_ssdg_20260819/report.md
git commit -m "feat: add MUSE Phase1 experiment launcher"
```

### Task 8: 追踪闭合、聚焦回归和发布准备

**Files:**
- Modify: `analysis/adv3b02_muse_ssdg_traceability_20260819.md`
- Modify: `automation_reports/CV-SincNet/phase1_adv3b02_muse_ssdg_20260819/report.md`

**Interfaces:**
- Consumes: Tasks 1-7的实现和测试结果。
- Produces: 18项逐条状态、正向追踪和反向追踪结论，以及N607发布所需单一release归档清单。

- [ ] **Step 1: 运行完整聚焦测试集**

Run:

```bash
/c/Users/lh594/.conda/envs/ssr-gpu/python.exe -m pytest \
  code/tests/test_muse_ssdg_schedule.py \
  code/tests/test_muse_ssdg_routing.py \
  code/tests/test_muse_ssdg_losses.py \
  code/tests/test_muse_ssdg_memory.py \
  code/tests/test_muse_ssdg_training_heads.py \
  code/tests/test_muse_ssdg_train_integration.py \
  code/tests/test_muse_ssdg_satellite.py \
  code/tests/test_muse_ssdg_checkpoint.py \
  code/tests/test_phase1_muse_launcher.py \
  code/tests/test_meta_ssl_pseudo_gate.py \
  code/tests/test_concat_sat_channel_aug.py \
  code/tests/test_phase1_p1_protocol.py -q
```

Expected: 全部PASS且无warning升级为error。

- [ ] **Step 2: 执行真实checkpoint无query smoke**

在本地`ssr-gpu`环境使用ADV3B02 checkpoint，限制一个batch，确认M3能够完成前向、反向、optimizer step、MUSE state保存和重新加载；smoke不得读取target query或test truth。报告记录命令、checkpoint路径、退出码和输出artifact。

- [ ] **Step 3: 完成追踪正反审计**

逐条更新MUSE-001至MUSE-018为`verified`或带证据的未完成状态；随后从每个新增/修改文件反向映射到至少一个MUSE ID。任何不能映射的生产逻辑必须删除或在规范中补充明确需求后重新审批。

- [ ] **Step 4: 检查diff与工作树范围**

Run: `git diff --check HEAD~1..HEAD`

Run: `git status -sb`

Expected: 不包含数据、checkpoint、日志、`local_artifacts`或既有未跟踪文件；本次计划文件均已提交。

- [ ] **Step 5: 提交追踪闭合记录**

```bash
git add analysis/adv3b02_muse_ssdg_traceability_20260819.md automation_reports/CV-SincNet/phase1_adv3b02_muse_ssdg_20260819/report.md
git commit -m "docs: close MUSE SSDG implementation evidence"
```

- [ ] **Step 6: 独立核对远端OID**

```bash
local_oid=$(git rev-parse HEAD)
remote_oid=$(git ls-remote origin "refs/heads/$(git branch --show-current)" | awk '{print $1}')
test "$local_oid" = "$remote_oid"
```

Expected: 返回0；否则将push状态报告为`FAILED`或`UNKNOWN`并保留本地提交。

## Plan Self-Review

- Spec coverage：MUSE-001至MUSE-018均映射到Task 1-8，没有规范要求缺少实现任务。
- Placeholder scan：计划不包含待填字段或未定义的“稍后实现”步骤。
- Type consistency：日程、路由、memory、prototype、训练头、训练循环和launcher接口名称在所有任务中保持一致。
- Scope control：首轮只实现并运行M0-M3单seed矩阵；多seed和S0-S8细消融不属于本计划。
