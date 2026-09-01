# ADV3B02-NMFDU-GATE-V1实施计划

> 本计划是编码和实验执行依据。当前只完成设计与现状对齐，不声称代码已实现、实验已运行或性能已提升。

**目标：**在保持`ADV3B02_CORE90_SOFT_E200`历史路径不变的前提下，新增忠实于设计报告的“干扰边缘化Fisher–判别性–不确定度门控”变体，使模型按当前样本、局部区域和参数方向的物理可辨识性选择raw、hom、phase、PA、HOS身份分支，并在所有证据失效时转向`g_null`。

**基线约束：**seed=`392002`、epochs=`200`、`z_id=160`、`L_s/U_s/V=0.07/0.63/0.30`、现有Core90卫星增强日程和`lambda_sat_cls=0.68`保持不变。新配对损失使用独立参数名，`lambda_sat_cons`继续为`0`。训练与选择仅使用Phase1源域数据；验证集只选epoch，不反向传播或更新状态；Phase2不进入本计划首轮实验。

**实现策略：**以纯数学统计层、共享理想激励重建层、五分支证据层、三层门控/融合层、训练调度层五层解耦。物理主项可审计，学习残差有界；所有新功能默认关闭并使用候选专属配置启用。

**追踪表：**`docs/CVS_PHASE1_ADV3B02_NMFDU_GATE_V1_TRACE_20260901.md`

---

## 一、不可改变的设计决定

1. **不是普通注意力。**现有`id_gate`、`freq_gate`保留原语义，不得改名为Fisher门控。新门控必须显式产生`I_b/D_b/S_b/U_b`、`g_null`和五分支权重。
2. **先边缘化干扰。**任何目标参数Fisher证据都必须在增益、公共相位、时延、CFO及相位低阶趋势等干扰方向被Schur补或等价投影后计算。
3. **先重建激励。**IQ/PA的激励统计来自`ŝ`，不能直接从接收信号`r`取矩并解释为发射激励可辨识性。
4. **五个样本级分支。**样本融合分支固定为`raw/hom/phase/pa/hos`。当前DAC/IQ信息归入raw分支的参数方向证据，不额外制造第六个样本级分支。
5. **物理主项不可被网络覆盖。**学习校正使用`delta_max*tanh(·)`，并默认对物理证据输入`stop-gradient`。
6. **无证据必须可表达。**所有分支不可靠时提高`g_null`，不得强迫五选一。
7. **可辨识性与判别性分离。**`I_b`只描述参数可观测程度，`D_b`只由Phase1标注源域身份判别统计估计。
8. **历史兼容。**未启用新变体时，模块构造、forward输出、state dict key和历史checkpoint严格加载行为不变。
9. **首轮保持因果单纯。**不混入ECRS响应面、FastTrust、CRRA或其他新机制；已有ECRS计划只共享规范化/理想激励重建接口。

## 二、目标数据流

```text
received IQ r
  ├─ nuisance estimator/canonicalizer ─ content estimator ─ ŝ + U_recon
  ├─ raw branch ─ local mask ─ IQ direction gate ─ h_raw
  ├─ hom/spectral branch ─ local mask ─ h_hom
  ├─ phase branch ─ nuisance-projected residual ─ local mask ─ h_phase
  ├─ PA branch ─ memory-polynomial Gram direction gate ─ h_pa
  └─ HOS branch ─ segment-confidence mask ─ h_hos

per branch: I_b + D_b + S_b + U_b
  └─ physical logit + bounded learned residual
       └─ softmax[g_null,g_raw,g_hom,g_phase,g_pa,g_hos]
            └─ sum_b g_b * L2(A_b h_b) ─ LayerNorm/L2 ─ z_id(160)
```

门控公式固定为：

```text
l_phys,b = a_b log(I_b + eps) + b_b log(D_b + eps)
           + c_b log(S_b + eps) - d_b U_b
delta_b  = delta_max * tanh(MLP_b(stopgrad(e_b)))
g        = softmax([l_null, l_phys,b + delta_b] / tau)
Q_sample = 1 - g_null
```

`a_b/b_b/c_b/d_b`使用非负参数化；`tau`设下界；`eps`是固定数值稳定项，不能按样本放大以掩盖秩亏。

## 三、实施任务

### 任务1：冻结变体契约和共享边界

**修改文件：**

- `code/configs/phase1_adv3b02_nmfdu_gate_v1.json`（新增）
- `code/model_dual_cvsincnet.py`
- `code/post_stage_common.py`
- `code/tests/test_adv3b02_nmfdu_legacy_contract.py`（新增）

**实施：**

1. 新增`physical_gate_variant=none|nmfdu_v1`，默认`none`。
2. 新增结构化输出`physical_gate_diag`，仅在显式诊断开关时保留逐样本张量；默认只返回聚合摘要，避免改变历史调用。
3. 配置固定五分支名、160维投影、`g_null`、三阶段epoch边界和候选专属损失权重。
4. 将共享规范化/理想激励重建接口放入独立模块；若后续实现ECRS，ECRS导入该接口，而不是复制一套估计器。NMFDU不得导入ECRS的响应面或岭回归输出。

**先写失败测试：**

- 历史配置构造的模型key集合与forward字段不变。
- Core90真实checkpoint可`strict=True`加载。
- 只有`nmfdu_v1`构造新模块和新参数。

**完成条件：**兼容测试通过，候选配置被解析但尚不要求新模块具有性能。

### 任务2：实现纯数学的有效Fisher和可辨识性统计

**新增文件：**

- `code/cvsrffi/identifiability_stats.py`
- `code/tests/test_identifiability_stats.py`

**公开接口：**

```python
effective_fisher_summary(j_target, j_nuisance, weight, eps)
complex_excitation_stats(s_hat, segment_ids, eps)
memory_polynomial_gram_stats(s_hat, order, memory_depth, weight, eps)
spectral_occupancy_stats(s_hat, weight, eps)
phase_residual_stats(r_canonical, valid_mask, polynomial_order, eps)
hos_confidence_stats(r_canonical, segment_ids, eps)
```

**实施要求：**

1. `J_eff=J_tt-J_tn(J_nn+eps I)^-1J_nt`，实现优先使用对称线性求解或正交投影，不显式求逆。
2. 所有Gram矩阵先以能量/trace归一化，再计算`lambda_min`、稳定条件数、有效秩、log-volume。
3. IQ输出`β=E[ŝ²]/E[|ŝ|²]`、`ρ=|β|`及2×2实参数Fisher谱。
4. PA输出`μ4/μ6`、幅度熵、有效动态范围、削顶率和功率覆盖；PAPR只作辅助字段。
5. HOS采用分段估计并返回段间协方差/置信度。

**合成测试：**

- BPSK的`ρ`接近1，QPSK/圆对称QAM的`ρ`更低；测试只验证统计方向，不把调制类型当身份标签。
- 恒包络激励的高阶PA Gram秩亏，幅度丰富激励的有效秩更高。
- 增益缩放后归一化Gram统计近似不变。
- 纯干扰方向在Schur边缘化后不抬高目标可辨识性。
- `eps`只能保证数值可解，不能把秩亏样本变成高可靠样本。

**完成条件：**CPU确定性单测通过，所有输出有限且梯度行为明确。

### 任务3：实现规范化与理想激励重建

**新增文件：**

- `code/cvsrffi/canonical_excitation.py`
- `code/tests/test_canonical_excitation.py`

**组件：**

- `NuisanceEstimator`：估计增益、公共相位、粗时延/CFO和有效采样掩码。
- `AnalyticCanonicalizer`：用解析变换消除已估计干扰，不改变物理sample ID。
- `ContentExcitationEstimator`：从规范化信号估计`ŝ`，同时输出重建残差、条件数和不确定度。

**安全边界：**

1. 估计器只读当前Phase1物理样本，不读TX类别、接收机类别、query角色或query truth。
2. 默认对进入门控的`ŝ`和证据张量截断反向梯度，防止网络把重建器训练成类别捷径。
3. 低置信重建不强行输出“可靠激励”；其不确定度必须进入`U_b`和`l_null`。
4. 同一物理IQ的FFT、等化或分段视图仍是一个样本，不增加K。

**测试：**增益/公共相位/CFO扰动恢复、重建失败回退、标签置换不改变重建结果、同physical ID视图不重复计数。

### 任务4：构建五个显式身份分支和局部门

**修改/新增文件：**

- `code/model.py`
- `code/cvsrffi/fisher_branches.py`（新增）
- `code/tests/test_fisher_branches.py`（新增）

**映射：**

- `raw`：复用当前时域/DAC特征，增加局部时域掩码；IQ方向门只控制相应子空间。
- `hom`：复用可兼容的频域骨干，增加有效带宽、占用、边缘能量、谱熵和频点局部门。
- `phase`：新增相位残差分支，先投影常数/线性/二次趋势，再编码稳定残差。
- `pa`：复用`MemoryPolynomialLift`，增加Gram方向门和幅度覆盖局部门。
- `hos`：新增分段高阶累积量分支，以段间置信度调节局部输出。

每个分支输出`BranchOutput(embedding, local_mask, direction_gate, evidence, uncertainty)`。局部门和方向门必须能独立记录，不能用一个样本级标量冒充三层门控。

**测试：**五分支维度/有限性、局部门范围、IQ/PA方向门退化行为、相位投影、HOS短窗置信度、batch size 1和混合长度掩码。

### 任务5：实现判别性、稳定性和不确定度状态

**新增/修改文件：**

- `code/cvsrffi/gate_evidence.py`（新增）
- `code/SSDG/train_ssdg.py`
- `code/tests/test_gate_evidence.py`（新增）

**定义：**

1. `I_b`由任务2/4的物理可辨识性统计归一化得到。
2. `D_b`由`L_s`上每分支类间散度与类内散度比估计，按epoch以EMA更新；验证集和`U_s`真值不得参与。
3. `S_b`来自同一源物理样本clean–LEO配对的证据稳定性和分支表征稳定性。
4. `U_b`联合重建残差、数值条件、分段估计方差、通道估计误差和模型失配标记。

`D_b`在门控专训阶段冻结，避免门控通过改变判别统计自我强化。所有EMA状态必须进入checkpoint，并支持确定性恢复。

**测试：**高Fisher低判别反例、EMA恢复、验证/无标签真值不可访问、类别索引置换一致性、无穷/NaN回退到高不确定度。

### 任务6：实现物理主项、有界校正、`g_null`和归一化融合

**新增文件：**

- `code/cvsrffi/fisher_gate.py`
- `code/tests/test_fisher_gate.py`

**组件：**

- `FisherDiscriminabilityUncertaintyGate`
- `NormalizedFiveBranchFusion`
- `NullEvidenceHead`

**实现要求：**

1. 五分支物理logit逐项可导出，系数非负参数化。
2. `delta_b`严格受`delta_max`约束；消融时可设为0。
3. `l_null`由总有效证据不足和总体不确定度构成，不输入类别ID。
4. 六路softmax和为1；`Q_sample=1-g_null`。
5. 每支先投影到160维并L2归一化，再加权求和，最后LayerNorm/L2；不得让原始范数决定路由。
6. 提供温度下界、门熵、分支利用率和塌缩告警，但告警不改变预注册停止规则。

**测试：**全退化时null上升、单分支可靠时对应权重上升、校正有界、范数公平、温度有限、梯度只流向允许参数、禁用学习校正时等于物理门。

### 任务7：接入模型与checkpoint

**修改文件：**

- `code/model.py`
- `code/model_dual_cvsincnet.py`
- `code/post_stage_common.py`
- `code/tests/test_adv3b02_nmfdu_integration.py`（新增）

**实施：**

1. Core90骨干产生五分支输入；`nmfdu_v1`替换身份融合点，但不改变`z_dom`和既有域对抗接口。
2. 对外继续输出160维`z_id`，附加可选诊断：五分支门、null门、三层局部门摘要、方向门摘要、`I/D/S/U`和`Q_sample`。
3. 训练恢复必须保存门控EMA、阶段编号和优化器组；阶段边界恢复不得重复冻结或跳阶段。
4. 历史checkpoint严格加载测试和新checkpoint round-trip同时通过。

**真实checkpoint smoke：**使用官方Core90 checkpoint、真实Phase1源样本，执行单batch无query forward/backward；确认旧路径数值/字段不变，新路径可产生prediction和有限诊断。

### 任务8：实现三阶段训练和报告规定的配对约束

**修改文件：**

- `code/SSDG/train_ssdg.py`
- `code/post_stage_common.py`
- `code/tests/test_nmfdu_training_schedule.py`（新增）

**阶段：**

|Epoch|可训练模块|门控方式|主要新增目标|
|---|---|---|---|
|1–80|五分支、投影、现有Core90骨干|等权或固定安全权重，门旁路|各分支身份辅助损失、物理统计稳定|
|81–120|只训练门控、null头和有界校正|完整物理门|oracle margin监督、物理先验KL、null校准|
|121–200|全模型低学习率联合微调|完整物理门|联合身份目标、融合配对一致性、可靠分支配对|

**损失：**

- `L_branch_aux`：五分支独立身份辅助损失，防止门控掩盖弱分支。
- `L_oracle_margin`：使用训练标签比较各分支身份margin得到软oracle，只在门控训练中使用。
- `L_gate_phys_kl`：约束门分布不任意偏离物理可靠性先验。
- `L_fused_pair`：clean和对应LEO的融合身份表征一致；两者门分布允许不同。
- `L_branch_pair`：只在clean和LEO两端该分支均可靠时施加。
- `L_null_cal`：用人工/物理退化但不使用身份真值，校准无有效证据情形。

这些损失使用新参数名；`lambda_sat_cons`保持0。`U_s`伪标签或一致性项只能乘`Q_sample`和原有置信度，不能读取无标签真值。阶段具体权重先由合成测试保证数量级，再在预登记单seed矩阵中冻结；不得使用V以外数据选权重。

**测试：**每阶段参数组、冻结/解冻、LR切换、断点恢复、门可变而融合配对、可靠性交集掩码、低`Q_sample`伪标签降权。

### 任务9：加入失败模式探针和可解释诊断

**修改/新增文件：**

- `code/cvsrffi/fisher_gate_diagnostics.py`（新增）
- `code/tests/test_nmfdu_failure_probes.py`（新增）
- `code/SSDG/train_ssdg.py`

**必须保存的摘要：**

- 按场景的`g_null`、五分支均值/分位数、门熵、分支利用率。
- IQ/PA Fisher最小特征值、有效秩、条件数和重建不确定度。
- phase/HOS稳定性、local mask覆盖率和方向门使用率。
- `Q_sample`与分类正确率、置信度、接收机/场景的相关性。

**专项探针：**

1. TX标签置换：物理证据不应随标签索引变化。
2. receiver probe：门控表征不得比身份基线显著强化接收机可预测性；结果作为科学诊断，不额外创造发布gate。
3. branch collapse：任一分支长期零使用或单分支垄断时记录，但不因中间性能停止实验。
4. model mismatch：所有统计失配时应提高`U_b/g_null`，而不是产生虚假高Fisher。
5. label shortcut：重建器/物理统计禁止接收类别输入，源码和运行时断言双重验证。

### 任务10：准备最小可证伪实验矩阵

**修改/新增文件：**

- `code/scripts/launch_phase1_adv3b02_nmfdu_gate_v1_queue_20260901.sh`（新增）
- `docs/CVS_PHASE1_ADV3B02_NMFDU_GATE_V1_TRACE_20260901.md`
- `automation_reports/CV-SincNet/<run-id>/report.md`（启动前在项目控制面创建）

**首轮同seed矩阵：**

|Row|配置|回答的问题|
|---|---|---|
|M0|原始Core90|历史锚点和实现回归|
|M1|五分支+同维归一化+等权融合|收益是否仅来自容量/分支重构|
|M2|`I_b`+`g_null`，无`D/S/U`、无学习校正|纯物理可辨识性是否有效|
|M3|`I_b+D_b+S_b-U_b`+`g_null`，`delta=0`|完整物理主项的贡献|
|M4|完整物理主项+有界校正+三层门控+三阶段训练|报告完整设计的净效果|

所有row使用seed=`392002`、E200、同一源域split和同一评估配置。每个完成训练的row必须保存最终checkpoint身份，并分别评估clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；不以一个LEO均值替代逐场景结果。

**首轮不做：**多seed、完整Target125、Phase2注册、FastTrust叠加、ECRS叠加、全量消融或未知拒识结论。

**进入多seed确认的预登记科学门槛：**M4相对M0在同row三LEO平均身份准确率有正增益，最差LEO场景不退化，clean下降不超过1个百分点，且没有证据显示收益来自receiver shortcut或门控塌缩。未达到门槛即作为有效负结果记录，不视为技术失败。

### 任务11：本地验证、一次P0/P1审查和N607发布

**本地验证顺序：**

1. 激活`ssr-gpu`，串行执行任务2–9的聚焦测试。
2. 执行现有Core90回归和真实checkpoint无query smoke。
3. 对新launcher做语法/参数解析检查，确认run ID和output root不可覆盖。
4. 仅进行一次独立P0/P1正确性审查；只接受会导致真实实验跑错、越权、覆盖输出、误杀进程、不能启动或不能产生合法prediction的问题。若有直接问题，修复后只做一次定点复审。
5. 提交并push本地代码，独立核对远端分支OID。
6. 按项目规则制作一个release归档，只做一次本地到N607的归档SHA对比和一次远端编译。
7. N607 preflight通过后启动；启动后只做一次PID/CWD/cmdline/GPU/log增长绑定检查。

不得把额外hash、seal、receipt、全量审查、完整矩阵或报告美化引入为发布前置条件；如出现此类要求，在报告中记`REJECTED_EXTRA_GATE`并继续最小流程。

### 任务12：独立评分和结论边界

prediction完整后，由独立scorer连接truth。报告必须保持每个row的seed、split、checkpoint、clean和三LEO逐场景指标、资源成本、门控诊断及结论在同一行或同一证据块中。

首轮允许的结论只有：

- 新门控是否在单seed Phase1源域清洁/LEO弱信道评估中优于同配置Core90。
- 增益主要来自容量、纯物理门、完整物理主项还是有界学习校正。
- 门控行为是否符合报告预期，是否存在receiver shortcut、分支塌缩或模型失配。

首轮不得声称Phase2注册增益、未知类拒识能力、跨数据集普适性或正式多seed统计优势。低性能只触发机制分析和下一候选决定，不触发技术停止。

## 四、建议提交切分

为便于审查和回滚，实施时按以下最小提交切分：

1. `feat: add nuisance-marginalized identifiability statistics`
2. `feat: add canonical excitation reconstruction for physical gating`
3. `feat: add five ADV3B02 physical evidence branches`
4. `feat: add NMFDU null-aware normalized fusion`
5. `feat: add staged NMFDU training and diagnostics`
6. `test: verify NMFDU protocol and legacy checkpoint compatibility`
7. `exp: preregister ADV3B02 NMFDU single-seed matrix`

每个提交只stage本次相关文件并自动push；不得把数据、checkpoint、日志、缓存或无关未跟踪文件带入提交。

## 五、计划完成定义

本计划获批后，任务1–9构成实现闭环，任务10–12构成最小真实证据闭环。以下条件全部满足才可称为本轮改造完成：

1. 追踪表NMFDU-01至NMFDU-22均有代码和测试落点。
2. 历史Core90严格加载与默认路径回归通过。
3. 真实checkpoint无query smoke通过。
4. M0–M4产生不可覆盖prediction并完成clean及三LEO评估。
5. 独立scorer输出同row结果，报告明确性能、资源、失败模式和claim boundary。
6. 正式代码、预登记和结果报告均已提交、push，且远端OID与本地`HEAD`一致。

在上述证据返回前，候选状态保持`PLAN_ONLY`或相应最小生命周期状态，不得提前写成“已优化”或“已提升”。
