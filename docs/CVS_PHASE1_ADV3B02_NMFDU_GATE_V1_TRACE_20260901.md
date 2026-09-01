# ADV3B02-NMFDU-GATE-V1设计追踪表

## 1.文档身份

- 设计来源：`E:\codex\home\attachments\14ef4110-92f5-4613-aace-dcbd985e79e0\pasted-text.txt`
- 对应实施计划：`docs/plans/2026-09-01-adv3b02-nmfdu-gate-v1-implementation.md`
- 基线：`ADV3B02_CORE90_SOFT_E200`，seed=`392002`，epochs=`200`，`z_id=160`
- 新变体：`ADV3B02_NMFDU_GATE_V1_E200`
- 当前状态：`IMPLEMENTED_LOCAL / REAL_CHECKPOINT_SMOKE_PENDING / NOT_RUN`

`NMFDU`表示`Nuisance-Marginalized Fisher–Discriminability–Uncertainty`。该名称用于明确区分报告要求的门控与现有`id_gate`、`freq_gate`等普通学习门。

## 2.边界

本改造仅面向Phase1源域训练和源域清洁/三种`leo_*_weak`评估。训练使用当前`项目.md`规定的`L_s/U_s/V=0.07/0.63/0.30`，不得读取Phase2 support/query、query truth或其派生状态。现有Core90默认路径、输出字段和checkpoint加载行为必须保持不变；只有显式选择新变体时才启用本设计。

仓库中已有ECRS计划，但尚未实现。两项设计可以共享“接收信号规范化→内容/理想激励重建”的基础接口；本变体不得把ECRS的响应系数、加权岭回归、响应面融合或其候选结论混入首轮因果比较。FastTrust、CRRA和其他未出现在本报告中的机制也不进入首轮候选。

## 3.逐项追踪

|ID|报告要求|计划落点|验收证据|状态|
|---|---|---|---|---|
|NMFDU-01|门控回答“当前样本、当前局部区域、当前参数方向是否可辨识”，而不是学习平均优胜分支|三层门控：局部时频掩码、参数方向门、样本级分支门|合成可辨识性单测和门控诊断|VERIFIED_LOCAL|
|NMFDU-02|Fisher必须边缘化增益、公共相位、时延/CFO等干扰参数|`effective_fisher_summary`实现Schur补或等价投影|干扰方向能量被消除、目标方向保留的单测|VERIFIED_LOCAL|
|NMFDU-03|IQ非圆性统计必须从重建理想激励`ŝ`获得，不能把接收响应当激励|共享`CanonicalExcitationEstimator`输出`ŝ`和不确定度|接收通道变化不被错误解释为调制可辨识性的单测|VERIFIED_LOCAL|
|NMFDU-04|IQ分支使用`β`、`ρ`及最小特征值/条件数等方向可靠性|raw分支内部设置IQ参数方向门|BPSK、QPSK/QAM和退化激励合成测试|VERIFIED_LOCAL|
|NMFDU-05|PA分支使用记忆多项式Gram矩阵，而非简单功率比|PA分支计算归一化Gram谱、秩、体积和功率覆盖|恒包络退化、幅度丰富激励可辨识测试|VERIFIED_LOCAL|
|NMFDU-06|PAPR只能作补充，需联合`μ4/μ6`、幅度熵、动态范围、削顶率|PA证据组输出完整幅度统计|尺度不变性和削顶退化测试|VERIFIED_LOCAL|
|NMFDU-07|SNR必须按分支定义|raw/hom/phase/PA/HOS各自提供可解释SNR或残差质量|证据字段齐全且不共享一个伪SNR|VERIFIED_LOCAL|
|NMFDU-08|频谱分支以有效带宽、占用、边缘能量、谱熵和频点能量为依据|hom/频谱证据模块|窄带、宽带和低边缘能量测试|VERIFIED_LOCAL|
|NMFDU-09|相位分支先投影常数/线性/二次相位，再测残差稳定性|phase分支实现nuisance projection|CFO/时延不抬高可靠性、cycle slip降低可靠性|VERIFIED_LOCAL|
|NMFDU-10|HOS可靠性必须包含分段方差/置信度|HOS分支输出累积量及segment covariance|短窗高方差不被当作高可靠性的单测|VERIFIED_LOCAL|
|NMFDU-11|样本级融合固定为raw、hom、phase、PA、HOS五个身份分支|五分支投影到同一160维球面后加权融合|分支名、维度、范数和融合输出测试|VERIFIED_LOCAL|
|NMFDU-12|必须有`g_null`无效证据通道|六路softmax：`g_null`加五分支权重，`Q_sample=1-g_null`|全分支退化时`g_null`显著上升|VERIFIED_LOCAL|
|NMFDU-13|可辨识性不等于身份判别性|训练源域每分支判别散度`D_b`，与物理项分开记录|高Fisher低判别性的反例测试|VERIFIED_LOCAL|
|NMFDU-14|门控需综合可辨识性、判别性、稳定性和不确定度|`logit_phys=a log I+b log D+c log S-dU`|逐项消融和数值稳定测试|VERIFIED_LOCAL|
|NMFDU-15|学习校正必须有界，不能覆盖物理主项|`delta_max*tanh(MLP(stopgrad(e_b)))`|校正绝对值不超过`delta_max`|VERIFIED_LOCAL|
|NMFDU-16|分支投影先归一化再融合|每支`A_b`投影、L2归一化，融合后LayerNorm/L2|单一大范数分支不能静态垄断|VERIFIED_LOCAL|
|NMFDU-17|训练分三阶段：分支独训、冻结分支训门、联合微调|E1–80、E81–120、E121–200|阶段冻结表、优化器参数组和恢复测试|VERIFIED_LOCAL|
|NMFDU-18|clean–LEO配对允许门不同，但融合身份一致；分支对齐仅在双方可靠时施加|新增`lambda_fused_pair`和可靠性交集掩码；不复用`lambda_sat_cons`|门可变、融合身份一致和低可靠分支不强配对的测试|VERIFIED_LOCAL|
|NMFDU-19|无标签样本只做校准/一致性，伪标签需乘样本质量|`U_s`损失使用`Q_sample`调制，不读取真实标签|标签访问负测、低质量样本权重下降|VERIFIED_LOCAL|
|NMFDU-20|多burst Fisher可加，但仅限独立物理样本|提供可选聚合函数，不将同一IQ的多个视图计作新shot|physical ID去重和K不变测试|VERIFIED_LOCAL|
|NMFDU-21|必须防止“响应当激励”、类别捷径、接收机分类、分支塌缩、过硬路由和模型失配|诊断字段、置换探针、熵/使用率监控和退化回退|专项负测及同row诊断|VERIFIED_LOCAL / REAL_SCENARIO_PENDING|
|NMFDU-22|预期门控行为需在清洁、低信噪比、深衰落、窄带和高阶调制等条件下可解释|保存逐样本/逐场景门权、物理证据和`Q_sample`摘要|清洁+三LEO场景同row分析|PENDING|

## 4.与现有ADV3B02的差距结论

当前实现已具有时域、频域、DAC/IQ和PA特征路径，但仍不等于报告设计：

1. 当前`rho`直接从接收IQ计算，不能作为报告定义的激励可辨识性证据。
2. 当前PA路径虽有记忆多项式提升，但没有Gram谱、秩、条件数和幅度覆盖证据。
3. 当前没有显式hom、phase、HOS五分支体系，也没有统一投影归一化。
4. 当前`id_gate`和`freq_gate`没有有效Fisher、判别散度、不确定度或`g_null`。
5. 当前训练没有报告要求的三阶段冻结/解冻协议。

因此，本计划采用新增显式变体，不重解释历史模型，不修改历史结果语义。

## 5.交付判定

本追踪表完成不等于方法实现或性能验证。只有代码、聚焦负测、真实checkpoint无query smoke、单seed最小矩阵、清洁及三LEO场景评估均完成后，才能分别标记`LOCAL_VERIFIED`、`ARTIFACTS_COMPLETE`或`ANALYZED`。任何未运行项目均保持`PENDING`。

## 6.实施进度

- 2026-09-01，任务1：`VERIFIED_LOCAL`。新增默认关闭的`physical_gate_variant=none|nmfdu_v1`契约；默认路径与显式`none`逐tensor一致；只有身份骨干的`nmfdu_v1`路径新增门控参数；训练CLI和候选配置已绑定。验证命令：`python -m pytest -q code/tests/test_adv3b02_nmfdu_legacy_contract.py code/tests/test_identity_only_forward.py code/tests/test_exact_ssdg_checkpoint_loading.py`，结果`9 passed`。这不表示NMFDU统计、融合或训练已经实现。
- 2026-09-01，任务2：`VERIFIED_LOCAL`。新增有效Fisher、复激励非圆性、归一化PA记忆多项式Gram、频谱占用、低阶相位干扰投影和分段HOS置信度纯统计API；合成测试覆盖Schur共线消除、BPSK/QPSK方向、恒包络秩亏、尺度不变、cycle slip和段间失稳。验证命令：`python -m pytest -q code/tests/test_identifiability_stats.py code/tests/test_adv3b02_nmfdu_legacy_contract.py`，结果`10 passed`。NMFDU-02及NMFDU-04至NMFDU-10仍保持`PENDING`，直到这些统计实际接入五分支和门控路径。
- 2026-09-01，任务3：`VERIFIED_LOCAL`。新增共享`NuisanceEstimator→AnalyticCanonicalizer→ContentExcitationEstimator`接口；仅解析处理标量增益、公共相位、CFO和粗时移，默认截断送入门控的`ŝ`与置信度梯度，零信息输入回退为高不确定度，并提供physical sample去重。参考辅助合成测试恢复gain/phase/CFO/shift且规范化NMSE小于`1e-5`；API不接收TX/query标签。验证命令：`python -m pytest -q code/tests/test_canonical_excitation.py code/tests/test_identifiability_stats.py`，结果`11 passed`。NMFDU-03仍保持`PENDING`，直到`ŝ`进入实际IQ/PA门控路径。
- 2026-09-01，任务4–6：`VERIFIED_CORE_LOCAL`。新增固定顺序`raw/hom/phase/pa/hos`五分支证据库、checkpoint可恢复且可冻结的源域判别性EMA、`I/D/S/U`合成、有界且截断上下文梯度的学习校正、显式`g_null`和归一化五分支融合。验证覆盖幅度丰富激励相对恒包络提高PA可辨识性、cycle slip及分段失稳提高phase/HOS不确定度、类别编号置换不改变判别性、冻结状态不自强化、非有限证据安全回退、无有效证据时空路由占主导、可靠分支胜出、物理-only校正严格为零及投影范数均衡。验证命令：`python -m pytest -q code/tests/test_fisher_branches.py code/tests/test_gate_evidence.py code/tests/test_fisher_gate.py`，结果`10 passed`。这些条目只表示门控核心局部成立；NMFDU-01及NMFDU-03至NMFDU-16仍保持`PENDING`，直到核心接入`CVSincNet`身份路径并通过端到端测试。
- 2026-09-01，任务7局部接入：`VERIFIED_INTEGRATION_LOCAL`。NMFDU现已接入`CVSincNet`身份融合点并仅由双骨干模型的身份骨干启用；raw/hom/phase/PA/HOS先经过局部覆盖和参数方向门，再进入`I/D/S/U→g_null+五分支→160维归一化融合`。IQ镜像与PA非线性目标Jacobian显式对增益、公共相位和线性时间趋势执行Schur边缘化；物理证据不接收标签，`D_b`仅在训练模式且显式`update_nmfdu_support`时更新。新路径输出结构化门控摘要，可选逐样本诊断，新checkpoint可`strict=True`恢复；legacy路径未增加模块、state key或forward字段。验证命令：`python -m pytest -q code/tests/test_adv3b02_nmfdu_integration.py code/tests/test_adv3b02_nmfdu_legacy_contract.py`，结果`9 passed`。真实Core90 checkpoint无query smoke和正式训练恢复尚未执行，因此任务7及对应追踪条目仍保持`PENDING`。
- 2026-09-01，任务8训练接入：`VERIFIED_TRAINING_LOCAL`。训练器已按E1–80、E81–120、E121–200实施分支预训练、冻结分支训练样本门和低学习率联合微调；使用稳定优化器参数组跨阶段保留AdamW状态，`training_stage`、`D_b`EMA及冻结状态均随checkpoint恢复。第一阶段固定五分支等权且关闭`g_null`竞争；第二、三阶段使用分类margin oracle、物理先验、空路由校准、使用率平衡；第三阶段仅对成对clean–LEO样本施加融合特征一致性和双方可靠分支交集一致性。`D_b`更新显式限制在第一阶段的清洁`L_s`掩码；`U_s`伪标签置信度和损失乘截断梯度的`Q_sample`，NMFDU路径不读取`U_s`真值统计。验证命令：`python -m py_compile code/SSDG/train_ssdg.py code/cvsrffi/nmfdu_training.py code/model.py code/model_dual_cvsincnet.py`以及`python -m pytest -q code/tests/test_nmfdu_training_schedule.py code/tests/test_adv3b02_nmfdu_integration.py`，结果`14 passed`。该任务完成了NMFDU-17至NMFDU-19的本地机制验证；真实checkpoint和真实场景结论仍属于后续层级。
- 2026-09-01，任务9诊断与多burst：`VERIFIED_DIAGNOSTICS_LOCAL`。新增逐样本门控证据采集、逐场景同row摘要、分支饥饿/过硬路由/null饱和检测、控制信号质量后的receiver probe及报告预期行为检查。多burst严格执行`Q_m=1-g_null`可靠性加权身份融合，并对独立观测的分支Fisher直接求和；重复physical ID会被拒绝，同一结构性秩亏不会因增加burst数被错误修复。验证命令：`python -m pytest -q code/tests/test_fisher_gate_diagnostics.py code/tests/test_nmfdu_multiburst.py code/tests/test_nmfdu_training_schedule.py code/tests/test_adv3b02_nmfdu_integration.py code/tests/test_adv3b02_nmfdu_legacy_contract.py code/tests/test_fisher_branches.py code/tests/test_fisher_gate.py code/tests/test_gate_evidence.py code/tests/test_canonical_excitation.py code/tests/test_identifiability_stats.py code/tests/test_identity_only_forward.py code/tests/test_exact_ssdg_checkpoint_loading.py`，结果`53 passed`。receiver shortcut、分支塌缩和各场景预期行为仍需真实评估数据确认，因此NMFDU-21仅标记本地机制成立，NMFDU-22保持`PENDING`。
- 2026-09-01，P0/P1审查修复：`VERIFIED_LOCAL / TARGETED_REVIEW_PASSED`。针对独立审查的四个P1，`ŝ`改为标签无关的软符号后验重建，不再复制接收IQ幅相；局部掩码改为在局部统计提取前作用，`g_null/Q_sample`通过可学习null方向真实改变融合方向；质量加权损失改为绝对降权；smoke允许父目录已存在但以独占创建拒绝覆盖结果文件。对应回归测试覆盖中等接收响应变化、模型失配不确定度、mask/null方向变化、统一低质量缩放和已有父目录写入。一次定点复审仅核对原四项，结论均为`FIXED`，对应六个测试文件结果`41 passed`。
- 2026-09-01，M0–M4最小矩阵：`VERIFIED_LOCAL / NOT_RUN`。M0固定为历史Core90 checkpoint只评估；M1为全程五分支等权容量对照；M2为`I_b+g_null`且门控及训练目标均不使用`D/S/U/δ`；M3为完整物理`I/D/S/U`且`δ=0`；M4为完整物理项、显式null和有界学习校正。M1控制器在全部200epoch持续训练分支和骨干，避免E81–120无有效可训练路径。不可覆盖launcher固定seed=`392002`、E200、`L_s/U_s/V=0.07/0.63/0.30`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`及clean/三种`leo_*_weak`评估，不含Phase2或query数据参数。本机Git Bash通道被错误路由至故障WSL，故本地`bash -n`未取得有效证据；后续release将在N607原生Linux Bash中完成语法与dry-run验证。完整聚焦套件结果为`72 passed`，关键Python模块`py_compile`通过。
