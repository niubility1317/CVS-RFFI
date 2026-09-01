# ADV3B02-NMFDU-GATE-V1设计追踪表

## 1.文档身份

- 设计来源：`E:\codex\home\attachments\14ef4110-92f5-4613-aace-dcbd985e79e0\pasted-text.txt`
- 对应实施计划：`docs/plans/2026-09-01-adv3b02-nmfdu-gate-v1-implementation.md`
- 基线：`ADV3B02_CORE90_SOFT_E200`，seed=`392002`，epochs=`200`，`z_id=160`
- 新变体：`ADV3B02_NMFDU_GATE_V1_E200`
- 当前状态：`PLAN_ONLY / NOT_IMPLEMENTED / NOT_RUN`

`NMFDU`表示`Nuisance-Marginalized Fisher–Discriminability–Uncertainty`。该名称用于明确区分报告要求的门控与现有`id_gate`、`freq_gate`等普通学习门。

## 2.边界

本改造仅面向Phase1源域训练和源域清洁/三种`leo_*_weak`评估。训练使用当前`项目.md`规定的`L_s/U_s/V=0.07/0.63/0.30`，不得读取Phase2 support/query、query truth或其派生状态。现有Core90默认路径、输出字段和checkpoint加载行为必须保持不变；只有显式选择新变体时才启用本设计。

仓库中已有ECRS计划，但尚未实现。两项设计可以共享“接收信号规范化→内容/理想激励重建”的基础接口；本变体不得把ECRS的响应系数、加权岭回归、响应面融合或其候选结论混入首轮因果比较。FastTrust、CRRA和其他未出现在本报告中的机制也不进入首轮候选。

## 3.逐项追踪

|ID|报告要求|计划落点|验收证据|状态|
|---|---|---|---|---|
|NMFDU-01|门控回答“当前样本、当前局部区域、当前参数方向是否可辨识”，而不是学习平均优胜分支|三层门控：局部时频掩码、参数方向门、样本级分支门|合成可辨识性单测和门控诊断|PENDING|
|NMFDU-02|Fisher必须边缘化增益、公共相位、时延/CFO等干扰参数|`effective_fisher_summary`实现Schur补或等价投影|干扰方向能量被消除、目标方向保留的单测|PENDING|
|NMFDU-03|IQ非圆性统计必须从重建理想激励`ŝ`获得，不能把接收响应当激励|共享`CanonicalExcitationEstimator`输出`ŝ`和不确定度|接收通道变化不被错误解释为调制可辨识性的单测|PENDING|
|NMFDU-04|IQ分支使用`β`、`ρ`及最小特征值/条件数等方向可靠性|raw分支内部设置IQ参数方向门|BPSK、QPSK/QAM和退化激励合成测试|PENDING|
|NMFDU-05|PA分支使用记忆多项式Gram矩阵，而非简单功率比|PA分支计算归一化Gram谱、秩、体积和功率覆盖|恒包络退化、幅度丰富激励可辨识测试|PENDING|
|NMFDU-06|PAPR只能作补充，需联合`μ4/μ6`、幅度熵、动态范围、削顶率|PA证据组输出完整幅度统计|尺度不变性和削顶退化测试|PENDING|
|NMFDU-07|SNR必须按分支定义|raw/hom/phase/PA/HOS各自提供可解释SNR或残差质量|证据字段齐全且不共享一个伪SNR|PENDING|
|NMFDU-08|频谱分支以有效带宽、占用、边缘能量、谱熵和频点能量为依据|hom/频谱证据模块|窄带、宽带和低边缘能量测试|PENDING|
|NMFDU-09|相位分支先投影常数/线性/二次相位，再测残差稳定性|phase分支实现nuisance projection|CFO/时延不抬高可靠性、cycle slip降低可靠性|PENDING|
|NMFDU-10|HOS可靠性必须包含分段方差/置信度|HOS分支输出累积量及segment covariance|短窗高方差不被当作高可靠性的单测|PENDING|
|NMFDU-11|样本级融合固定为raw、hom、phase、PA、HOS五个身份分支|五分支投影到同一160维球面后加权融合|分支名、维度、范数和融合输出测试|PENDING|
|NMFDU-12|必须有`g_null`无效证据通道|六路softmax：`g_null`加五分支权重，`Q_sample=1-g_null`|全分支退化时`g_null`显著上升|PENDING|
|NMFDU-13|可辨识性不等于身份判别性|训练源域每分支判别散度`D_b`，与物理项分开记录|高Fisher低判别性的反例测试|PENDING|
|NMFDU-14|门控需综合可辨识性、判别性、稳定性和不确定度|`logit_phys=a log I+b log D+c log S-dU`|逐项消融和数值稳定测试|PENDING|
|NMFDU-15|学习校正必须有界，不能覆盖物理主项|`delta_max*tanh(MLP(stopgrad(e_b)))`|校正绝对值不超过`delta_max`|PENDING|
|NMFDU-16|分支投影先归一化再融合|每支`A_b`投影、L2归一化，融合后LayerNorm/L2|单一大范数分支不能静态垄断|PENDING|
|NMFDU-17|训练分三阶段：分支独训、冻结分支训门、联合微调|E1–80、E81–120、E121–200|阶段冻结表、优化器参数组和恢复测试|PENDING|
|NMFDU-18|clean–LEO配对允许门不同，但融合身份一致；分支对齐仅在双方可靠时施加|新增`lambda_fused_pair`和可靠性交集掩码；不复用`lambda_sat_cons`|门可变、融合身份一致和低可靠分支不强配对的测试|PENDING|
|NMFDU-19|无标签样本只做校准/一致性，伪标签需乘样本质量|`U_s`损失使用`Q_sample`调制，不读取真实标签|标签访问负测、低质量样本权重下降|PENDING|
|NMFDU-20|多burst Fisher可加，但仅限独立物理样本|提供可选聚合函数，不将同一IQ的多个视图计作新shot|physical ID去重和K不变测试|PENDING|
|NMFDU-21|必须防止“响应当激励”、类别捷径、接收机分类、分支塌缩、过硬路由和模型失配|诊断字段、置换探针、熵/使用率监控和退化回退|专项负测及同row诊断|PENDING|
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
