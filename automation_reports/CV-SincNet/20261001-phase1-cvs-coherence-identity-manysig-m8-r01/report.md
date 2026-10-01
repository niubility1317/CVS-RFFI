# CVS 连续复相关身份表征

run_id：`20261001-phase1-cvs-coherence-identity-manysig-m8-r01`。状态 LOCAL_VERIFIED，尚无本轮正式 E200 或 clean 结果。两候选每4seed、纯身份CE、scratch、无增强/域骨干。只改变相位增量公式，两候选参数与原配对物理候选、全部初始权重逐项相同；性能优先、资源次要。

|候选|参数/有效CE参数|Conv/Linear MAC/包|
|---|---:|---:|
|coherence_phase|165393|10003748|
|coherence_dsq|165521|10007588|

正 eps 的连续复相关可避免原 atan2 实部平移后的构造奇点，并按能量弱化相位可信度。5项聚焦检查通过；组件构造导数、相关模界、真实 CE 梯度、单包独立和相位旋转性质核实。数学性质只适用于补充统计，正式源记录未出现非有限梯度，不将数学构造缺陷归因于测试 seed 波动。完整源曲线及代数性质作为本轮依据，未利用目标成绩选择候选/epsilon/预算/seed。

见[设计分析](../../../docs/CVS_COHERENCE_IDENTITY_RESEARCH_20261001.md)、[验证](evidence/local_validation.json)、[组件导数](evidence/equation_derivative_validation.json)、[源曲线依据](evidence/source_curve_basis.json)和[登记](experiment.json)。默认源完整冻结后进行4新clean＋20原控制复用的独立测试；尚未发布，P0/P1独立审查待补充。

独立 P0/P1 审查 PASS，5 项检查审查者独立复测通过。见[审查](evidence/independent_review.json)。
