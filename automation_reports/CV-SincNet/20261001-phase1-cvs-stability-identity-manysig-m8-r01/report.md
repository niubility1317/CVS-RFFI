# CVS 性能优先研发：物理补充表征

run_id：`20261001-phase1-cvs-stability-identity-manysig-m8-r01`。状态 LOCAL_VERIFIED，尚无新 E200 或 clean 结果。此前在读取 attentive clean 结果前固定的两种结构，各四 seed，从零同划分/预算、唯一身份 CE、无增强/域骨干。性能优先，成本次要。

|候选|总参数/有效 CE 参数|Conv/Linear MAC/包|
|---|---:|---:|
|phase_delta|165393|10003748|
|phase_dsq|165521|10007588|

只新增原生默认预算的相位增量（8 通道）和可选频谱差分（4 通道），保留原始 time/frequency/PA 路径、池化、宽度、嵌入、融合和残差头。数学性质只适用于补充分支，不宣称整个网络相位/信道不变。4 项聚焦检查通过，新模块实际 CE 梯度、两个补充分支性质、单包独立、有限输入及源权限核实。CPU profile 仅验证计量，GPU 成本与性能待测。

全部 scratch E200×50、L6300/U56700unused/V27000、split392005、四固定 seed。源性能最高冻结一个后默认 clean4 新预测＋原20控制只读复用，共24行独立评分。target 不回流设计、源选择或重训；不测未选候选，不改旧所有产物，不干预健康进程。见[本地验证](evidence/local_validation.json)、[登记](experiment.json)、[数学/通信/物理/RFFI 设计](../../../docs/CVS_STABILITY_IDENTITY_RESEARCH_20261001.md)和[测试结果读取前固定的假设](../../../docs/CVS_PHYSICAL_STABILITY_HYPOTHESES_20261001.md)。独立 P0/P1 审查与发布证据随后补充。

独立P0/P1审查PASS，4项检查审查者独立运行全部通过，[审查证据](evidence/independent_review.json)。
