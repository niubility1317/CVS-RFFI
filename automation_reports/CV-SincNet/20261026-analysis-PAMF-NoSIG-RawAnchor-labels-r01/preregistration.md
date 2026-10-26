# 冻结 raw 标签锚点分析

本行是本地CPU上的既有源预测分析，不启动网络、不读取IQ或checkpoint，不作目标测试。run `20261026-analysis-PAMF-NoSIG-RawAnchor-labels-r01`，只有model seed2026092701。前置两次源诊断失败保留原结论：RGPL14/17正确、净纠正−1；恢复专家联合33/33正确、净纠正0。RawLabeler在原困难池净增加312条正确，但置信度≥.95的标签仅93.83%正确，需要独立检验锚点互补性。

固定两行：RawLabeler E40＋NoSIG-Sup E50为`SupAnchor`；RawLabeler E40＋NoSIG-PL E50为`PLAnchor`。只取原始IQ的源V27000冻结logits；原NoSIG E30 confidence<.95定义原2326条困难池。两模型各自confidence≥.95且argmax同类才认证，不平均权重、不拟合新头。共同初始化／源数据意味着错误可能相关，不据一致性推断可靠。

两个候选均在联合结果查看前固定：**认证≥200、观察precision≥99.5%、相对原E30 raw净纠正≥100**。不调阈值、不增加第三教师直到过线。两个固定结果及单模型控制、全RX/day/TX纠正／引错、原困难池错误重叠全部报告。即使通过，也只说明标签来源可行；学生必须另外固定代码／矩阵、实测源和三个学生seed的原始／DURA配对目标性能。

从已完成并独立核实的source控制／专家输出只读收集raw logits、原hard-pool confidence、可见V标签与域元数据。原role_ids逐项匹配，不新增数据验证或hash层。所有候选mask／label先保存，再用可见源V标签计质量。U／target真值与IQ读取0，模型forward和权重读取0。

本地命令：`PYTHON -X utf8 -m experiments.pamf_channel_pl.raw_anchor_analysis --inputs local_artifacts/pamf_ssl_review_20261026/raw_anchor/inputs --output local_artifacts/pamf_ssl_review_20261026/raw_anchor/results --commit COMMIT`。CWD为E:/type10-7，验证过的Phase1 Python/NumPy，CPU，无依赖安装。Git commit是实际分析代码身份。本行不是N607模型发布，不制造GPU smoke或release包；N607只有冻结源输出的只读取回。

独立不可覆盖本地results root，SupAnchor／PLAnchor各自有预测与质量文件；唯一owner `codex-pamf-nosig-rawanchor-analysis-20261026`。缺输入、源角色／物理顺序／shape／finite不符则结束并保留；科学失败照实归档，不自动U标注／学生训练，不覆盖上游任何产物。
