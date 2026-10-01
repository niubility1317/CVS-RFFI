# CVS 连续复相关身份表征

run_id：`20261001-phase1-cvs-coherence-identity-manysig-m8-r01`。状态 TRAINING_COMPLETE，完整源 E200 已完成，clean 尚未完成。两候选每4seed、纯身份CE、scratch、无增强/域骨干。只改变相位增量公式，两候选参数与原配对物理候选、全部初始权重逐项相同；性能优先、资源次要。

|候选|参数/有效CE参数|Conv/Linear MAC/包|
|---|---:|---:|
|coherence_phase|165393|10003748|
|coherence_dsq|165521|10007588|

正 eps 的连续复相关可避免原 atan2 实部平移后的构造奇点，并按能量弱化相位可信度。5项聚焦检查通过；组件构造导数、相关模界、真实 CE 梯度、单包独立和相位旋转性质核实。数学性质只适用于补充统计，正式源记录未出现非有限梯度，不将数学构造缺陷归因于测试 seed 波动。完整源曲线及代数性质作为本轮依据，未利用目标成绩选择候选/epsilon/预算/seed。

见[设计分析](../../../docs/CVS_COHERENCE_IDENTITY_RESEARCH_20261001.md)、[验证](evidence/local_validation.json)、[组件导数](evidence/equation_derivative_validation.json)、[源曲线依据](evidence/source_curve_basis.json)和[登记](experiment.json)。默认源完整冻结后进行4新clean＋20原控制复用的独立测试；尚未发布，P0/P1独立审查待补充。

独立 P0/P1 审查 PASS，5 项检查审查者独立复测通过。见[审查](evidence/independent_review.json)。

## N607源训练发布已核实

代码`282c456861064670040625ccd12d539659c1f654`已push并独立核对远端OID。新release传输校验、远端compile与Torch2.1冷进程双候选CE反向检查通过。独立读回dispatcher PID908102的CWD/argv；8行各占一块GPU，进程与resolved/log增长已核实，状态SOURCE_TRAINING。实际6300L/56700Uunused/27000V、每轮50步、纯CE/无增强/域骨干关闭与目标访问关闭一致。发布后源码保持不可变，后续clean按另一个不可覆盖run发布。详见[发布后读回](evidence/source_launch_readback.json)。

## 后续确认路径与当前交接

源 dispatcher PID `908102`，实际源代码提交 `282c456861064670040625ccd12d539659c1f654`。已独立核实 8 行进程、GPU、实际配置和日志增长。后续 clean 协议独立 P0/P1 审查 PASS，3 个新增协议检查由审查者独立通过，本地总计 38 项聚焦检查通过。见[确认路径审查](evidence/independent_clean_protocol_review.json)。尚无本轮目标成绩，正式 clean 配置和登记等待真实 8 行 E200 完成后按源规则冻结；不从中途性能提前选择，也不重复启动或热改已发布训练。

恢复先只读 inspect 本 run，核实 dispatcher/row 进程或完整终态；本地 `.codex_tmp/read_stability_source_complete.py` 完整核对 1600 epoch、80000 step、实际角色/物理分支状态与源规则。真实冻结后由 `.codex_tmp/prepare_stability_confirmation.py` 建立选中 4 行新预测＋旧 20 行只读控制的确认 run `20261001-phase1-cvs-coherence-clean-manysig-m24-r01`，Git 提交推送核对后发布，最后独立评分并完整报告。目标仍 ACTIVE，当前已完成设计、验证、审查、发布及健康训练启动，性能改进尚未得到完整实验确认。

## 测试结果之前的条件性源研究依据

完整 source 终态审计通过后，在本轮 clean 预测和评分前，仅读取合规 scratch source checkpoint 的分类器权重聚合 Gram，不读取目标数据、不推理或拟合。选中源候选四 seed 的最小类别向量角约85.54°至90.49°；等角单纯形代数界约101.54°，该差距不证明当前模型存在错误或后续一定提升。见[源分类头几何](evidence/source_classifier_geometry.json)和[条件性前瞻假设](../../../docs/CVS_SIMPLEX_SOURCE_HYPOTHESIS_20261001.md)。本轮选择、预算和预测计划保持已冻结状态；新假设尚未登记或执行，仅在目标仍未充分达成时继续源研发。

## 完整源训练与冻结

8 行均完成 E200/10000 步，完整 1600 epoch、80000 step、CSV 和全部 stdout 已解析，无运行期技术异常；实际源数据角色一致，scratch/noaug/nodomain/CE-only 核实。按登记的性能优先规则冻结 `coherence_phase`，源选择独立复算完全一致。源性能完全并列后才考虑参数和计算，不使用测试反馈。确认 run `20261001-phase1-cvs-coherence-clean-manysig-m24-r01` 已预登记，默认完成 4 个新预测和 20 个冻结控制预测的独立评分。见[evidence/source_selection.json](evidence/source_selection.json)和[evidence/source_completion_validation.json](evidence/source_completion_validation.json)。
