# CVS 性能优先研发：物理补充表征

run_id：`20261001-phase1-cvs-stability-identity-manysig-m8-r01`。状态 TRAINING_COMPLETE，尚无本轮完整 E200 或 clean 结果。此前在读取 attentive clean 结果前固定的两种结构，各四 seed，从零同划分/预算、唯一身份 CE、无增强/域骨干。性能优先，成本次要。

|候选|总参数/有效 CE 参数|Conv/Linear MAC/包|
|---|---:|---:|
|phase_delta|165393|10003748|
|phase_dsq|165521|10007588|

只新增原生默认预算的相位增量（8 通道）和可选频谱差分（4 通道），保留原始 time/frequency/PA 路径、池化、宽度、嵌入、融合和残差头。数学性质只适用于补充分支，不宣称整个网络相位/信道不变。4 项聚焦检查通过，新模块实际 CE 梯度、两个补充分支性质、单包独立、有限输入及源权限核实。CPU profile 仅验证计量，GPU 成本与性能待测。

全部 scratch E200×50、L6300/U56700unused/V27000、split392005、四固定 seed。源性能最高冻结一个后默认 clean4 新预测＋原20控制只读复用，共24行独立评分。target 不回流设计、源选择或重训；不测未选候选，不改旧所有产物，不干预健康进程。见[本地验证](evidence/local_validation.json)、[登记](experiment.json)、[数学/通信/物理/RFFI 设计](../../../docs/CVS_STABILITY_IDENTITY_RESEARCH_20261001.md)和[测试结果读取前固定的假设](../../../docs/CVS_PHYSICAL_STABILITY_HYPOTHESES_20261001.md)。独立 P0/P1 审查与发布证据随后补充。

独立P0/P1审查PASS，4项检查审查者独立运行全部通过，[审查证据](evidence/independent_review.json)。

## N607源训练发布已核实

代码`ba089c05a10c72f48007b9db1b9997cf3441a422`已push并独立核对远端OID。新release传输校验、远端compile与Torch2.1冷进程双候选CE反向检查通过。独立读回dispatcher PID878839的CWD/argv；8行各占一块GPU，进程与resolved/log增长已核实，状态SOURCE_TRAINING。实际6300L/56700Uunused/27000V、每轮50步、纯CE/无增强/域骨干关闭与目标访问关闭一致。发布后源码保持不可变，后续clean按另一个不可覆盖run发布。详见[发布后读回](evidence/source_launch_readback.json)。

## 后续确认路径与当前交接

源 dispatcher PID `878839`，实际源代码提交 `ba089c05a10c72f48007b9db1b9997cf3441a422`。已独立核实 8 行进程、GPU、实际配置和日志增长。后续 clean 协议独立 P0/P1 审查 PASS，3 个新增协议检查由审查者独立通过，本地总计 34 项聚焦检查通过。见[确认路径审查](evidence/independent_clean_protocol_review.json)。尚无本轮目标成绩，正式 clean 配置和登记等待真实 8 行 E200 完成后按源规则冻结；不从中途性能提前选择，也不重复启动或热改已发布训练。

恢复先只读 inspect 本 run，核实 dispatcher/row 进程或完整终态；本地 `.codex_tmp/read_stability_source_complete.py` 完整核对 1600 epoch、80000 step、实际角色/物理分支状态与源规则。真实冻结后由 `.codex_tmp/prepare_stability_confirmation.py` 建立选中 4 行新预测＋旧 20 行只读控制的确认 run `20261001-phase1-cvs-stability-clean-manysig-m24-r01`，Git 提交推送核对后发布，最后独立评分并完整报告。目标仍 ACTIVE，当前已完成设计、验证、审查、发布及健康训练启动，性能改进尚未得到完整实验确认。

## 截至当前的完整可用源日志审计

已读八行全部当前完整 step/epoch/CSV 和 stdout，覆盖 47 至 53 轮。实际纯身份 CE、梯度有限、phase/DSQ 激活状态及源角色均符合登记，未发现运行期异常；八个实际训练进程存活。源验证准确率和训练损失只是中途收敛证据，不作选模或 clean 性能确认。持续写入时尚未结束的末行单独记录，不作为完整记录解析，E200 完成后仍执行完整终态审计。见[当前全日志审计](evidence/source_live_full_log_audit.json)。

## 完整源训练与冻结

8 行均完成 E200/10000 步，完整 1600 epoch、80000 step、CSV 和全部 stdout 已解析，无运行期技术异常；实际源数据角色一致，scratch/noaug/nodomain/CE-only 核实。按登记的性能优先规则冻结 `phase_dsq`，源选择独立复算完全一致。源性能完全并列后才考虑参数和计算，不使用测试反馈。确认 run `20261001-phase1-cvs-stability-clean-manysig-m24-r01` 已预登记，默认完成 4 个新预测和 20 个冻结控制预测的独立评分。见[evidence/source_selection.json](evidence/source_selection.json)和[evidence/source_completion_validation.json](evidence/source_completion_validation.json)。
