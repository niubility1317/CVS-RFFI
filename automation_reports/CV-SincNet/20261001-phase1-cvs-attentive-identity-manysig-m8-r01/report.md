# CVS 性能优先研发：包内注意力统计池化

run_id：`20261001-phase1-cvs-attentive-identity-manysig-m8-r01`。状态 LOCAL_VERIFIED，尚无新 E200 或 clean 结果。两种池化×四 seed，均从零训练，同划分/预算、唯一 CE、无增强/域骨干。源性能最高优先，资源成本次要。

|候选|总参数/有效 CE 参数|Conv/Linear MAC/包|
|---|---:|---:|
|attentive_mean|164417|9716260|
|attentive_moments|164609|9716260|

注意力与波动增益初始化 0，整个模型初始输出与相同随机种子的原残差 CVS 一致。仅改变包内池化，不扩大卷积深度/通道、融合或分类头。评分为包内有界 softmax，原特征值用于加权均值；波动候选合成回原通道数。3 项聚焦检查通过，包括完整模型初始退化、真实 CE 梯度、权重正性/归一化/有界比值、单包独立推理、退化输入和 scratch 防护。CPU profile 仅验证计量，正式 GPU 数据待测。

按 CVS 实验执行与实验登记技能完成本轮矩阵；全部 source E200 后冻结一个候选，默认新确认 run 24 行：4 新预测＋20 冻结控制，只读相同物理 query，独立评分。未选候选不测试，target 不回流研发/源选择/重训。保留原所有产物和负结果；不干预健康进程。见[本地验证](evidence/local_validation.json)、[登记](experiment.json)与[数学/通信/物理/RFFI 分析](../../../docs/CVS_ATTENTIVE_IDENTITY_RESEARCH_20261001.md)。独立 P0/P1 审查与发布后证据随后补充。

独立P0/P1审查PASS，3项检查审查者独立运行全部通过，[审查证据](evidence/independent_review.json)。

## N607源训练发布已核实

代码`c2a2a8d827f3037157fa8b696d8a7f41e7a94137`已push并独立核对远端OID。新release传输校验、远端compile与Torch2.1冷进程双候选CE反向检查通过。独立读回dispatcher PID846967的CWD/argv；8行各占一块GPU，进程与resolved/log增长已核实，状态SOURCE_TRAINING。实际6300L/56700Uunused/27000V、每轮50步、纯CE/无增强/域骨干关闭与目标访问关闭一致。发布后源码保持不可变，后续clean按另一个不可覆盖run发布。详见[发布后读回](evidence/source_launch_readback.json)。

## 后续确认路径与当前交接

源 dispatcher PID `846967`，实际源代码提交 `c2a2a8d827f3037157fa8b696d8a7f41e7a94137`。已独立核实 8 行进程、GPU、实际配置和日志增长。后续 clean 协议独立 P0/P1 审查 PASS，3 个新增协议检查由审查者独立通过，本地总计 30 项聚焦检查通过。见[确认路径审查](evidence/independent_clean_protocol_review.json)。尚无本轮目标成绩，正式 clean 配置和登记等待真实 8 行 E200 完成后按源规则冻结；不从中途性能提前选择，也不重复启动或热改已发布训练。

恢复先只读 inspect 本 run，核实 dispatcher/row 进程或完整终态；本地 `.codex_tmp/read_attentive_source_complete.py` 会完整核对 1600 epoch、80000 step、实际角色与源规则。真实冻结后由 `.codex_tmp/prepare_attentive_confirmation.py` 建立选中 4 行新预测＋旧 20 行只读控制的确认 run `20261001-phase1-cvs-attentive-clean-manysig-m24-r01`，Git 提交推送核对后发布，最后独立评分并完整报告。目标仍 ACTIVE，当前仅完成新设计、验证、审查、发布及健康训练启动，没有证明性能提升。

## 零初始化数学核对

对当前实际实现进行合成 float64 自动求导核对，零注意力的均值参数 Jacobian 等于包内交叉协方差，零波动增益的 Jacobian 等于通道标准差对角矩阵，两者最大绝对误差均为 2.22e-16。它属于结构实现的数学验证，不是正式数据训练或测试结果，不修改源训练、选模和已发布 release。见[数学核对](evidence/math_derivative_verification.json)及设计分析中“零初始化为何仍能学习”。确认报告将只读完成的本 run scratch 权重，补充全部 24 个池化分支的参数学习状态；不作物理系数或测试性能结论。

## 完整源训练与冻结

8 行均完成 E200/10000 步，完整 1600 epoch、80000 step、CSV 和全部 stdout 已解析，无运行期技术异常；实际源数据角色一致，scratch/noaug/nodomain/CE-only 核实。按登记的性能优先规则冻结 `attentive_mean`，源选择独立复算完全一致。源性能完全并列后才考虑参数和计算，不使用测试反馈。确认 run `20261001-phase1-cvs-attentive-clean-manysig-m24-r01` 已预登记，默认完成 4 个新预测和 20 个冻结控制预测的独立评分。见[evidence/source_selection.json](evidence/source_selection.json)和[evidence/source_completion_validation.json](evidence/source_completion_validation.json)。

## 独立 clean 确认完成

选中 `attentive_mean` 由独立子 run `20261001-phase1-cvs-attentive-clean-manysig-m24-r01` 完成四 seed clean 测试，原 20 预测只读复用，共 24 行、192 条评分。准确率 78.0146% ± 0.8144%，较上轮残差 CVS -0.4397 个百分点（2/4 seed 提升）。本轮没有进一步提高平均准确率，参数更少不能替代性能目标。源 run 未读取 target，未选候选不测试，测试评分不回流调参或重选。见[完整确认报告](../20261001-phase1-cvs-attentive-clean-manysig-m24-r01/report.md)。


## 2026-10-03完整测试补齐

本批全部8个固定E200模型已纳入[全量clean测试报告](../20261003-phase1-cvs-all-frozen-clean-backfill-manysig-m304-r01/report.md)，每模型168,000个测试样本；此前未晋级且未测试的候选也已补测，已有预测复用。304行预测固定后统一独立truth-last评分，完整分类决定复算通过。测试结果见汇总报告和逐seed/RX/TX表；历史源域选择记录保持原样。
