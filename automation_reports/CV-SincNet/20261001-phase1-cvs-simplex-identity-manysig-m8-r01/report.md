# CVS 等角身份分类头

run_id：`20261001-phase1-cvs-simplex-identity-manysig-m8-r01`。状态 TRAINING_COMPLETE，完整源 E200 已完成，clean 尚未完成。仅身份骨干/普通CE/无增强，从零训练两个源域前瞻候选各4seed。性能优先，参数成本次要。

|候选|总/有效CE参数|Conv/mm MAC/包|常驻状态字节|
|---|---:|---:|---:|
|simplex_learned|165233|10008548|661704|
|simplex_fixed|164433|10003748|662224|

MAC不含QR因子分解和归一化等算子；实际N607训练/推理耗时、峰值显存待正式测量。CPU测量只是实现验证，不作为星载成本或训练性能证据。

4项聚焦检查通过，涵盖两候选真实CE/有限非零梯度、单纯形Gram与满秩、单包独立、零/常量输入、严格状态回读、配对初始编码器和原型、性能优先选模及scratch防护。见[验证](evidence/local_validation.json)、[设计](../../../docs/CVS_SIMPLEX_IDENTITY_RESEARCH_20261001.md)、[测试前固定假设](../../../docs/CVS_SIMPLEX_SOURCE_HYPOTHESIS_20261001.md)、[预登记](experiment.json)。源与clean接口各一次独立P0/P1审查PASS（审查者分别独立复测4项与3项），合计40项本地检查与compileall通过；默认完整源选定后4新clean＋20原控制复用独立测试。

历史推送曾因自动审批被拒绝；权限环境更新后已完成push及远端OID独立核实，并发布正式源训练。该历史阻塞未改变源候选、结构、超参数、seed或预算。该交付限制不改变科学选模/测试权限或模型设计。见[独立审查](evidence/independent_review.json)。

## N607实际版本数值验证

独立只读核实 N607 的 Torch2.1.0+cu121，在 CPU 上使用精确已提交分类头源码与既有 immutable coherence 编码器。两候选×4seed 各3次合成普通CE更新，全部前向/梯度有限、编码器/PA及可学习框架梯度非零、框架满秩，Gram最大误差3.57627869e-07，最小框架奇异值1.154132。单包独立及零/常量输入通过，配对初始原型逐项相同，实际有效参数与预登记相符。

未加载checkpoint、读取正式source/target数据、创建或修改远程文件、占用GPU或启动正式实验。只验证实际运行版本的组件数值正确性，不能代替完整E200或测试性能。独立读回确认正式release/run/log路径均不存在。见[运行版本证据](evidence/remote_cpu_simplex_validation.json)。

## N607源训练发布已核实

代码`03f1aa69d81b949fcf0be457164ebc846fe1f8c5`已push并独立核对远端OID。新release传输校验、远端compile与Torch2.1冷进程双候选CE反向检查通过。独立读回dispatcher PID1035284的CWD/argv；8行各占一块GPU，进程与resolved/log增长已核实，状态SOURCE_TRAINING。实际6300L/56700Uunused/27000V、每轮50步、纯CE/无增强/域骨干关闭与目标访问关闭一致。发布后源码保持不可变，后续clean按另一个不可覆盖run发布。详见[发布后读回](evidence/source_launch_readback.json)。

## 完整源训练与冻结

8 行均完成 E200/10000 步，完整 1600 epoch、80000 step、CSV 和全部 stdout 已解析，无运行期技术异常；实际源数据角色一致，scratch/noaug/nodomain/CE-only 核实。按登记的性能优先规则冻结 `simplex_fixed`，源选择独立复算完全一致。源性能完全并列后才考虑参数和计算，不使用测试反馈。确认 run `20261001-phase1-cvs-simplex-clean-manysig-m24-r01` 已预登记，默认完成 4 个新预测和 20 个冻结控制预测的独立评分。见[evidence/source_selection.json](evidence/source_selection.json)和[evidence/source_completion_validation.json](evidence/source_completion_validation.json)。
