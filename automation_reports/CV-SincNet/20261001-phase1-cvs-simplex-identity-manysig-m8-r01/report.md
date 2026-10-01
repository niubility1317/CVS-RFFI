# CVS 等角身份分类头

run_id：`20261001-phase1-cvs-simplex-identity-manysig-m8-r01`。状态 LOCAL_VERIFIED，尚未发布或训练。仅身份骨干/普通CE/无增强，从零训练两个源域前瞻候选各4seed。性能优先，参数成本次要。

|候选|总/有效CE参数|Conv/mm MAC/包|常驻状态字节|
|---|---:|---:|---:|
|simplex_learned|165233|10008548|661704|
|simplex_fixed|164433|10003748|662224|

MAC不含QR因子分解和归一化等算子；实际N607训练/推理耗时、峰值显存待正式测量。CPU测量只是实现验证，不作为星载成本或训练性能证据。

4项聚焦检查通过，涵盖两候选真实CE/有限非零梯度、单纯形Gram与满秩、单包独立、零/常量输入、严格状态回读、配对初始编码器和原型、性能优先选模及scratch防护。见[验证](evidence/local_validation.json)、[设计](../../../docs/CVS_SIMPLEX_IDENTITY_RESEARCH_20261001.md)、[测试前固定假设](../../../docs/CVS_SIMPLEX_SOURCE_HYPOTHESIS_20261001.md)、[预登记](experiment.json)。源与clean接口各一次独立P0/P1审查PASS（审查者分别独立复测4项与3项），合计40项本地检查与compileall通过；默认完整源选定后4新clean＋20原控制复用独立测试。

GitHub远端交付尚未完成：自动审批拒绝向现有具体远端推送，等待用户授权目的地；代码与报告可本地提交，尚未远程发布、无训练PID或正式结果。该交付限制不改变科学选模/测试权限或模型设计。见[独立审查](evidence/independent_review.json)。
