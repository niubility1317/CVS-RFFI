# Stage2-B结构化晚期编码块适应traceability

目标：基于冻结`ADV3B02_CORE90_SOFT_E200`，实现并验证与≤1%Norm/Gate稀疏更新明确不同的support-only连续晚期编码块适应方法。本文仅追踪目标要求，不新增实验gate。

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|P2-01|目标输入白名单|Phase2只读取固定target received IQ、合法support标签、冻结类原型/映射、冻结checkpoint和预登记配置|`项目.md`、`docs/PROJECT_PROTOCOL.md`、runtime合同/适配模块|local_verified|协议文本与负测|NPZ先核对精确成员集合，再只物化允许字段|
|P2-02|目标禁止项|禁止一切source/clean样本及样本级派生状态；禁止query真值、角色、批量类别反馈|runtime合同、适配模块、协议负测|local_verified|非法字段/路径失败闭合|query只在适配冻结后逐样本推理|
|M-01|冻结边界|冻结分类/判决头、类原型和非选中基座参数|新适配模块|local_verified|参数前后逐项比较|不新增可训练分类器|
|M-02|结构化块|只更新一个或两个连续晚期非分类特征块；目标5%–15%，硬上限20%，不得退化为Norm/Bias/Gate|新适配模块、真实checkpoint smoke|pending|state_dict键、参数占比、非Norm参数变化|A=`f3→f_pool→f_proj`；B=`t3`|
|M-03|support-only目标|可组合support监督、冻结原型锚定、监督对比、同IQ合法view一致性和冻结基座漂移约束|新适配模块|local_verified|loss trace与无query API测试|原型只读|
|M-04|资源预算|适配更新≤40步|新适配模块、runner|local_verified|41步失败闭合、实际更新计数|正式配置固定20步；smoke仅1步|
|Q-01|query隔离|训练期间不载入query；适配状态完全冻结后逐样本独立推理；query不参与选层、超参、早停、回滚、原型或模型更新|predictor/runner、协议负测|local_verified|签名检查、状态前后比较、独立scorer|scorer先完整验证prediction，再首次打开truth|
|V-01|本地验证|完成RED→GREEN→邻近回归和真实checkpoint无query smoke|测试、新适配模块、runner|pending_remote_smoke|聚焦pytest、邻近pytest、真实checkpoint smoke|本地集成55项通过；真实checkpoint smoke待执行|
|V-02|独立审查|每个候选最多一次P0/P1审查，只报告会直接使真实实验跑错/越权/覆盖/无法启动或无合法prediction的问题|Git diff与测试证据|APPROVED|一次初审及一次仅针对原问题的定点复审|5个P1均闭合；定点复审无剩余P0/P1|
|G-01|版本与发布|精确stage、commit/push并回读远端OID|Git分支、run报告|pending|local HEAD=remote OID|不stage无关未跟踪文件|
|E-01|最小矩阵|同checkpoint、VALIDATED_ONCE row、split、seed、K和判决规则运行单seed Target5/Target25|runner、run报告、prediction/scorer|pending|PID/CWD/cmdline/GPU/log增长；prediction闭合；same-row评分|数据句柄匹配后直接复用，不重验|
|E-02|MRIOR基线|仅合规source-free MRIOR可作同协议基线；无合规同row结果时比较结论=`UNKNOWN`|run报告|UNKNOWN|输入权限、预算、row逐项配对|未发现合规同row结果；历史MRIOR会加载source IQ/label，不能作同协议基线|
|E-03|晋级/失败|候选需均值≥MRIOR+1.0pp、floor≥MRIOR+0.5pp且满足资源/协议；未达标记`SCIENTIFIC_FAILURE_NO_PROMOTION`并在硬停止前尝试B|独立评分与run报告|pending|same-row指标与判定|不得扩大权限或预算|
|S-01|硬停止|香港时间2026-08-24 05:00后不启动/派发/扩展/切换新工作，不终止正常RUNNING实验，只汇总真实状态|目标与最终报告|pending|香港时间读回与状态汇总|`RUNNING`不得写成完成或性能结论|

## 设计冻结

- 候选A：`id_backbone.f3→f_pool→f_proj`。`f_pool`无参数；源码参数21,056，以历史真实ADV3 feature参数368,225为分母时占5.718%。
- 候选B：`id_backbone.t3`。源码参数37,824，以同一分母计占10.272%。
- 两个候选均须在真实checkpoint加载后重新计算基座总参数与可训练比例；目标范围为5%–15%，超过20%直接失败闭合。
- 分类头、判决头、Gate、地面类原型与非选中参数保持冻结。训练只使用target support，适配完成后恢复全模型`eval`且全部参数`requires_grad=False`。
- 当前未发现合规、同row的source-free MRIOR Target5/Target25结果；在新的严格配对证据出现前，MRIOR胜负结论固定为`UNKNOWN`。

## 本地验证证据

- RED：`ssr-gpu`解释器在项目`code`路径导入`cvsrffi.stage2_structured_late_block_adaptation`，得到预期`ModuleNotFoundError`。
- RED→GREEN：适配、runner、ground原型导出、target row导出和独立scorer均先观察到预期失败；修复后与邻近回归合并共55项通过。
- 定点复审：原5个P1（物化前白名单、真实二维原型来源、预登记一致性、K-shot闭合、独立scorer）全部闭合，结论`APPROVED`。
- 静态验证：8个Python交付入口经`py_compile`通过。
- 本机PyTorch导入与CPU微型反向传播明显缓慢，但测试均正常闭合；该环境时延不作为科学或发布结论。

> 香港时间2026-08-24 05:00为本目标的硬停止时间。到点后不得启动、派发、扩展或切换任何新候选、实验、审查或实现工作；不得终止届时仍在正常运行的N607实验。只汇总真实状态、已有证据、正在运行的任务和未完成条件，不得将`RUNNING`表述为已完成或已取得性能结论。
