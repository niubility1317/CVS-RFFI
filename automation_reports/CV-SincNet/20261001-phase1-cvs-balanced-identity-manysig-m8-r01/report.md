# CVS继续研发：分支平衡融合与有符号PA投影

run_id：`20261001-phase1-cvs-balanced-identity-manysig-m8-r01`。状态：LOCAL_VERIFIED。用户授权继续原目标；本轮两候选四seed均从零训练，纯CE、身份骨干、无输入/信道增强，固定原划分与E200×50预算。默认收尾包含源域冻结后单候选clean测试，不以source训练结束作为全部实验完成。

|候选|总参数/有效CE参数|Conv/Linear MAC/包|
|---|---:|---:|
|balanced_fusion|113665|9657476|
|signed_balanced_fusion|113665|9657476|

相对上一轮164225参数，再减少30.7870%；原CVS382146参数，当前减少70.2561%。MAC只小幅下降，不能据此声称训练大幅加速。CPU资源结果只证明测量路径可执行，GPU成本待本轮实际测量。

候选选择只读本轮两候选四seed E200源V和最差源RX，0.5/0.5加权，0.2个百分点内较低MAC优先，再较少参数。每行checkpoint固定E200，无target反馈、早停、重训或挑最高test seed。后续确认run预登记24行：新候选4预测＋原16基准和上一轮4残差CVS冻结预测只读复用。仅clean，168000同物理query、6TX、7RX；独立scorer最后连接truth。

此前任何source/target产物保留，没有继承权重、optimizer或teacher；parent仅表示架构分析来源。保持VALIDATED_ONCE句柄和物理数据契约，不重复数据验证。健康任务不停止、不重启、不热改；每GPU最多两个总训练进程，空闲显存至少12GB。只在预登记技术错误条件下处理所属run，低性能保留作为结果。

已有4项聚焦检查PASS，验证真实CE梯度、单样本推理、退化输入、状态不更新、schema、scratch/target/augmentation拒绝和源选择不读取target。详见[本地验证](evidence/local_validation.json)、[逐行登记](experiment.json)和[原理及设计](../../../docs/CVS_BALANCED_IDENTITY_RESEARCH_20261001.md)。独立P0/P1审查及发布证据将记在本run。当前没有正式E200或新clean结果。

N607已用短连接独立核实普通用户szu2070436088/host dell-DSS8440、项目路径、磁盘及8块GPU均无计算进程。既有PowerShell preflight包装在解析ssh -G时失败；改用Python subprocess argv读取相同配置，核实目标与identity后直接认证成功，未变更SSH配置或身份。

独立P0/P1审查已完成PASS，审查者独立运行4项聚焦检查全部通过，无阻断项，见[审查证据](evidence/independent_review.json)。
