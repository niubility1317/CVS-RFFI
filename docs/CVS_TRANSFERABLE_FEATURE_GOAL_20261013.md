# 可迁移特征提取器目标

主目标保留：固定目标clean三model-seed均值≥84%，每个Phase1骨干≤50轮。基线是历史旧伪标签门控＋EMA＋cos，不能以纯CE替代。每次实现、配置、完整正/负结果提交并推送Git。

用户补充目标：Phase1类别W丢弃后，实际160维特征在普通cosine换头、固定少样本适应中仍有效，最终须验证旧＋真正未见新TX统一竞争。原头84%不能单独证明特征增强；来源/预算/数据和Phase2权限仍遵守PROJECT_PROTOCOL。

功能边界：fθ包含Sinc/时频/PA网络、base_norm/pa_norm/gain、ReferenceCVS response projection/gain和实际NoExplicitCFO输入策略；gW仅具体类别的weight/bias。本实现 ResidualPhysicalHead无bias，丢弃6×160 W，scale30作为新普通cosine统一常数。诊断h349不替代joint160。新feature_only路径复用原components，原训练仍CE联合学习，默认分类前向保持。

先做固定边界与换头验证：20261013-phase1-frozen-feature-rehead-source-manysig-m9-r01，9已冻结原终轮全部保留。L6300拟合新普通cosine头20轮，V27000评分；另5源RX轮换头拟合留出、K5/10、3固定support draws。不更新骨干，不打开U/target；所有RX/TX已经被骨干见过，此项不是新类/Phase2证据。

下一步采用用户给出的独立路线：临时support ridge分类器重建辅助训练（原骨干、原旧门控/EMA/cos不变），与PA三阶可学习跨延迟乘性交互独立消融；均从零≤50轮，不串接依据target挑选的权重。只有单独收益在换头后保留才组合。此对话先承担分类器重建辅助目标，另活动对话当前为时序/PA多尺度卷积，未擅自给其发送任务。复频谱跨频关系为后续独立结构路线，不把已做功率上下文attention当作相同机制。

新类边界：目前ManySig6TX全部参与原Phase1，包括隐藏U，均只能叫old。真留TX研究必须从新数据契约起从零训练，heldTX从L/U/teacher全部排除；当前权重不能作为真新类的基座。正式Phase2只读p2_min_v1/VALIDATED_ONCE固定LEO receivedIQ与合法support，不读source/clean、不用query拟合，old+new全类逐包独立竞争。不能把源域support/V或clean头重训写成Phase2零适配DG。

证据状态：频域功率上下文三seed原头基线80.6454%、局部79.5353%、全局79.4298%，完整结果20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01/final_analysis.md，未达84%，全部负结果保留。此结果不用于attention调参或选择性重跑。特征边界synthetic检查与一次P0/P1审查PASS；真实源域换头结果待回。

阶段证据更新：源域冻结换头20261013-phase1-frozen-feature-rehead-source-manysig-m9-r01全部9row/297readouts/279heads已完成；实际source特征差0，featureupdates0、旧W删除、targetaccessfalse，完整结果报告已推送5cb8b0ba8989767e63b1d0e7cc7016ef56d41a0d。基线source原头97.8864%、新cos头98.0580%、原型96.3951%、5/10shot头95.48395%/95.97860%；均不是target指标。

路线六数学组件已实现但尚未接入源训练或启动：dual ridge=.1、scale10、normalize eps1e-4、160维/6类，线性solve不inverse，无持久head参数。focused primal等价误差7.11e-15、两侧梯度范数.380812/.365234、数值梯度、类置换/支持顺序置换/逐query独立/重复及秩亏支持/zero finite/外部autocast和19 negatives PASS。不把数学组件PASS写成实验性能收益。后续冻结cadence/sourceRX采样/辅助权重/预算与同预算基线读出矩阵后再发布。

当前新冻结矩阵：20261013-phase1-episodic-feature-legacycos-clean-manysig-m9-r01，SAME_RX/CROSS_RX各3scratch50、baseline3同seed终轮复用；每4step辅助episode2775、sourceL12support/24query、ridge.1/scale10/weight.2。全部sourceRX轮换，主RNG保留，分类头结构不变。各row自动原头/全L普通cosine新头/原型三读出clean后truth-last；84%主指标固定CROSS_RX原头，但原头达84仍不单独证明新TX迁移。另一活动对话最新路线为GN信息恢复/分支竞争抑制，与本路线独立。

路线六完整结果：20261013-phase1-episodic-feature-legacycos-clean-manysig-m9-r01 已完成全部 9 行/27 读出、4536000 决定、486 聚合记录独立重计；6 新 scratch50，共 66600updates/16650episodes。CROSS_RX 原头80.7494%（基线80.6454%，+0.1040pp，1/3正）；新普通cos78.0115%（+0.2478pp，1/3正）；原型75.7250%（+1.2863pp，3/3正）。未达84%，普通换头无稳定提升，不能声明新TX或Phase2迁移完成。负/边际结果完整上传；不追加目标反馈调参。当前先只读核实现有真实未见TX capsule与support-only普通cos接口，正式特征迁移证据尚未完成。

下一条独立结构路线已实现并登记 20261013-phase1-cross-frequency-legacycos-clean-manysig-m9-r01：功率化前cross-frequency复关系/同容量power关系，各3scratch50，baseline三读出完整复用；fθ新增45504params，旧W完全丢弃后实际joint160 maxdiff0。Vhealth-only，自动clean原头/普通换头/原型truth-last；尚未发布、尚无性能结果。已有practical Phase2 capsule不冒充LEO_WEAK，真正未见TX正式迁移证据仍未完成。

复频谱路线已发布 VERIFIED，执行code 9c171f9f13c7675eb06a4f2fc82d31105919becf；6scratch50新行已核实真实进程与epoch10至12日志增长，baseline三读出完整复用闭环。所有新行方案已固定，不读新目标分数调参，训练后自动3读出clean评分。当前未完成，无84%或新TX/Phase2收益声明。
