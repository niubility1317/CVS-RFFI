# 可迁移特征提取器目标

主目标保留：固定目标clean三model-seed均值≥84%，每个Phase1骨干≤50轮。基线是历史旧伪标签门控＋EMA＋cos，不能以纯CE替代。每次实现、配置、完整正/负结果提交并推送Git。

用户补充目标：Phase1类别W丢弃后，实际160维特征在普通cosine换头、固定少样本适应中仍有效，最终须验证旧＋真正未见新TX统一竞争。原头84%不能单独证明特征增强；来源/预算/数据和Phase2权限仍遵守PROJECT_PROTOCOL。

功能边界：fθ包含Sinc/时频/PA网络、base_norm/pa_norm/gain、ReferenceCVS response projection/gain和实际NoExplicitCFO输入策略；gW仅具体类别的weight/bias。本实现 ResidualPhysicalHead无bias，丢弃6×160 W，scale30作为新普通cosine统一常数。诊断h349不替代joint160。新feature_only路径复用原components，原训练仍CE联合学习，默认分类前向保持。

先做固定边界与换头验证：20261013-phase1-frozen-feature-rehead-source-manysig-m9-r01，9已冻结原终轮全部保留。L6300拟合新普通cosine头20轮，V27000评分；另5源RX轮换头拟合留出、K5/10、3固定support draws。不更新骨干，不打开U/target；所有RX/TX已经被骨干见过，此项不是新类/Phase2证据。

下一步采用用户给出的独立路线：临时support ridge分类器重建辅助训练（原骨干、原旧门控/EMA/cos不变），与PA三阶可学习跨延迟乘性交互独立消融；均从零≤50轮，不串接依据target挑选的权重。只有单独收益在换头后保留才组合。此对话先承担分类器重建辅助目标，另活动对话当前为时序/PA多尺度卷积，未擅自给其发送任务。复频谱跨频关系为后续独立结构路线，不把已做功率上下文attention当作相同机制。

新类边界：目前ManySig6TX全部参与原Phase1，包括隐藏U，均只能叫old。真留TX研究必须从新数据契约起从零训练，heldTX从L/U/teacher全部排除；当前权重不能作为真新类的基座。正式Phase2只读p2_min_v1/VALIDATED_ONCE固定LEO receivedIQ与合法support，不读source/clean、不用query拟合，old+new全类逐包独立竞争。不能把源域support/V或clean头重训写成Phase2零适配DG。

证据状态：频域功率上下文三seed原头基线80.6454%、局部79.5353%、全局79.4298%，完整结果20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01/final_analysis.md，未达84%，全部负结果保留。此结果不用于attention调参或选择性重跑。特征边界synthetic检查与一次P0/P1审查PASS；真实源域换头结果待回。
