# 可迁移特征提取器目标

当前主目标：目标 clean 三 model-seed 均值 ≥84%，每个 Phase1 骨干训练不超过 50 轮。对比基线固定为历史旧伪标签门控＋EMA＋cos，不以纯 CE 替代。每次探索实现、配置及完整正负结果均提交、推送 Git，并独立核对远端分支 OID。目标仍未完成。

最新用户约束（2026-10-13）：分类头使用网络默认分类头，仅随骨干联合训练。禁止冻结骨干后专门训练分类头，禁止把 source 换头、support-only W 训练或其他分类器用作后续优化与评估步骤。Phase1 用原默认 ResidualPhysicalHead（6×160，scale30）直接 clean 推理，预测后没有 optimizer 更新。本约束覆盖此前 pasted design 的换头/临时分类器安排；历史已完成的诊断保留，不能替代默认头负结果或声称达标。未使用的 support_cosine 接口已删除，默认头纯推理入口在 experiments/cvs_default_head，政策代码 4a8235a4dab2fc9971e95d7b76ca5a22ff670732 已推送并核对远端。

## 优化对象与证据边界

研究对象是可迁移 fθ，包括 Sinc、时频、PA、参考响应、base_norm/pa_norm/gain 与实际 joint160 的类无关计算。h349 诊断向量不替代网络真正分类前的 joint160。保持默认头结构和训练算法，新增机制必须通过真实特征梯度与冻结后的默认头测试证明效果，不专门拟合 W。

ManySig 源域 6 TX 都已参与 L/U/V 或 teacher，因此只能叫旧 TX；隐藏 U 标签不等于网络没见过该 TX。真正未见 TX 不能在任何 Phase1 角色或祖先 checkpoint 中出现。新类/Phase2 证据仍未完成；未来按 PROJECT_PROTOCOL 的 p2_min_v1、VALIDATED_ONCE、固定 received IQ/support/query 权限和默认网络流程执行，不访问 source/clean，不用 query 拟合，不专门训练头。已查明 practical capsule 不能冒充 LEO_WEAK，旧弱场景 cache 的物理 ID 重叠不能直接作为当前正式数据。数据发现仅为只读元数据核验，没有启动新的 Phase2 实验。

源 V 只检查健康，不依其微小涨分选择模型；本对话采用已授权 clean-only 完整三 seed 矩阵。方案在目标访问前固定，测试结果不回流调参、候选排序或选择性重跑；已暴露基准不声明独立盲测。健康其他 owner 的进程保持原计划，每 GPU 总实验进程最多 3。

## 已完成探索

|路线|固定默认头三 seed clean 均值|结论与报告|
|---|---:|---|
|旧门控＋EMA＋cos LEGACY50|80.6454%|已核实 source 来源、固定 final50，后续同 seed 对比复用|
|局部/全局功率上下文|79.5353% / 79.4298%|均低于基线；20261013-phase1-frequency-context-eval-recovery-manysig-m9-r01，Git 60b2fa 已交付|
|SAME_RX / CROSS_RX 历史 episodic 特征目标|80.6204% / 80.7494%|CROSS 仅 +0.1040 个百分点，1/3 为正；完整 9 行结果 Git 752a8f20f6dcbc51abc195f6f78b597ed89ea6bd|
|同容量功率关系 POWER_RELATIONS|76.8591%|相对基线 −3.7863 个百分点，3/3 为负|
|功率化前复频谱关系 COMPLEX_RELATIONS|79.3058%|相对基线 −1.3397 个百分点，1/3 为正；未达 84%|

最新跨频 run 为 20261013-phase1-cross-frequency-legacycos-clean-manysig-m9-r01，执行代码 9c171f9f13c7675eb06a4f2fc82d31105919becf。6 个新 scratch50 模型全部结束，完整 66600 step 记录、450 epoch 记录、9 个真实默认头 checkpoint 与 1512000 个 clean 决定已独立核验；162 条默认头聚合指标重新计算一致。新增特征参数实际改变，无非有限跳步，所有 queue/worker terminal。完整结果见 [final_analysis.md](../automation_reports/CV-SincNet/20261013-phase1-cross-frequency-legacycos-clean-manysig-m9-r01/final_analysis.md) 和同目录 results.json。

历史 source 冻结换头（9 行/297 读出）以及 episodic/cross-frequency 的全 L 新 cosine/原型诊断均在最新约束之前已经完成。θ 与原默认 W 保持冻结，历史结果完整保留；它们不算当前主指标，也不会继续作为未来步骤。source-only 诊断不冒充目标新类或 Phase2。跨频历史额外 324 条结果单独归档 historical_readout_results.json，避免与默认头结果混用。临时 ridge 分类器路线也不再作为新增实验。

## 后续方向

继续沿源数据与结构分析设计特征机制，保持默认头与旧门控＋cos 对比。用户原列出的 PA 跨延迟三阶乘性交互属于尚未实施的独立结构路线，可与同容量卷积独立消融；浅层/中层域稳定机制须记录实际层位、梯度对象、RX 域分组及身份保护配对。任何新行都直接默认头预测，不能调用历史 source_freshhead/readout 入口。

另一活动对话“继续优化 Phase1 双骨干域泛化”当前路线是 GN 信息恢复/分支竞争抑制。本对话的功率化前跨频关系及后续 PA/特征机制与其不同；未擅自给其他对话发送任务。所有负结果保留，目标 84% 仍 active。
