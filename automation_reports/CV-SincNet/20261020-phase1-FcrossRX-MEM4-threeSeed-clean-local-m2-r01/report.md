# MEM4 固定三种子补充实验

用户在首种子 clean 结果公开后明确要求“要跑三种子，单种子提升难以说服”。本 run 固定补齐 MEM4-s2026092702、MEM4-s2026092703，不再采用旧源域扩种子门槛。旧 run 的源域 FAIL、首种子及 ANCHOR 对照全部保留；不修改历史输出。

复用原 MEM4 seed2701（85.3268%，同seed BASE85.1393%）和匹配本机 BASE三种子。三种子合并前不宣布稳定提升。本次为暴露测试基准后的用户指定扩展，不声称独立盲测。

科学配方完整复用原 MEM4：F（三路SIG_TIME＋统一居中/物理单位Sinc）＋RDrop1＋跨RX同TX，当前锚点有梯度，当前与最近3个L批次的键停止梯度；独立层位/随机前向记忆，物理ID去重。temperature=.2，5轮预热、10轮升权至.1，普通增强、LS.01、旧门控＋自身EMA＋cos。只用可见L构建记忆，U真值隐藏，默认头正常联合训练。无星地增强，测试仅clean。

|新训练行|预算|后续|
|---|---|---|
|MEM4-s2026092702|从零50轮＝33L＋17U，11100步|固定final50→clean168000预测→独立CPU truth-last评分|
|MEM4-s2026092703|从零50轮＝33L＋17U，11100步|同上，负结果完整保留|

已有MEM4-s2026092701不重训，BASE三种子不重训；不另训头，记忆不导出，推理结构相同。

执行：本机验证过的Python3.10/PyTorch2.10cu128，FP32/noTF32，CPU线程2、加载workers0；物理GPU0所有owner最多3个source/predict worker，dispatch前freeRAM≥4.5GiB，CPU scorer最多1个。健康SYM4不停止、不重启、不热改。新run与release均不可覆盖。实际路径仅保存在本机私有运行记录，公开相对别名见experiment.json。

代码标识：本实现提交（发布时写入Git commit）；入口 experiments.cvs_f_memory_confirm50.driver，完整参数、CWD和固定路径见experiment.json execution/code。独立P0/P1只审本次接线与固定扩展，原机制已审且CPU/CUDA通过的证据复用。Focused checks：两剩余配方逐字段科学等价；原MEM4参数、zero-input logits/features/export精确相同；13项协议负例；真实匹配BASE02权重无query检查一次通过并丢弃，不继承到新训练。

停止规则：数据角色、物理ID、原生元数据、来源、配置、输出冲突、非有限数值、预算/Sinc/EMA/记忆错误或worker技术失败时停止新dispatch，保留产物及健康worker。资源未知则等待；不按性能停止、调参、选择性重跑。

自动收尾：每行训练并冻结后立即进入clean预测队列，预测全量固定后独立连接truth。最终合并旧MEM4首种子、新MEM4两种子与各seed匹配BASE，报告逐种子差、均值、样本SD、正差数、逐TX召回、Macro-F1、最差RX/day、U曝光/接受率及完整训练成本。必须三种子齐备才产出最终汇总。代码、配置与汇总结果提交推送并独立核对远端HEAD；原始IQ、权重、日志和预测不入Git。
