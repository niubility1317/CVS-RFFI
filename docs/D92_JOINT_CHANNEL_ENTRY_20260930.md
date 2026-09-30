# Joint channel support入口与独立汇总

本入口实施[联合通道设计](D92_JOINT_NEXT_MECHANISM_20260930.md)。算法、实际运行配置和发布由主任务冻结；本文不宣称已经启动或完成实验。

## 入口边界

`tools/evaluate_d92_joint_channel_probe.py`只加载已有合法support五块缓存，调用`cvsrffi.d92_joint_channel_local_ridge`。不加载Phase1 checkpoint，不更新encoder，不读取query或源域样本。原LocalRidge分类头保留为R0；候选与R0使用相同物理support切片。A没有合法匹配预测器，A与B−A均写null/N/A，B0是本次旧support基线，不能充当地面A。

三个路径为R0、R_channel_seq、R_channel_reset。两条候选共享B_channel。C_channel继承B参数及近端锚点，C_reset使用零参数与零锚点；C两路共享同一个prepared。新增类数为0时直接复用B，不产生C拟合。真实K1只做数值诊断，不拟合或评分；one-shot proxy仅使用一份真实物理support，并精确复用原方法。

外折按每类物理ID排序分配，outer-held不进入prepare、训练、优化步选择或任何头统计。core负责B/C旧ID、标签及原始五块缓存的继承校验。内折held是训练监督，日志明确标为`INNER_SUPPORT_TRAINING_NOT_VALIDATION`。入口只在候选拟合结束后计算outer-held准确率、真实类别竞争间隔、相对R0的winner变化及正确/错误转换。

## 完整日志与失败证据

每个row保留`fit_trace.jsonl`、`compact.jsonl/csv`、`fit_stages.jsonl/csv`、`training_events.jsonl`、`training_events_compact.jsonl/csv`及`training.log`。标准输出还进入row的`probe.log`。完整逐步事件保存实际736维参数和梯度；紧凑事件保留损失、正确数及参数/梯度各块的范数、和、极值。未测值保持null。源域验证标为不可用，不额外访问数据补齐。

成功训练的计数和state在outer-held评分前登记。评分失败仍保留已完成更新、参数、prepared和阶段审计，不丢弃实际发生的训练费用。异常不触发自动重训、调整学习率或添加数值jitter。

## 独立汇总

`tools/summarize_d92_joint_channel_probe.py`先核对四row、160个parent完整marker及配置/来源绑定，之后才读取固定score。它从物理ID、原score、完整训练审计重新计算配对指标、实际拟合费用、优化更新和内外折隔离。OOF每个物理held只计一次；proxy先在parent内平均全部anchor，再等权平均parent。重复旧support和训练步不视为独立实验。

输出完整K×新增类数、model/cohort、receiver/scenario及old-only/new-present分层。训练目标与outer-held统计分开标记。资源报告分别给出实际CPU平台、训练/评分墙钟、峰值RSS、完整持久状态及新增地面数据载荷。部署包或传输字节未实测时为N/A，不把5888 B参数向量当作整个部署模型。

入口和汇总的合成测试由主任务在已验证环境串行执行；本实现worker不运行真实实验、SSH或Git交付。
