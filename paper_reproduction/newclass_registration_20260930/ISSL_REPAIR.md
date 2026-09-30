# ISSL修复版与真实WiSig验证

2026-09-30，用户要求修复新类准确率为0时发现的代码问题。原作者仓库仍锁定`f58eda144063c7f150fbb7c0d0f31aaccb9d3bbe`且保持未修改；原版`acceptance.py`与r01/r02产物保留。修正版入口为`issl_fixed.py`，实现标识`issl_fixed_v1`。

## 修复内容

- KD保留作者温度交叉熵数值，直接返回可微loss，teacher仅detach；不额外乘T²，不改变原损失权重和温度。
- SSL每个batch执行`zero_grad(set_to_none=True)`，消除跨batch梯度累计。
- 负样本相似度使用`q @ queue.T`；初始queue和入队key均归一化，统一点积尺度；queue脱离计算图，精确保留最后K条。
- 固定teacher为eval且不可训练，独立增量分支结束后核对其参数和BN状态未改变。momentum key以no_grad前向，以EMA更新参数及浮点BN状态，整数计数复制，避免eval key使用永不更新的BN状态。
- 新分类头随网络加入optimizer，并记录新增行的梯度与实际更新。保留作者扩头方式、ResNet、logits对比空间、裁剪比例、旧训练样本重复、Adam/LR和阶段流程。

原版SSL首轮最大loss约14620，梯度最高约188126；这些是完整日志中的训练证据，并非query反馈。初始未归一化队列与后续单位向量混用，使负样本尺度不一致。两轮预算只包含很少的监督更新，原来新类为0不能单独归因于KD。

## 固定验证计划

|run|base|SSL|transfer|独立incremental|目的|
|---|---|---|---|---|---|
|r03|2|2|2|2|与修复前r02预算匹配的执行回归|
|r04|50|20|100|100|给监督新类学习足够更新的固定预算诊断|

两组在任何新query评分前预登记，均scratch；r04不继承r03。作者LR依次为0.03、1e-5、0.03、1e-4，损失权重、温度与StepLR保持原设置。不从query评分选择轮数、checkpoint、参数或重跑。旧3类、新3类、K=128，query每类64，RX1-1/2021_03_01，同一真实WiSig物理ID划分。无LEO，仅`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`，不作为正式CVS对比或原论文数据复现。

13项新增修复测试与22项既有行为/协议测试通过；独立P0/P1审查无阻断项。真实运行中需核对KD/新增head梯度、teacher不变、有限loss/梯度、全部预测及4个checkpoint恢复预测一致。完整逐步JSONL/CSV、文本stdout与按epoch紧凑JSONL/CSV均保留，训练准确率仅由训练/support标签计算。B阶段不在作者原流程中，记N/A，不虚构旧类单独适应结果。

- [r03登记与结果](../../automation_reports/CV-SincNet/20260930-diagnostic-issl-wisig-s392005-r03/report.md)
- [r04登记与结果](../../automation_reports/CV-SincNet/20260930-diagnostic-issl-wisig-s392005-r04/report.md)

ISSL论文全文及原始数据仍未获得。修复以作者实现中可定位的计算图、梯度和队列错误为依据，不宣称已核实原论文所有公式。实际学习效果以两份run报告与独立评分为准。

## 实际结果

|run|主分支旧类|主分支新类|H|独立incremental新类|训练步|
|---|---|---|---|---|---|
|r03|35.4167%|0.0000%|0.0000%|0.0000%|66|
|r04|100.0000%|100.0000%|100.0000%|75.0000%|1710|

35项测试通过；两组共8个checkpoint恢复预测逐条一致，全部日志已分析。r03证明修复机制生效但两轮新类仍为0；r04主分支旧/新均100%，独立incremental新类75%，确认固定预算下可以学习新类。预算增加与代码修复共同作用，不能将提升单独归因于某个修复。数据仅6类、同RX/同日、无LEO，未声称跨域泛化或原论文复现。详见两份run报告及[机器可读结果](issl_repair_results.json)。
