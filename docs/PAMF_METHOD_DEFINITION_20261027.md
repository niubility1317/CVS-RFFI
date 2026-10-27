# 当前 PAMF：三视图 SIG-TIME 与 Tri-SIG-TIME SSL

2026-10-27 用户最新指定：PAMF 默认恢复为三视图 SIG-TIME；完整半监督方法固定为此前的“三视图＋原版 REPL＋固定时间打乱”。本定义覆盖此前单视图／普通 PL 作为当前默认的约定。监督版保持既有机制，普通 PL 保留为明确的独立基线。

## 统一名称

半监督方法命名为 **Tri-SIG-TIME SSL**，中文为**三视图 SIG-TIME 时序重排半监督学习**。

英文完整名称：**Tri-view SIG-TIME Semi-Supervised Learning with Fixed Temporal Permutation**。

论文主表中完整方法简称 **PAMF**；首次介绍写作“PAMF with Tri-SIG-TIME SSL”。Tri 表示三个独立学习的 SIG-TIME 描述视图，Fixed Temporal Permutation 表示固定局部特征时序重排，不是伪标签或测试样本的全局重排。

|正式展示名|用途|数据与训练目标|签名视图／局部顺序|
|---|---|---|---|
|PAMF-CE|神经网络结构对比、结构分支消融的纯 CE 基线|仅 L_s 的真实标签 CE；不使用 U_s；关闭特征对比、KL、EMA、伪标签及其他辅助目标|三视图骨干；默认保留原自然顺序，显式结构消融按该行声明|
|PAMF-Sup|主实验中的监督版|保留此前完整有标签机制：跨 RX 身份对比、ordinary/raw 配对 KL、As 真实标签 CE 且 As KL=0、LS0.01；U 完全关闭|三视图；保持原自然顺序|
|PAMF-PL|普通半监督基线|PAMF-Sup＋普通置信度伪标签 CE，阈值0.95、权重1、伪标签LS0；不额外加入 REPL 机制|三视图；保持与监督版相同的自然顺序|
|PAMF|完整主方法|完整有标签机制＋Tri-SIG-TIME SSL，即原版 REPL 的全部机制＋固定局部时序重排|三视图；固定重排，训练和推理一致|

“纯 CE 监督”与“PAMF 监督版”是两种明确不同的实验条件。PAMF-Sup 保留既有有标签辅助目标，不能标为 CE-only。PAMF-PL 只在 PAMF-Sup 上增加普通 PL，不改变监督版的结构顺序或其他机制。不得把完整方法的固定时序重排静默加入历史监督／PL 对照。

纯 CE 允许使用预登记的有标签输入增强和信道视图，其标签仍来自同一 L_s 物理记录；损失只能由真实标签 CE 组成。历史严格 CE 控制的 LS0、辅助信道槽 CE 权重0.68按实际行保留，不把数据增强说成半监督。新矩阵另行明确增强开关，不因本次命名擅自增删。

## 三视图网络结构

输入为FP32 I/Q `[B,2,256]`，采样率25MHz，沿用原数据预处理。保留时域Sinc/CNN、频谱、PA记忆多项式、相对参考响应及原联合余弦分类头；最终身份特征160维、六类头尺度30。六类完整识别器232993参数，三视图签名模块55968参数。

三个签名视图读取同一份t2特征 `[B,96,32]`，各自独立学习96→6投影，形成4个长度8窗口。每窗口7维一阶总增量＋21维二阶反对称面积，共28维；四窗口112维，三视图拼接336维，再映射336→160，经归一化及有界逐维增益残差融合。三个视图不是三份独立IQ样本，也不是REPL的六次学生前向。

完整PAMF使用固定排列 `[0,5,3,6,2,1,4,7]`（零起始索引），生成种子2026102207；窗口首尾不动，重排特征路径后附加0至1时间坐标。所有模型种子、L/U及推理使用同一排列。原IQ和整个时域CNN不被重排。

详细逐层尺寸及参数分解见[已完成结果的结构核对](../automation_reports/CV-SincNet/20261027-phase1-pamf-repl-shuffled-views-m16-r02/results_analysis.md)。固定重排没有新增可训练参数。

## Tri-SIG-TIME SSL 的固定机制

保持已有实际执行配方，不借更名增加或删除机制：

1. 当前学生eval/no_grad生成硬伪标签，阈值0.95，U批量256；TX真值始终隐藏。没有EMA教师或原生历史U门控。
2. REPL阶段U总权重6，伪标签LS0.01，保持原CE剂量归一化1.25及六次学生前向、ordinary/raw配对KL、浅中层跨RX对比。
3. 对接受的同伪身份、不同源RX样本进行保护配对；浅中层MixStyle概率0.5、Beta参数0.1，六次U前向共享同一配对方案。
4. 在部署使用的160维完整融合特征上计算对比目标，权重0.05；沿相同合法配对进行接收机特征外推，alpha0.25，使用原默认头，推理不增加网络或分类头。
5. 以完整当前L目标为锚，按t2／分类头／其他参数分组处理U梯度，U/L范数上限1。U的As槽维持原raw IQ安排，不新增U信道增强。
6. 网络采用上述三视图SIG-TIME和固定局部时序重排，训练、冻结预测保持一致。

现有已完成证据的训练日程为20＋10＋20＝E50、11100更新：E1–13仅L；E14–20普通PL；E21–50完整REPL。后37轮均联合L/U。L批量128，原默认头始终与骨干联合训练，不另训头。

这些是现有配方定义，不把模型种子、数据划分、训练预算或增强设置自动扩展到未来所有补充实验。后续预算和矩阵由用户指定的补充范围预登记。

## 证据范围与历史名称

当前完整方法的已完成证据是 `20261027-phase1-pamf-repl-shuffled-views-m16-r02` 中 **view_count=3 的8行**，模型seed2026092701至08。Main split直接识别，clean＋六practical，每景cross_rx126000与cross_both42000，分别统计并另给168000合并口径；8种子clean86.6307±1.0082%，六景平均70.8577±0.4990%。

同run的单视图8行保留为单视图结构对照，不标为当前完整PAMF。历史原顺序REPL、T1、普通PL、单视图论文计划均保留原数据、权重、目录、执行ID及成绩。新名称只增加有范围的展示别名，不改变历史训练身份。

本次是用户在已公开结果之后指定当前主方法，不声称独立盲选。当前普通PL是自然顺序三视图3种子，仍缺“普通PL＋同固定重排”的方法归因对照；不能仅凭现有组合差值把收益全部归于REPL。

DURA v1.0仍是独立恢复器。当前三视图8seed证据仅为直接识别；历史DURA＋其他F的结果不会自动变成DURA＋当前PAMF，需要该配对的独立证据。

## 后续补充实验的名称与规则

|实验分组名称|固定解释|
|---|---|
|Network architecture comparison (CE only)|仅纯CE、有标签源数据；不使用U，不引入半监督|
|PAMF structural ablation (CE only)|三视图骨干的结构分支对照，按每行结构声明；纯CE、U关闭|
|Learning regime comparison|明确区分PAMF-Sup、PAMF-PL、PAMF，不与纯CE网络面板混称|
|Main split|标准数据划分；已完成当前完整PAMF的8seed直接识别证据|
|Receiver split 1／2|两套已定义接收机重划分，沿用实际集合|
|10-class setting|十类设置|
|1% labeled／5% labeled|对应标注比例|
|1／2／3／4／6 source receivers|对应源接收机数量；5源点按原Main模型／共同目标口径另行标明|
|Seven-scenario robustness|clean及六种practical逐景，注明数据划分、学习方式及读出|

示例：`Main split · PAMF · Tri-SIG-TIME SSL · Direct inference`；`Receiver split 1 · PAMF-PL · Threshold 0.95`；`Main split · PAMF-CE · w/o PA branch`。

本次不新建或启动补充实验，不恢复已暂停的单视图普通PL队列，不改动其他owner任务。后续由用户从这些有明确含义的分组和设置中选择部分补充，不自动重跑整套主矩阵。

机器可读当前定义：[pamf_current_method.json](../configs/pamf_current_method.json)；固定半监督配方：[pamf_ssl_trisigtime.json](../configs/pamf_ssl_trisigtime.json)；名称映射：[pamf_paper_names.json](../configs/pamf_paper_names.json)。
