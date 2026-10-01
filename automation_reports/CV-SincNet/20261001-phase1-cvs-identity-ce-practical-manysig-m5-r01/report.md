# CVS身份骨干交叉熵对照实验

本实验回答：在与CVCNN-CE相同的物理数据划分、交叉熵监督、星地信道增强和优化预算下，CVS的身份网络是否具有优势。

## 固定方法与条件

| 项目 | 本次设置 |
|---|---|
| 网络 | 原生CVS M/lite_d身份骨干，branch_ablation=no_dac；域骨干不实例化 |
| 分类头 | 保留原生cosine分类头，scale=30；不传标签，关闭CosFace margin |
| 优化目标 | E1至79仅clean CE；E80至200为clean CE＋0.68×satellite CE |
| 关闭项 | DAOT、FastTrust、域监督／对抗、PL、EMA teacher、MixStyle、CRRA及全部附加损失 |
| 数据 | ManySig，6个TX；equalized=1，中心256点，逐包RMS归一化 |
| 源接收机／日期 | RX 1、3、4、6、8；day 1、2、3 |
| 目标接收机／日期 | RX 0、2、5、7、9、10、11；day 0、1、2、3 |
| 物理角色 | 原source_contract完全匹配；L/U/V＝6300/56700/27000，比例0.07/0.63/0.30；U不参与本次训练 |
| 信道 | 与CVCNN复用PracticalResidualAugment：residual/post_sync/noeq，25 MHz，augmentation_seed=receiver_seed=2027 |
| 实际增强日程 | E80至90：mid／low urban，p=0.60；E91至200：high／mid／low urban，p=0.80；E80以前不执行卫星前向 |
| 优化 | AdamW，lr=0.0002，wd=0.0001；cosine至0.000001；FP32，不裁剪梯度 |
| 预算 | batch=128，不丢末批；每轮50步、6300例，200轮，共10000步；不早停 |
| 权重 | 每行从零初始化，无checkpoint／teacher／EMA／旧原型继承，固定E200末轮权重 |
| 种子 | 392005历史参考单列；2026092701至2026092704用于四种子均值与样本标准差 |

身份骨干保留自身时域、频域和PA身份表征。它们属于身份网络内部结构，与被关闭的独立域骨干不同。原生不参与CE路径的诊断projection/head仍保留在state_dict中，不增加监督损失；报告区分总参数与实际梯度使用，不能把不存在的辅助指标写成已测量值。

同物理ID和epoch使用相同增强抽样。不同架构消耗不同随机数，初始化和batch随机序列不能称为相同。本实验比较的是身份网络及其原生分类头；不能把差异单独归因于某一卷积层。

## 预测、评分和结论

每个训练入口不构建target loader，不读取目标真值或历史评分。先保存最终权重和源域训练完成证据，再在独立预测进程中使用既有VALIDATED_ONCE Phase1 paired capsule。预测器只读取IQ和opaque ID，使用相同规则输出clean及唯一satellite观测上的六类预测。五行全部固定后，另起独立scorer连接truth。原始prediction保留，不用目标结果选权重、删除种子、调参或选择性重跑。

结果应给出四个固定种子的clean、星地总体、high、mid、low urban准确率与Macro-F1，保留每种子和每接收机结果，并报告相对已有CVCNN-CE的逐种子百分点差值及四种子统计。任何负收益同样保留。既有完整CVS（DAOT＋RC4）只作为描述性参考，它使用U和不同训练步数，不能把本次与其差值解释为同预算的机制因果贡献。

目标benchmark此前已公开评分，所以本次属于用户指定的固定配置架构对照，不冒称新盲测。尚未评分时，CVS是否具有优势记为待验证，不预告性能。

## 验证与执行状态

本地4项聚焦测试通过：身份骨干真实CE更新和checkpoint往返；E79／E80增强与损失边界；checkpoint／target／附加损失／预算负测及输出防覆盖；完整小型冻结预测→truth-last独立评分。登记结构和本地／远端Python载荷语法检查通过。一次独立P0/P1审查结果另附；不存在性能门槛或额外许可。

实际commit、PID、GPU、有效配置、日志增长及进度由本run的evidence读回记录证明；实验源训练、预测、评分状态分别列出。失败仅处理所属行，保留产物，无性能停机和自动重启；不影响其他实验。

详细配置、六类seed角色、数据来源、命令、路径、launch owner及停止规则见[experiment.json](experiment.json)。日志保存逐步JSONL、每轮紧凑JSONL／CSV及完整文本stdout，记录clean／sat CE、权重、学习率、梯度、源域V、耗时与峰值显存；未启用项明确为false或N/A。
