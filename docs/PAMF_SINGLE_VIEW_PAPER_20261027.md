> 历史已暂停计划：默认结构现已由用户改回三视图，完整PAMF使用Tri-SIG-TIME SSL；网络比较／结构消融仅纯CE、U关闭。以下保留原单视图120行计划，不启动或自动恢复。见[当前定义](PAMF_METHOD_DEFINITION_20261027.md)。

# PAMF 单视图默认结构与论文实验重跑

2026 年 10 月 27 日，按用户最新要求，后续未限定的 PAMF 默认采用 **单视图 SIG-TIME**，保留时域、频域、功放和参考响应及原默认分类头。半监督仍使用普通置信度 PL，不使用 EMA 或历史复杂门控。历史单视图、三视图、T1、REPL 权重及成绩保留原身份，不更名冒充新结果。

## 默认结构和两个损失面板

- 完整单视图 PAMF：196001 个参数（含六类原分类头）；单个 96→6 投影、四窗 112 维路径描述、112→160 投影，路径模块 18976 参数。十类头另增加 640 参数。不是删除 SIG-TIME，也不裁切三视图权重。
- **Main comparison：**保持既有完整有标签学习骨架，只把三视图改为单视图；监督与普通 PL 各三种子。保留 ordinary/raw 成对 KL、As 真实标签 CE／独立 KL0、LS0.01、浅中层跨 RX 身份对比。
- **Branch ablation：**有标签目标仅 `CE(ordinary,y)+0.68 CE(channel_slot,y)`，LS0；无 CRIC、成对 KL、EMA、U 熵或其他辅助目标。监督组无 U loader；半监督组在同一有标签 CE 上加入普通 PL 的交叉熵，不另加一致性。
- 普通 PL：当前学生 eval/no_grad、阈值 0.95、权重 1、整个 U batch 平均；L128、U256，前 13 轮仅 L，后 37 轮联合 L/U。全部模型 E50／11100 更新、20＋10＋20 分段，原默认头联合训练，不另训头。CE 指损失，不代表关闭输入或信道增强。

## 可读的学术命名

|旧定位字段|新展示名称|中文|
|---|---|---|
|F0|Main split|标准划分|
|F1_rx_repartition_r01|Receiver split 1|接收机划分一|
|F2_rx_repartition_r01|Receiver split 2|接收机划分二|
|ManyRx|10-class setting|十类设置|
|label01 / label05|1% labeled / 5% labeled|1%／5% 标注|
|m1/2/3/4/6-natural|1/2/3/4/6 source receiver(s)|源接收机数量|
|supervised / pl|Supervised / Semi-supervised (PL)|监督／普通 PL 半监督|
|raw / DURA-L / DURA-LU|Direct inference / DURA trained on L / DURA trained on L+U|直接识别／仅 L 恢复器／L+U 恢复器|
|cross_date / cross_rx / cross_both|Unseen date / Unseen receivers / Unseen receivers and date|跨日／跨接收机／双跨|

示例：`Main split · Full PAMF · Semi-supervised (PL) · Channel augmentation`。原 run/output 路径不移动、不重命名；旧字段只用于数据句柄兼容与附录追溯。历史 F1/F2 不能改称当前 Receiver split 1/2 而隐去实际集合差异；T1、F01–F21 等历史方法改用其实际机制名称展示，原编号仅放追溯列。机器可读映射见 [pamf_paper_names.json](../configs/pamf_paper_names.json)。

## 固定矩阵和复用

科学 run：`20261027-phase1-pamf-single-view-paper-m120-r01`。唯一逐行矩阵见 [experiment.json](../automation_reports/CV-SincNet/20261027-phase1-pamf-single-view-paper-m120-r01/experiment.json)。

|面板|模型单元|新训练|复用|
|---|---:|---:|---:|
|11 数据设置：增强监督／普通 PL，各三种子|66|60|6|
|Main split：无信道增强监督／普通 PL，各三种子|6|6|0|
|8 种结构：纯 CE 监督／CE+PL，各三种子|48|42|6|
|合计|120|108|12|

模型种子为 2026092701／02／03，split392005。主比较 11 设置保持原数据集合：标准五源、两套重划分、十类、1%／5% 标签及 1／2／3／4／6 源自然预算。5 源曲线复用 Main split 的同种子模型，仅重汇总共同六目标，不新增训练点。源训练量随源 RX 数变化，不解释为固定样本量下纯接收机多样性的因果效应。

8 结构为 Full PAMF、w/o Time coordinate、w/o Temporal signature、w/o PA branch、w/o Reference response、Moment pooling、Shuffled temporal order、Phase-randomized reference。除删除路径的控制外均保留一个描述视图；矩统计控制为单视图、相同参数量，不使用旧 COV_X3。

六个 Main split 完整监督骨架单视图模型已由既有 owner 完成，复用 `20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01`。CE 监督完整单视图和无路径各三个模型，复用 `20261024-phase1-PAMF-REPL-completion-m201-r01` 中各自实际严格 CE 行。不是复用 REPL 或三视图模型。所有复用核对实际源契约、同 seed、目标未参与训练／选模、E50／11100 和最终头，并在首次 source-only smoke 严格加载真实权重。

## 数据和测试

沿用 11 份既有 VALIDATED_ONCE 数据，源日期 2021-03-08／15／23，D0=2021-03-01。标准 L6300／U56700／V27000，源 RX[1,3,4,6,8]；重划分一源[0,1,2,5,7]，重划分二源[0,3,9,10,11]。U TX 隐藏，V 不更新模型。训练从匹配数据契约零初始化，新增模型不继承其他划分权重。

有增强每行测试 clean＋六 practical；无增强只测试／报告 clean。每行三种互斥群体：源 RX 的未见 D0、目标 RX 的已见日、目标 RX 的 D0。标准每景 30000／126000／42000，分列报告；168000 仅合并后两群，不与 198000 三群综合混用。D0 比训练日期早，不称未来外推。

每行保留 raw、同契约 DURA-L、DURA-LU，支持严格监督主系统、PL 主系统及固定恢复器权限四格。DURA、原七视图 IQ 和合法恢复缓存全部复用；新 G 训练和恢复缓存均为 0。监督 F＋DURA-LU 单列权限差异，不能称全系统 L-only。已完成的完全匹配预测按物理 ID 复用，只补缺失路由／群体。该行全部声明预测固定后，独立 CPU scorer 才连接 truth。目标结果不回流调参、挑轮、重排、筛种子或选择性重跑。

主比较沿既有三场景均匀 As；纯 CE 分支沿既有 practical residual 三场景 .40/.35/.25 及 .30/.60/.80 施加日程，0.68 辅助 CE 剂量匹配。无信道增强主比较将 As 槽替为 raw 保持真实标签 CE 剂量，普通输入扰动保留。训练损失差异如实分层，不将两面板当成同一训练配方。

## 发布和自动收尾

本地 Git 实现／配置提交后发布到 N607 普通账户的独立 release/run root；精确解释器、CWD、路径与实际 commit 记录于本地私有 runtime、远端 runtime 和 launch evidence。每张卡最多一个本 run 训练／预测进程，每卡总数最多两个且计入其他 owner；free≥12000 MiB，最多 8 个本 run GPU worker；CPU scorer 最多一个。不停止、接管、热改其他 owner 健康任务。

launcher 第一步真实 checkpoint 无 query smoke：源 L 前向，以及单视图各结构的实际 L128／U256 backward、纯 CE 分支无辅助目标、十类原头和监督 U0 检查。PASS 自动进入预登记完整矩阵，不以性能判断是否继续。一次独立 P0/P1 审查限直接错误；两个直接接口缺陷修复后只作原问题定点复核。禁止额外封签／哈希／报告审批门槛。

任意技术／输入权限／输出冲突／无法执行或无法产生合法预测的失败停止新派发，现有健康 worker 完成且产物保留；低分不停止。逐行训练结束有资源即预测，行全部预测固定后评分，最终自动生成三 seed 均值／样本 SD、逐景／RX／TX／日期、Macro-F1、净纠错及可读命名结果。所有源阶段／预测／评分和继承成本保留，复用不冒称历史成本为零。当前科学矩阵没有新训 DURA 或外部方法。

当前是方法默认与已授权论文复现；历史 2026-10-26 实验总览继续保留为时间点快照。新结果完成前，不将旧三视图成绩标成当前单视图成绩。
