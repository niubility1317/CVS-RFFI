# Phase3源域类别与接收机留出开发

2026-10-28。属于完整Phase3目标的源侧研发步骤，不替代95%正式跨RX星地拒识、无标签适应、发现或持续注册验收。目标表不进入本阶段选择。本设计与上一轮提出的source-only留出路线一致。

## 任务与首轮矩阵

独立run为`20261029-phase3-source-holdout-s2026092701-r01`（10月28日设计，29日发布）。先只做一个model seed、一个TX-pair/RX留出fold，不直接展开三fold或正式目标矩阵。原source旧6TX中标签0/1（14-10/14-7）完全不参与该模型训练；训练其余四类，映射为0至3。源RX8完全不训练，训练source RX[1,3,4,6]；不读取任何target接收机记录。

只从原合法L_s6300条选择四TX×四RX×三日期训练记录，共3360条。原source split_seed实读为392005，model seed为2026092701，二者分别登记。U完全关闭、不打开。完整原V27000条仅在训练冻结后导出并作无参数更新的源开发校准/评价；其旧类2/3/4/5映射0/1/2/3，留出类0/1映射4/5。V不拆为训练集，不更新BN、原型或EMA。RX8的5400条V单独报告，其中known3600、proxy unknown1800。所有训练物理ID与V及留出角色互斥。

模型从零随机初始化，四类完整识别器232673参数、160维、三视图SIG-TIME、固定排列[0,5,3,6,2,1,4,7]、原无bias余弦头scale30。不得加载已见过留出TX的六类checkpoint、teacher或恢复器，不放宽历史6类reader/训练器断言。本开发器为L-only clean CE、LS0、无额外增强/半监督；不能把它称为完整六类Tri-SIG-TIME SSL或与其作等成本比较。

预算固定20个完整数据epoch，每轮含尾batch，batch128，因此27步/epoch、累计540步。AdamW lr0.0003、weight_decay0.0001、betas[0.9,0.999]、eps1e-8，每更新步cosine退火至0；shuffle seed2026102802。默认头随全部物理分支共同训练，无另训头。训练期间不forward V；最终epoch20固定，不早停或按V选择权重。

## 源开发几何与选择

从冻结模型的合法训练L特征计算归一化类中心；V不得拟合类中心或协方差。预登记比较五个原分数MSP/maxlogit/margin/energy/cosine，以及三个共享类内收缩协方差Mahalanobis分数，收缩系数固定0.05/0.2/0.5。后者只保留聚合类中心和全局协方差，不存逐样本源特征为部署依赖。协方差在L单位特征减去各自类中心后的残差上估计，按trace/d收缩，Cholesky白化后逐样本取最近类负平方距离；不存在query统计或成对query矩阵。

八种最终分数各自在V known部分固定99%保留分位数；V proxy unknown只用于源开发指标与是否推进，不参与网络训练或原型/协方差拟合。完整V中包含RX8 known，因此RX8仅对模型参数训练留出，不能把其校准后的指标称为完全未见接收机阈值泛化确认。报告完整V及RX8的known闭集/含拒识准确率、宏误拒、proxy unknown宏拒识、AUROC/OSCR。推进条件为总体及留出RX proxy unknown宏拒识≥95%、known宏误拒≤5%、已知正确率下降≤2pp；不达即记录NO_PROMOTION，不扩大矩阵或转target调阈值。

若首fold满足条件，按事前顺序补充留出标签[2,3]/RX6与[4,5]/RX4两个独立scratch fold；只在三fold一致满足后按源规则冻结一个候选机制。source V上的选择属于开发证据，不声称独立确认。最终六类模型、星地目标、无标签适应/注册仍需按完整目标单独验证；本阶段没有target性能结果。

## 文件、分工与接口

- 数据代理：`source_fold.py`与对应负测；`load_fold(cache, contract, fold)`仅构造source L/V，返回train_x/y/ids、validation_x/y/ids/receivers和派生角色说明，禁止U/target。
- 模型代理：`source_train.py`及对应测试；仅调用已验证scratch构造器，固定完整预算、源L无query smoke、训练/冻结、特征导出和最后完整标记。
- 主Agent：`source_geometry.py`与对应测试、最小登记、集成/版本/发布/监控、结果分析和后续推进。
- 独立审查：本新source候选仅一次P0/P1审查，不重复上轮冻结候选审查。

`source_features.npz`仅为本源开发私有产物：train_features/y/ids和validation_features/logits/y/ids/receivers。必须等`SOURCE_DEVELOPMENT_FROZEN`完整标记后才执行源规则评价。模型checkpoint、特征、原始epoch日志不提交Git。

## 资源、发布与停止

默认N607，首fold一个GPU worker，每卡本任务最多新增两个进程，不停止/修改其他用户工作。新不可覆盖run/release根，一次archive传输比较及compile。source模型从scratch起步，首步用真实合法L输入作无query前向smoke；不存在可以合法继承的四类checkpoint，不为满足形式而加载污染六类权重。前轮真实六类checkpoint smoke已完成，仅作为对应冻结模型的既有证据。

聚焦协议负测覆盖训练角色泄漏、V状态更新、预算/尾batch、初始化/输入越权和输出覆盖。技术停止仅限错角色/契约、无法运行/导出、输出冲突或同类系统故障；低科学表现不停止、不重启、不选择性改预算。固定训练结束自动源规则评价、报告与登记；无真实target预测，不读取或复用上轮target评分来选规则。

## 当前状态

LOCAL_VERIFIED。scratch构造、source派生/训练/规则评价/调度已完成，全部Phase3聚焦测试73项PASS；本source候选一次独立P0/P1审查PASS。真实N607 source metadata的15域、eq1、L70/V300每格、角色ID和split_seed392005均独立核实；未读取IQ/U/target做该格式检查。尚未启动此run，无source留出实测结果。源数据和目标基准分开，完整Phase3目标仍ACTIVE。
