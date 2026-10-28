# Phase3 完整PAMF的源端开放集开发数据

2026-10-29。四类CE20代理适合隔离未知TX和接收机，但它不是当前完整Tri-SIG-TIME SSL的复现，训练预算也远小于后者。下一路线回到用户满意的当前六类完整PAMF，保留其权重和默认身份头，另外准备与正式未知设备不相交的source辅助设备，用于开放集开发。这一数据run为`20261029-phase3-source-oe-data-r01`，只制作源数据，不加载模型、训练或预测。

## 新数据依据与划分

对N607四个compact文件仅做前262144字节的静态pickle header解析，没有unpickle IQ。ManySig／ManyTx／ManyRx／SingleDay分别含6／150／10／28个TX，后三者及ManySig都在ManyTx的150类内。Full实际包含174类；差集24类均与旧六类和正式四个未知类不交。metadata证据在`local_artifacts/pamf_phase3_source_oe_metadata_20261029_r01/`，包含每个源RX/date单元实际Full计数；该身份差集不等于穷尽全项目历史未暴露。

原source RX为1-19、18-2、19-2、2-19、3-19，日期D1／D2／D3不变。在24个候选中，恰有以下7类在15个RX/date单元均不少于70条。只按集合、容量和字典序分配，不看IQ或识别成绩。

|TX|单元最小数量|15单元总数|角色|
|---|---:|---:|---|
|11-14|998|16332|辅助暴露训练|
|14-19|155|2438|辅助暴露训练|
|17-20|268|4520|辅助暴露训练|
|18-19|148|2476|辅助暴露训练|
|19-15|127|2180|源端新设备验证|
|19-16|235|4222|源端新设备验证|
|7-1|237|4062|源端新设备验证|

每TX/RX/date固定随机选择70个记录，训练辅助4类共4200条，验证辅助3类共3150条。两个角色按TX隔离，旧六类与正式11-4／3-13／12-20／5-20全部排除。正式目标RX24-18不读取；原source L_s/U_s/V不读取或改标，特别不解封原U_s标签。

## 制作与公开边界

复用既有Full原始验证，仅按成员索引读取5个source RX×3日共15个PKL，逐个释放，不重扫20个ZIP或新增hash链。canonical接收记录为FullWiSigEq／TX／RX／date／eq1／index；训练与验证全局物理互斥。选择seed为2026102905，与角色无关地按RX/date/TX确定抽样；不按质量筛样，不因结果选择性重采。

只执行匹配source的iq_record一次：原256点eq记录变为FP32[2,256]和unit RMS，不重新裁剪或均衡。这里保存的是source预处理IQ，不生成星地received；后续若做source信道增强仍属于同一物理记录，不计额外样本。目标端既有practical residual固定received保持原身份。

每role/public仅有iq.npy、ids.npy、metadata.npz。metadata只含合法source域标签rx_i／day_i／domain；RX对应原Main index1／3／4／6／8，不能把Full index1／10／13／16／33冒充Main编号。opaque ID与整体混洗不公开TX或TX分块。逐条真实TX只在role/truth/truth.npz和私有builder报告保存，用于后续独立按TX宏评分；算法不获得辅助设备身份标签。

辅助训练角色本身提供“源端未知”的成员监督，**这是额外源监督，不是零源标签方案**。目标仍零TX标签。源验证辅助TX不进入训练，新拒识器的数据、参数、成本和无OE对照必须单列。当前六类身份网络与默认头在本数据阶段完全不改，后续方法另立run；不将辅助7类当作目标注册成功。

## 后续方法边界

[Outlier Exposure原论文](https://arxiv.org/abs/1812.04606)使用与测试异常分布分开的辅助异常训练检测器，这只提供研究依据，未证明对本RFFI输入或跨RX星地条件有效。我们后续先固定当前六类PAMF，比较无OE与有OE的拒识读出；源known校准仍使用合法V，辅助新设备验证只用于源开发，不读取正式target。

源端浅中层crossRX监督与双前向RDrop的四类匹配诊断保留为备选；它可以复用L3360，却不能叫完整PAMF-Sup或TriSSL。原生完整PAMF11100步、每L阶段六次前向及U机制，均不能与当前四类CE20/540步混称同预算。当前priority是直接利用完整六类基座的新开放开发入口。

## 当前状态

READY_TO_BUILD。科学边界已在PROJECT_PROTOCOL的Phase3源端辅助开放集开发节定义，builder及6项针对性测试已实现并通过（10个subtests），含7350次真实iq_record合成验证；本候选一次独立P0/P1审查PASS，审查者独立运行6项通过。尚未读取真实15个PKL、构建7350条辅助源数据或训练拒识器。原始数据可用和容量事实不代表开放识别性能，95%目标不变。
