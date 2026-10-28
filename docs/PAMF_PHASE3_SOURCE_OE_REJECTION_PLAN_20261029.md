# Phase3 当前完整六类PAMF的源端OE拒识开发

2026-10-29。候选拟为`20261029-phase3-source-open-readout-s2026092701-r01`。研究问题是：保留当前完整六类Tri-SIG-TIME SSL的身份网络与默认头，额外的源端未知成员监督能否改善对未参与读出训练的新设备的拒识。目标端零TX标签、95%拒识与旧类保护目标不变。本候选只做source开发，不能替代后续跨RX星地无标签适应或匿名注册验收。

## 基座和数据

固定当前Main split、三视图SIG-TIME、232993参数、完整六类默认余弦头、固定时间排列[0,5,3,6,2,1,4,7]的seed2026092701最终E50。基础模型仍只继承原source六类L/U/V契约；原L6300、V27000用于源锚点／阈值，原U_s标签不读。不能加载四类CE20代理替代这个基座，不能改写Phase1/2权重或声称CE代理是完整TriSSL。

辅助source数据为已验证`20261029-phase3-source-oe-data-r01`：4个训练暴露TX共4200条、另外3个源新设备验证TX共3150条，均来自原5源RX×D1–D3，均不在四compact的150TX并集、旧六类或正式四个未知类中。训练／验证TX互斥，每类1050条。训练器只能知道这批训练辅助样本属于“非当前六类”的源成员状态，不能获得每条辅助TX身份；3个验证TX只在全部预测固定后用于独立宏评分。这是额外源监督，不称零源标签；目标端不提供此监督。

## 特征导出和固定源几何

仅前向冻结F，先用合法L的2条完成真实checkpoint smoke，PASS后继续L6300／V27000／辅助训练4200／辅助验证3150，共40650条完整前向，smoke另2条。不加载DURA、teacher或原U_s，不读取RX24-18及其768／1920正式观测。四role全部导出后才写完成标记。

同一次完整前向捕获8个原始分支：fused160、time160、frequency160、pa160、reference29、signature336、spectrum128、pa_moments48。reference使用response(x)，坐标27保持0；spectrum取freq_gate之前4×32输入；PA取24通道均值／总体标准差。所有分类logits仍为完整6类默认头，不使用缺少reference／signature的核心中间logits。

仅L拟合每个分支的固定几何。fused保持原L2；其余7组先按L的全局mean/std(ddof0)标准化，std≤1e-8置1，再L2。各类归一化均值为mu_k，共享类内S分母n−6，C=.8S+.2max(trace(S)/d,1e-8)I，W=chol(C)^−1。保存mean、scale、6个原型和W，不能保存逐样本源特征作为部署输入。某raw分支的零类均值允许用epsilon规范化，不能产生NaN或删除其他分支。

每条描述符由每分支对6个原型的cosine降序6维、负Mahalanobis平方距离降序6维构成，共96维；再加原完整logits的MSP、maxlogit、top1−top2 margin、logsumexp，共100维。类别列排序使描述符对已注册标签置换不变，不按具体TX/RX设专属判决。变换逐样本使用冻结统计，不做批量／验证／query归一化。

## 两个匹配读出与控制

只训练linear_oe（100→1，101参数）和mlp16_oe（100→16 ReLU→1，1633参数），无dropout。原F和身份头完全冻结，新增参数、source成员标签及额外计算单列。描述符全局mean/std只由source L和辅助训练两组拟合，std≤1e-8置1；源V和辅助验证不进入这些统计或梯度。

损失为0.5×mean softplus(−g_known)+0.5×mean softplus(g_OE)，分数高表示known。CPU AdamW固定lr.001、wd.0001、betas(.9,.999)、eps1e-8；每模型固定200次全批量feature更新（每步6300＋4200），不验证早停、不扫参数。linear零初始化，MLP使用model seed的CPU RNG fork、Xavier uniform及零bias；不触碰全局CUDA RNG。Torch／NumPy桥保持已验证的tolist路径。

同时保留原MSP、maxlogit、margin、energy、cosine五个无OE控制，总计7条规则。每条只用known V27000分数的1% lower分位冻结一个全局接受阈值（99% retention）；不使用辅助验证或正式target校准阈值。原known V只做这种无模型更新的既定阈值校准，辅助新设备验证只评分。

先冻结几何、读出和阈值，再固定V与辅助验证全部预测，最后独立scorer连接known V标签及辅助验证TX，报告旧六类宏准确率／误拒、相对同输入原闭集损伤、辅助新3TX宏拒识、逐RX／TX及置信区间。已接受样本始终使用原完整身份头argmax，不另训或替换身份分类器。

## 科学推进

只有linear_oe／mlp16_oe是本次可推进候选，5个静态控制不能单独触发OE方法推进。候选需unknown宏拒识≥95%，每个源RX也≥95%；旧类宏误拒≤5%、含误拒准确率较同输入闭集下降≤2pp，整体和各源RX均保护。各辅助TX结果全部报告，不能按成绩剔除设备。两OE都过则按较小参数优先linear，否则只选达标者；未过则NO_PROMOTION，不读取正式target、不扩seed或扫阈值挽救。

上述只是源开发门槛。即使通过，也没有证明未见RX、星地三景、零标签适应或注册成功；后续仍需按总目标固定确认矩阵，旧类、新注册类与后续严格未知分别验收。正式unknown4与本source辅助训练／验证TX始终隔离。

状态：READY_FOR_RELEASE。导出器、读出核心、fit/predict执行器、独立评分和四阶段launcher均已实现；64项测试及35个subtests通过，本候选一次独立P0/P1审查PASS，审查者独立运行同64项通过。搬迁后仅按给定oe_data_root读取已验证数据，旧E/F路径只保留来源说明；不重验数据或增加成员hash。真实F6checkpoint两条Lsmoke是导出首步，PASS立即继续。当前尚未真实导出、训练或评分，没有OE拒识性能或95%达标证据；实际commit、命令、环境、GPU、不可覆盖root、停止规则由同名run的experiment.json固定。
