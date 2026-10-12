# 固定三成员证据融合：零训练源检验

原source三成员probmean61.7778%，正式clean三primaryseed81.3254%，84未达成。本候选不改变权重、成员或数据角色，只检验类别证据合并；原source实际15个fixedfinal30 native、5RX每3个内部初始化成员、一个primaryensemble种子；不是三份独立sourceensemble。source模型/prob预测由原严格分析核验config/预算/noU/no target/冻态/物理ID对齐，真实matching sourcecheckpoint noquerysmoke另行先行PASS即自动接续。

三个公式在结果前固定，逐packet全6类同规则：MEAN_LOGIT平均原logits，相当几何概率共识决策；UNIT_MARGIN每模型每包6class logits减classmean、除L2norm clamp1e-8后平均；MAJORITY三个模型单包argmax票数+0.001固定probmean破tie。这里模型票数是同一个包的3份预测，绝非query真实类别批次数、配额或全局分配。无任何RX/role/querymetadata/truth、拟合/校准/权重选择/温度调参、模型或query更新。0新增训练/优化器/fit，原成员全部保留。

三个readout×五RX固定共81000源决策，全部持久化后sourceV评分，ID按实际heldV每RX5400核对。资格预登记：相对原固定probmean source均值>=+1pp、4/5RX非负、worst非劣1pp；完整15后只合格读出最高source均值冻结，完全并列按MEAN_LOGIT/unit/majority。未过不访问目标。通过才复用每primary2701/02/03各自3个原已验证全源final30权重（共9），不新增训练，在新不可覆盖目录按所选同一公式自动clean+六practical，各168000，全七视图固定后独立truth-last评分；不以target改readout/成员或选择性重跑。目标benchmark已暴露，非新独立盲测，也不称卫星部署。

源父15历史99900更新/450模型轮及正式9模型历史成本分别报告，不把读出零训练误写为方法无训练成本。一次独立candidatefamily P0/P1 PASS、局部query独立/几何等价/缩放与zero/majority及3权限负测、匹配真实checkpoint smoke，一次实际Git code/config、不可覆盖run/release/output、N607preflight/archiveSHA/compile/PID-CWD-argv-GPU-loggrowth。沿用初始GPU env独立worker的既有容量保护，<=3总进程含其他owner/free>=12000MB，普通账号唯一owner不改健康任务。仅按系统技术/权限/输出碰撞故障保留并停止所属run，不按分数停止。精确矩阵/命令/环境/CWD/路径/stoprule见experiment.json，无额外gate。
