# 五条件冻结默认头：源域身份与退化归因预登记

本次只读取预定 D0/D1/D2/D3/D4 的 source final50，比较同一 L_dev720、固定 D1/A0 clean 正确集合和一份共同的六 practical 源视图。D0/D3/D4来自修正后的 r02，D1/D2保留原 A0/A-J 身份；原r01无通用增强D0诊断不进入比较。五个实际final50现已核实，本分析已自然完成。输入、逐模型来源、种子、命令、CPU及输出根唯一记录在 [experiment.json](experiment.json)。

固定网络 SIG_TIME、内部门控／参考响应与原 cosine6×160、scale30。严格 own_model 读取实际 checkpoint、source_contract、inner_source_contract、initialization、source_frozen/completion，检查同一5580/720物理折、从零继承、final50/11100、无target训练或选模、无拟合头。源 RX1/3/4/6/8、day1/2/3、6TX；U不开并检查None，V容器可能由现有cache物化但丢弃，V前向0。任何模型未冻结或实际折不同都在创建输出根前拒绝，不用历史full-L权重替代。

## 原头、身份关系与分层

先对真实 A0 原头做2样本无query smoke，PASS直接继续。固定A0 clean正确集合仅作为共同保持率分母，全部720样本仍进入总体与分层准确率；分别统计相对A0的错误修复和新增错误。原头margin为（真类logit−最强异类logit）/30，不将余弦变化称为头margin。

读取实际T/F/PA/参考/SIG/joint六个融合向量，记录同样本clean/ownEMA余弦、最近不同TX余弦、二者差、分支方差、gain及实际贡献幅度。另计算同TX正参照与异TX负参照共享RX/day、参照RX与anchor不同的平均余弦差；用分组均值精确等价于合法triplet平均，不拟合分类器，也不分配N³张量。u→c与c→u分别记录，不用同TX作负样本。

按RX、LOS/PARTIAL/BLOCKED、locked/degraded/unlocked和新增噪声SNR区间记录准确率、固定集合保持率、修复/破坏、margin和身份差。保留接收质量代理、signal_to_added_noise_db、残余CFO及核长度；新增噪声SNR不作为总SNR。实际共同源流2026101402、RX/day session、原residual/post_sync/EQfalseG1/referenceSNR30/边界/延迟不变；formal target IQ不读取、不修改。

## 梯度与零更新边界

mid/low urban 使用同一次真实 clean+urban 前向，D4 clean anchor保留计算图。源评价使用共同的0.5clean CE与0.5satellite CE作梯度参照；D0的satellite CE仅为假设性诊断，D0实际未训练它。D1额外目标0，D2使用原ownEMA clean-correct正配对及原固定J系数，D3/D4使用各自原固定源系数和当前同次前向stop-grad keys。额外W/gain梯度必须0，原训练的实际联合更新和clip/AdamW效果仍从完整native日志读取。

所有路径eval，无optimizer/EMA/head-fit更新，autograd.grad不累积.grad；完成后逐值比较模型与ownEMA state_dict并检查参数冻结。eval梯度不冒充训练Dropout、裁剪和AdamW更新。几何变近、方差变小或loss变低都不能单独证明身份恢复；需结合原头修复/破坏和不同TX距离解释，不能据此宣称唯一物理原因、可训练上界或跨seed有效。

## 预算、执行与当前证据

固定36722个前向：真实CP smoke2＋固定A0 clean720＋5×（student clean720＋ownEMA clean720＋六sat4320＋mid/low配对clean1440）。新增epoch、optimizer、EMA和拟合头更新均0。五模型的E50历史/新训练成本与本CPU诊断墙钟分开报告，不隐藏控制或技术开销。

N607普通账号、CPU2线程、CUDA_VISIBLE_DEVICES为空；不可覆盖run/output、唯一owner与精确Python/CWD见登记。冻结后按既有最小流程一次Git身份、一次archive传输比较/remote compile、首步真实CP smoke及独立PID/CWD/argv/日志读回。源/来源/冻结/非有限/预算/输出/技术错误只退出本owned诊断，保留产物，不自动重试，不修改健康训练。正式七视图仍由各训练行队列负责，本分析不重复测试、评分或改source-family决定。

本地5项针对性测试PASS（0.831s）：匹配RX/day关系与显式triplet平均一致、U/target RX拒绝、真实配对图D4 clean梯度与W/gain边界/状态不变、未冻结来源拒绝、固定集合修复/破坏分开计数、正确原A0配置实际构造信道。一次独立P0/P1审查发现并修复错误信道类/配置组合，原问题定向复核PASS。五个实际final50的本次源分析现已完成，结果见下；正式测试仍由原行队列执行。

## 实际完成与源域结果

VERIFIED：发布commit f172f98d67b19c65d7246b025a7ba2af770d197a，PID3116403自然结束，archive/compile/真实CP smoke通过；独立PID/CWD/argv匹配，日志81→218字节。CPU2、CUDA空；36722前向、83.073758秒，optimizer/EMA/head-fit0，U truth/query读取0、V前向0；五student/ownEMA状态完全未变。实际模块__file__为adv3b02_xuc/code/leo_practical。

|条件|源clean %|源low %|固定703保持 %|修复|新增错|原头margin/30|匹配参照身份差|
|---|---|---|---|---|---|---|---|
|D0|98.1944|48.7500|49.5021|32|164|-0.004115|0.082066|
|D1|97.6389|66.9444|68.2788|0|0|0.095764|0.178083|
|D2|97.9167|67.2222|68.2788|12|12|0.081752|0.117052|
|D3|97.6389|67.6389|68.8478|14|10|0.089000|0.258420|
|D4|97.6389|68.1944|69.2745|14|7|0.088870|0.259873|

D3/D4平均参照身份差提高，但原头margin下降，源low只净增5/9包；固定703集净增4/7包。BLOCKED D1/D3/D4为167/169/174正确（386包），<3dB组72/70/75（206包）。这是小量源错误恢复，不是足够正式增益；正式主报告保留两候选源条件FAIL及不扩种子。全局u→c/c→u聚合近似相同符合平衡集合对称性，不能用于区分两方向训练。

完整源几何/真实分支/gain/梯度明细、sample_identity.jsonl.gz、源resolved config在 E:/type10-7/local_artifacts/cvs_semantic_mechanism_20261015；9图共用实际artifact，已生成查看。图与正式全量表见[同轮主报告](../20261015-phase1-sigtime-semantic-e50-manysig-s2026092701-m5-r02/report.md)。0更新源评价不能证明唯一物理原因、可训练上界或跨seed收益。
