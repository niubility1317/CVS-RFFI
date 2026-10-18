# Satellite dropout：完整 E30 单种子匹配结果

状态：ANALYZED；stage clean guard FAIL，独立source判据 PASS。模型seed只有2026092701，尚无可重复clean正收益证据。

当前目标为固定practical识别优先、clean相对同预算同seed无As最多下降0.1个绝对百分点；不再要求独立Ag或全部特征/逐包预测保持。

Run 20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01。复用NOAS/AS_RD两个原P43固定E30控制，新增AS_CE与AS_DET各scratchown20+10；相同fullL6300/U56700隐藏/V27000、全部RX1/3/4/6/8、day1/2/3、split392005、IQ256。F/SIG_TIME_X3/统一Sinc/joint160/NoExplicitCFO/defaultcos30骨干与头联合训练，0新头/部署分支。

## 核心结果

|视图|NOAS %|AS_RD %|AS_CE %|AS_DET %|DET−NOAS pp|DET−RD pp|DET−CE pp|
|---|---:|---:|---:|---:|---:|---:|---:|
|clean|85.247619|85.341071|85.214286|85.121429|-0.126190|-0.219643|-0.092857|
|practical_high|42.685119|81.405357|81.019643|79.198810|36.513690|-2.206548|-1.820833|
|practical_mid|40.995238|78.661310|78.345238|76.707738|35.712500|-1.953571|-1.637500|
|practical_low_suburban|39.472619|75.861905|75.639286|74.018452|34.545833|-1.843452|-1.620833|
|practical_high_urban|41.439881|77.767262|77.452976|75.819048|34.379167|-1.948214|-1.633929|
|practical_mid_urban|35.132738|62.806548|62.654762|61.638095|26.505357|-1.168452|-1.016667|
|practical_low_urban|30.139881|51.244048|51.170238|50.606548|20.466667|-0.637500|-0.563690|
|urban mean|35.570833|63.939286|63.759325|62.687897|27.117063|-1.251389|-1.071429|

AS_DET clean相对NOAS为-0.126190pp，净修复-212包；0.1pp在168000包上对应最多168个净新增错误，整数判定为net×1000≥−168000。mid/low增益分别+26.505357/+20.466667pp，urban均值+27.117063pp。

这是单seed结果；即使clean为正也不宣称稳定双提升，低于0.1pp的正变化明确视为微小观察。source判据未通过时按原规则关闭固定satellite-dropout机制，不根据正式成绩翻案、挑seed或调整层/概率/槽/预算。

|视图|DET F1 %|DET 最差RX %|DET clean−视图 pp|
|---|---:|---:|---:|
|clean|84.836798|77.737500|0.000000|
|practical_high|78.691672|71.712500|5.922619|
|practical_mid|76.242797|69.770833|8.413690|
|practical_low_suburban|73.595011|67.541667|11.102976|
|practical_high_urban|75.321924|68.887500|9.302381|
|practical_mid_urban|61.229309|56.458333|23.483333|
|practical_low_urban|50.276030|46.675000|34.514881|

全部四行/七视图的F1、最差RX、RX/day/TX分层及默认头margin保存在formal_metrics.csv和formal_summary.json；不以总体均值掩盖分层退步。

## 错误修复与归因

|比较|视图|修复|新增错误|净修复|固定NOAS clean正确集合的原/新保持率 %|
|---|---|---:|---:|---:|---:|
|AS_DET__NOAS|clean|3031|3243|-212|100.000000/97.735588|
|AS_DET__NOAS|practical_mid_urban|51200|6671|44529|38.619288/67.771757|
|AS_DET__NOAS|practical_low_urban|42924|8540|34384|32.807089/55.118143|
|AS_DET__AS_RD|clean|2638|3007|-369|97.408809/97.735588|
|AS_DET__AS_RD|practical_mid_urban|6460|8423|-1963|68.782119/67.771757|
|AS_DET__AS_RD|practical_low_urban|8342|9413|-1071|55.632751/55.118143|
|AS_DET__AS_CE|clean|1771|1927|-156|97.767708/97.735588|
|AS_DET__AS_CE|practical_mid_urban|5147|6855|-1708|68.575438/67.771757|
|AS_DET__AS_CE|practical_low_urban|6652|7599|-947|55.491007/55.118143|

AS_CE隔离satellite KL移除，AS_DET−AS_CE检验satellite dropout输出绕过；AS_DET−AS_RD是完整配方净变化。两新行均保留全部As标签CE、ordinary/raw/U原行为，原mask完整生成且6L前向相同；不把旧AsPairCE失败改称成功或归功于增加算力。相同IQ的两次As无dropout输出一致是实现检查，不是与clean保持一致的科学门槛。

## 源结果和物理分层

|source条件|判定|
|---|---|
|stage_clean|True|
|retained_As_gain|True|
|useful_vs_original|True|
|dropout_attribution|True|

完整V27000 clean来自原生冻结期指标，未保存每包完整V预测，不能声称从每包预测独立重算该值。固定源V1800/两draw的raw和三practical logits均独立重算，两个draw仍是1800唯一物理包。低urban状态/锁定及新增噪声比分层见source_metrics.csv和source_confusions.json，稀有/单例不删；source面板fc2.45GHz、训练/正式2.462GHz的既有差异保留披露。signal_to_added_noise_db不是总SNR，正式receivedIQ未保存逐包物理标签，源分层改善不是特定损伤的因果证明。

## 完整训练、验证与成本

全6660步/30epoch、两段原U/ownEMA、监督尺度、satellite-only作用、L/U出现序列及匹配随机流的核对见training_logs.json。未新增teacher、headfit或模型NN用于独立分析。新增与复用模型成本分列，不能把历史控制视为免费；两段optimizer重置和own20父成本计入各模型E30。实际训练/验证、教师和预测NN与wall分别列于costs.json。

正式独立重算504个矩阵、84次分区合并，源矩阵952个；全部logits/argmax/原scorer一致。source独立决定完成后才读取固定正式truth，原receivedIQ/SNR/信道/困难样本均未改。

未完成：跨模型seed复现、稳定clean正收益，以及任何尚未达到的阶段指标。本实验不建立新Ag或Ag×As交互，也不作真实在轨验证声明。

## 全部四行补充指标与成本

下表每格为 macro-F1／最差 RX accuracy，均为百分数；与前述全部七视图准确率配套，来源为固定正式504组CM独立重算，只有一个模型种子。

|视图|NOAS|AS_RD|AS_CE|AS_DET|
|---|---:|---:|---:|---:|
|clean|84.980／78.450|85.044／78.363|84.943／78.642|84.837／77.738|
|practical_high|39.311／34.304|81.084／71.908|80.732／73.021|78.692／71.713|
|practical_mid|37.955／33.521|78.360／70.213|78.079／71.188|76.243／69.771|
|practical_low_suburban|36.759／32.054|75.573／67.742|75.379／68.883|73.595／67.542|
|practical_high_urban|38.481／34.000|77.423／69.108|77.164／70.129|75.322／68.888|
|practical_mid_urban|33.447／29.496|62.450／56.458|62.333／57.246|61.229／56.458|
|practical_low_urban|29.185／25.750|50.887／46.379|50.762／47.025|50.276／46.675|

每行原生 student train6078780NN、student eval918004NN、teacher train0NN、teacher eval963900NN，native合计7960684NN；L计5114880NN。

|行|新/复用|训练与原生验证 wall（秒）|七视图预测 NN|预测 wall（秒）|
|---|---|---:|---:|---:|
|NOAS|历史复用|2146.190|1176002|48.089|
|AS_RD|历史复用|2860.909|1176002|49.650|
|AS_CE|新增|2896.522|1176002|48.625|
|AS_DET|新增|2907.271|1176002|47.300|

新增两行60 model-epochs／13320更新，native15921368NN，外部source25200NN、正式2352004NN、shared smoke1538NN，总新18300110NN。复用两行历史native15921368NN、source25200NN、正式2352004NN，历史合计18298572NN，原发布shared smoke不擅自分摊。分析r01/r02失败及r03恢复均0模型NN，r03完整pipeline wall60.704秒。

独立source四条件全部PASS：fullV DET−NOAS−.007407pp、DET−RD+.096296pp、DET−CE+.022222pp；V1800两draw mid/low均值 DET66.958333%、NOAS39.583333%、RD65.944444%、CE66.194444%。source summary的mid_low_*_pp历史字段实际存绝对准确率百分数，不能当配对增益。正式方向相反，不能以source PASS覆盖正式阶段失败；也不能将相对NoAs的完整As收益归因于dropout绕过。

AS_RD同seed clean+.093452pp、AS_CE−.033333pp，二者在本seed满足clean阶段约束；原完整As三seed clean均值−.3210pp的历史结论不因此改写。AS_DET本seed相对原As各正式视图均下降，未验证新增收益；所有参数保持冻结，不据正式分数调层/概率/预算或挑种子。E50原完整As预算实验已在读取本次正式汇总前由既有三seed source依据固定，与本次新机制结果分开。
