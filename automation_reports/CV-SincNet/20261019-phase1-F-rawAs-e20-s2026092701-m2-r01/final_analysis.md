# 纯 F rawAs scratch E20：完整七视图与源拟合分析

状态：**ANALYZED / VERIFIED；固定 whole-raw 配方源规则 FAIL，关闭，不追加 E50**。目标未达成。模型 seed 2026092701，n=1；每视图固定 168000 个物理样本，原 received IQ/单包默认头协议不变。B0/B1 复用原完整预测和评分，未重训或重测。

## 结论与事前决定

唯一假设是第一条 L 监督槽位使用 raw，是否改善相对 ordinary+As 的受损观测学习。RAW_AS 正式 clean 80.740%、low urban 52.156%，相对 B1 分别 -0.471、-0.480 个百分点；不支持该假设。RAW_NOAS 的 clean 只有 80.577%，相对原 B0 下降 3.587 个百分点。RAW_AS 相对 RAW_NOAS 的 clean 虽有 0.163 个百分点正差，不能取代对原 B0 的 -3.424 个百分点净作用，也不是可重复双提升。

在新增两行独立评分之前，源决定于 2026-10-19T06:45:29.091Z 固定：mid/low 两场景、两 draw 平均增量 0.222 个百分点，小于预设 +0.5；full V clean 对 B0 -0.089、对 RAW_NOAS -0.307 个百分点，后者超过 −0.2 容差。两项失败。正式结果不回流修改源规则，不通过比例、seed 或日程救援。预备的 E20+30 状态方案未执行。

## 实际方法与匹配边界

固定 F 为三路 SIG_TIME＋统一 Sinc，900722 参数，实际 joint160，NoExplicitCFO，内部融合与原 cosine 默认头正常联合训练；无 CROSS_RX、RDrop、外部 CSIL 损失、新头或部署分支。均同 seed scratch E20＝13 L＋7 U、4440 更新，原 AdamW/cosine20、smoothing0.01、T1。数据 L6300/U56700（TX 始终隐藏）/V27000，RX1/3/4/6/8、day1/2/3、split392005 不变。

B0 为原普通增强 NoAs 单 L 前向；B1 为 0.5 ordinary CE＋0.5 fullPractical(raw) CE；RAW_NOAS 为 0.5 raw CE＋0.5 raw CE；RAW_AS 为 0.5 raw CE＋0.5 fullPractical(raw) CE。新增两行第一槽先完整执行原 ordinary 作为 shadow，丢弃结果，再返回调用前 raw；shadow 不进入 NN/loss。U 的原 weak/strong、batch_neighbor 资格与 ownEMA 不变。所有合法 L As 均真实标签 CE，无置信度筛选。As 从 raw 生成，无 ordinary→As 叠加或 As 后归一化；原传播/遮挡/噪声/接收机/同步补偿依赖与全部15个RX/day硬件session保留，不按TX绑定。

四行初始206个state条目逐张量完全相同，完整4440 L批和1554 U批的实际ID序列均相同；B1和RAW_AS完整物理信道汇总相同。这些是独立读回结果；不由seed相同推定全部训练RNG轨迹完全相同。旧B0/B1 config保留了历史RDrop/E50预算说明字符串；实际native mode为ORIGINAL、累计20，本报告以实际运行而非旧描述字段为准。

## 源端资格与拟合

full V clean 是27000个源验证样本；三个信道指标是固定1800物理ID×2个新draw，各3600次观测，不是独立模型seed。

|行|full V clean %|最差源 RX %|low suburban %|mid urban %|low urban %|
|---|---|---|---|---|---|
|B0|98.056|94.815|62.694|52.778|42.750|
|B1|97.763|94.259|89.667|75.639|62.139|
|RAW_NOAS|98.274|95.352|59.500|49.056|40.083|
|RAW_AS|97.967|94.574|90.028|75.389|62.833|

另有0更新固定180 L面板诊断，详见 [源拟合报告](../20261019-diagnostic-F-rawAs-sourceFit-e20-s2026092701-m2-r01/final_analysis.md)。四模型 raw180均180/180正确，不能推广为全部L拟合100%。low urban重建训练流仅69个L，B1为45/69（65.217%、标准CE0.829691），RAW_AS为47/69（68.116%、CE0.810532）；同一69 L的两个新draw，B1为89/138（64.493%、CE0.862350），RAW_AS为90/138（65.217%、CE0.850040）。源V新low观测分别62.139%和62.833%。受损观测在训练期流中已存在明显误差；whole-raw改动仅有微小源改善，不能称为物理上限或训练集整体准确率。

重建使用原命名随机流、Config/session/seed及真实step/slot，并核对RAW_AS当步真实消费ID与As128。旧/新两行均没有历史As IQ字节留存；RAW_AS也未持久化逐包scene_index，只能按实际调用流重建。same-L新draw与旧观测的state/lock难度组成可能不同，特别mid urban更难，不能把准确率差直接叫过拟合。

## 七种正式视图

准确率（%）：

|视图|B0|B1|RAW_NOAS|RAW_AS|
|---|---|---|---|---|
|clean|84.164|81.211|80.577|80.740|
|practical_high|51.095|80.755|47.907|79.235|
|practical_mid|49.149|78.355|45.749|77.093|
|practical_low_suburban|47.230|75.923|43.665|74.871|
|practical_high_urban|48.966|77.277|45.723|75.939|
|practical_mid_urban|40.274|63.473|36.715|62.701|
|practical_low_urban|33.649|52.636|30.337|52.156|

宏 F1（%）：

|视图|B0|B1|RAW_NOAS|RAW_AS|
|---|---|---|---|---|
|clean|83.847|80.612|79.996|80.092|
|practical_high|49.431|80.328|45.255|78.769|
|practical_mid|47.584|77.994|43.489|76.707|
|practical_low_suburban|45.804|75.574|41.829|74.517|
|practical_high_urban|47.211|76.902|43.227|75.557|
|practical_mid_urban|38.675|63.164|35.171|62.479|
|practical_low_urban|32.215|52.145|29.350|51.790|

最差目标 RX 准确率（%）：

|视图|B0|B1|RAW_NOAS|RAW_AS|
|---|---|---|---|---|
|clean|74.313|67.750|70.821|69.167|
|practical_high|40.433|74.033|43.158|71.242|
|practical_mid|39.642|72.471|41.500|70.142|
|practical_low_suburban|38.862|70.600|39.896|68.504|
|practical_high_urban|39.142|70.767|41.513|68.571|
|practical_mid_urban|33.783|58.646|33.558|57.321|
|practical_low_urban|29.346|48.817|28.108|47.833|

RAW_AS三个suburban相对自身clean的差距，high 1.505、mid 3.647、low 5.870 个百分点；必须同时看上述绝对准确率，不能通过降低clean宣称差距改善。

## 配对错误与种子边界

下表均按相同168000物理ID配对。n=1模型seed，配对均值就是该差值，样本标准差不可估计（null）；正差seed数只为0/1或1/1，不声称稳定。

|比较|视图|修正错误|新增错误|准确率差/配对均值 pp|正差seed数|
|---|---|---|---|---|---|
|RAW_AS_minus_B0|clean|3579|9331|-3.424|0/1|
|RAW_AS_minus_B0|practical_high|53883|6608|28.140|1/1|
|RAW_AS_minus_B0|practical_mid|54081|7135|27.944|1/1|
|RAW_AS_minus_B0|practical_low_suburban|54263|7827|27.640|1/1|
|RAW_AS_minus_B0|practical_high_urban|52584|7269|26.973|1/1|
|RAW_AS_minus_B0|practical_mid_urban|47906|10230|22.426|1/1|
|RAW_AS_minus_B0|practical_low_urban|43885|12794|18.507|1/1|
|RAW_AS_minus_B1|clean|2120|2911|-0.471|0/1|
|RAW_AS_minus_B1|practical_high|2128|4682|-1.520|0/1|
|RAW_AS_minus_B1|practical_mid|2802|4922|-1.262|0/1|
|RAW_AS_minus_B1|practical_low_suburban|3348|5115|-1.052|0/1|
|RAW_AS_minus_B1|practical_high_urban|2750|4998|-1.338|0/1|
|RAW_AS_minus_B1|practical_mid_urban|5183|6480|-0.772|0/1|
|RAW_AS_minus_B1|practical_low_urban|6703|7509|-0.480|0/1|
|RAW_AS_minus_RAW_NOAS|clean|6064|5790|0.163|1/1|
|RAW_AS_minus_RAW_NOAS|practical_high|57634|5003|31.328|1/1|
|RAW_AS_minus_RAW_NOAS|practical_mid|58163|5504|31.345|1/1|
|RAW_AS_minus_RAW_NOAS|practical_low_suburban|58810|6385|31.205|1/1|
|RAW_AS_minus_RAW_NOAS|practical_high_urban|56593|5830|30.216|1/1|
|RAW_AS_minus_RAW_NOAS|practical_mid_urban|52758|9103|25.985|1/1|
|RAW_AS_minus_RAW_NOAS|practical_low_urban|48716|12060|21.819|1/1|
|RAW_NOAS_minus_B0|clean|3354|9380|-3.587|0/1|
|RAW_NOAS_minus_B0|practical_high|14057|19413|-3.188|0/1|
|RAW_NOAS_minus_B0|practical_mid|13999|19712|-3.401|0/1|
|RAW_NOAS_minus_B0|practical_low_suburban|13923|19912|-3.565|0/1|
|RAW_NOAS_minus_B0|practical_high_urban|13816|19264|-3.243|0/1|
|RAW_NOAS_minus_B0|practical_mid_urban|13491|19470|-3.559|0/1|
|RAW_NOAS_minus_B0|practical_low_urban|13088|18653|-3.313|0/1|

另固定原B0 clean正确的141396个样本，保持同一集合：

|行|clean|practical_high|practical_mid|practical_low_suburban|practical_high_urban|practical_mid_urban|practical_low_urban|
|---|---|---|---|---|---|---|---|
|B0|100.000|56.999|54.744|52.579|54.569|44.708|37.159|
|B1|94.031|90.574|87.575|84.613|86.518|70.286|57.804|
|RAW_NOAS|93.366|53.676|51.367|49.019|51.188|40.945|33.668|
|RAW_AS|93.401|89.204|86.426|83.658|85.249|69.492|57.242|

此表是条件正确保留率（%），不是全测试集准确率，也不用于候选选择。

## 完整日志、骨干与成本

完整读取每行4440 mechanism/native/unit/shadow记录、20 native/companion epoch、两份各20行CSV与全L/U消费记录；不是采样日志。每步0.5/0.5 CE公式最大误差1.20e−7，native实际入口raw/extra/L对应误差为0，RAW_AS有标签阶段总loss最大误差2.99e−8。两行4440步第一输入raw逐张量验证、每步额外一次L前向与计数一致；ordinary返回raw MSE最大0、shadow共568320包、其中247341包发生了被丢弃的实际扰动。

每行1554隐藏U步骤、396900 U学生包、396900 ownEMA eval包与4440 EMA更新已核实；没有“所有教师为0”的成本误述，仅外部教师为0。joint形状[2,160]；Sinc两坐标、三路SIG_TIME/时频/PA/融合/原默认头关键参数均实际变化，未冻结骨干另训头。真实Sinc学习率/decay物理单位检查通过，两行无技术错误行。终轮raw CE为RAW_NOAS 0.085、RAW_AS 0.122；RAW_AS的As CE仍 0.686。native train_tx_acc只代表第一监督槽，不冒充As准确率。

|行|L学生NN包|U学生NN包|ownEMA eval包|As生成包|ordinary shadow包|外部教师包|训练墙钟秒|
|---|---|---|---|---|---|---|---|
|B0|568320|396900|396900|0|0|0|962.933|
|B1|1136640|396900|396900|568320|0|568320|2067.391|
|RAW_NOAS|1136640|396900|396900|0|568320|0|1047.833|
|RAW_AS|1136640|396900|396900|568320|568320|0|2012.295|

B1历史外部CSIL仅eval shadow，不进训练loss/RNG，但仍单列实际成本；本轮未重算B0/B1。每新增行另有594004源训练eval包、12600源资格包、1176002正式预测包（168000×7＋首smoke2）。本次完整日志/冻结预测重计0新模型前向、0新预测和0新评分。新增源拟合诊断另为2881包、0更新；所有成本分别记录，未宣称相同墙钟。

## 执行、产物与边界

训练代码/config commit `3d4e188890dce6df2c20c9df5576450d06a2eac4`；终轮分析工具提交 `8419091824d61ced995f3414f2dad7a64d0a6377`；源拟合诊断代码 `8c062e0a71a001cb744200bbbf17fbe0c643a226`。普通N607原Python/Torch环境，CPU2、FP32/TF32off，训练GPU4/5，所有owner每GPU最多2进程且空闲至少12000MB。训练owner codex/root/F-rawAs20，dispatcher2139499，两行训练PID2139570/2139653，CWD PROJECT/releases/cvs_F_rawAs_E20_20261019_r01。原命令及逐行配置见 experiment.json，输出 PROJECT/runs/20261019-phase1-F-rawAs-e20-s2026092701-m2-r01。单归档一次传输比较、远端11文件编译、实际scratch checkpoint无query首smoke、一次P0/P1审查及启动后PID/CWD/argv/GPU/日志增长均VERIFIED。未停止/重启/热改健康任务。

所有七视图prediction先固定，source_matrix_decision在新增评分前固定，独立truth-last完成。既有benchmark已暴露，不称新独立盲测。无E50、无扩seed，low urban>60%与clean可重复正收益均未实现。

图表已生成、取回并逐张查看：

- [七视图与完整训练曲线](E:/type10-7/exports/cvs_pureF_rawAs_20261019/seven_views_and_full_training.png)；[PDF](E:/type10-7/exports/cvs_pureF_rawAs_20261019/seven_views_and_full_training.pdf)。
- [配对增量与固定clean正确集合保留](E:/type10-7/exports/cvs_pureF_rawAs_20261019/paired_errors_and_clean_retention.png)。
- [中文源拟合准确率](E:/type10-7/exports/cvs_pureF_rawAs_20261019/source_fit_accuracy.png)；[中文标准CE](E:/type10-7/exports/cvs_pureF_rawAs_20261019/source_fit_CE.png)。

完整私有运行读回、源判定、逐步日志重计、全部混淆矩阵及RX/day/TX分层保留于 local_artifacts/F_rawAs_E20_20261019/complete_analysis.json；原始weights/logs/predictions保持远端原不可变root，不纳入Git。
