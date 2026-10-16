# 真实同 TX 观测配对：源域完整结果

固定源规则 **FAIL**，一个model seed／一个held源RX1，0target。C0仅复用原结果，两个新模型从零E20。

|scene|C0 accuracy|SAME_RX|CROSS_RX|CROSS−C0 pp|CROSS−SAME pp|
|---|---:|---:|---:|---:|---:|
|clean|83.4444%|84.3148%|83.8704%|+0.4259|-0.4444|
|practical_high|38.4259%|30.0926%|25.2963%|-13.1296|-4.7963|
|practical_mid|36.1667%|28.1667%|24.2037%|-11.9630|-3.9630|
|practical_low_suburban|34.0370%|26.9444%|24.0556%|-9.9815|-2.8889|
|practical_high_urban|37.0000%|28.5741%|24.4630%|-12.5370|-4.1111|
|practical_mid_urban|30.1667%|25.6852%|22.6852%|-7.4815|-3.0000|
|practical_low_urban|25.6852%|22.5741%|20.7037%|-4.9815|-1.8704|

预登记两个主比较／F1／seenV／健康检查逐项：{'vs_C0': False, 'vs_same': False, 'F1': False, 'seen': True, 'complete_health': True}；数值 {'C0': {'accuracy_gain_pp': 0.4259259259259296, 'macroF1_gain_pp': 0.43136369271473685, 'seen_drop_pp': -0.04629629629629983}, 'SAME_RX': {'accuracy_gain_pp': -0.4444444444444473, 'macroF1_gain_pp': -0.456890945690569, 'seen_drop_pp': -0.037037037037035425}}。不以SAME_RX事后替代主候选；失败不调比例/换RX/seed救援。

完整40新模型轮／8880更新／全部日志／F clip与SupCon，原L/U共5994批与C0逐条一致；每行568320真实partner均同TX/day、不同物理包，全部5040合法L ID和所有允许RX配对cell覆盖，原U truth始终隐藏。

新增网络包7158036；两行各5040 bank IQ读取、568320 secondary物理曝光和ordinary生成；bank读取及partner视图构造墙钟见measured_costs.csv，后者不含骨干前向。矩阵墙钟1253.322s。C0既有E20／4440更新和网络费用独立列出，无父权重继承。预算用尽不等于收敛。

两新行都保留.5原primary监督、.5partner监督／四L前向、原F+CROSS_RX／默认头联合训练／原U与ownEMA。新算子只选真实观测，旧ordinary原处理是共有条件。CROSS−C0包含独立样本与独立ordinary多样性的共同变化，CROSS−SAME才检验跨RX的增量。不能把它写成数字IQ变换或模拟替换RX硬件。

首折源资格即使PASS，也不是正式target、跨模型seed、独立Ag+As组合或lowurban>60%的证据。只支持另登记源确认。若FAIL保持结束，不由任何目标分数推翻。没有训练As、额外对齐或新部署模块/分类头，正式IQ与practical配置保持。全部TX/day/state/quality负结果、repair/break、固定C0 clean正确集合保持率见CSV。


实际状态 **VERIFIED／ANALYZED，科学判定 FAIL_SOURCE_RX1_ONLY**。两 worker 与 dispatcher 自然结束，无技术 failure／重试／热改。训练代码9b2eebcfa809c8d5cfee6523d35b84aa10ded1a4，离线结果读取代码db56f4c781e7706d3dcc894e6d219df7dc6071e7；两者分开记录，未改运行 release。

默认头 clean：C0 83.444444%、SAME_RX84.314815%、CROSS_RX83.870370%。CROSS 对 C0 只增加23/5400个正确包（修复83／新增错误60），+0.425926 pp，未达+1 pp；对 SAME 减少24个正确包，-0.444444 pp，未达+0.5 pp，宏F1还低0.456891 pp。seen V和完整更新／覆盖健康通过，不能覆盖主性能失败。SAME 相对 C0 的+0.870370 pp只作控制读数，不改成获胜候选。

同时，CROSS 对 C0 的六个 practical 全部下降：high -13.129630、mid -11.962963、low_suburban -9.981481、high_urban -12.537037、mid_urban -7.481481、low_urban -4.981481 pp。SAME 也全部低于 C0；没有 A_s 训练，不能由 clean 小幅变化推导 practical 改善。C0 clean 正确集合4506包在low urban的保留率分别为25.943187%、23.280071%、21.193964%（C0/SAME/CROSS），集合固定，不随候选重选。

保持 F＋CROSS_RX SupCon 三路 SIG_TIME／CENTER_25K／CENTER_UNITS，fs25MHz、IQ256、equalized1、native RMS、NoExplicitCFO、原joint160/default cosine30联合训练。温度0.2、最大0.1、warm5/ramp10、time_down/t2.pool上的原L同TX不同物理RX正对不改，仅复用第一对原RDrop前向。全部默认头／SIG三路／Sinc参数正常联合更新，每步F physicalclip执行，206初态与C0逐键匹配；无额外部署参数、诊断头、embedding约束或跨真实观测KL。

本折fit RX3/4/6/8：L5040、隐藏U45360、seenV21600；held RX1的L/U不训练，heldV5400只FINAL20读取。两个新模型均scratch E20＝13L＋7U、4440更新，0父权重/EMA/optimizer/统计继承。原U/ownEMA/旧门控/cos和strong_noise_std=0.015保持，U TX真值从不解封。不是全源L6300，也不是内部5580/720合同。C0只复用相同合同下已冻结预测／日志／trace，不加载其checkpoint。

两新模型0.5 RD(primary old ordinary)＋0.5 RD(real partner old ordinary)，每RD内同一已生成IQ两次dropout，LS0.01／RDrop alpha1；四L网络前向匹配C0。第二观测只来自当前filtered原train_loader.dataset的L，同真TX、同day、不同物理包；SAME保持RX，CROSS在其他三个fit RX中均匀选择。bank在自身scratchCP smoke通过后读取，保留原intact IQ和native RMS；raw RX/day domain沿原domain_label_map正确映射到compact模型域，第二观测使用自身域。非数字IQ扰动、非硬件替换，也没有增加可见标签ID数量。

实际4440步trace独立复核全部568320次partner：真TX/day相同、不同physical packet、全部5040合法L ID覆盖，SAME72个／CROSS216个允许配对cell完整。原primary L/U全5994批在三个模型逐条相同。secondary采样和ordinary流独立，保护原loader/dropout/U随机流；CROSS−C0包括独立观测与独立ordinary的共同变化，CROSS−SAME才隔离RX组织增量。两个比较均按预登记判定，不调比例、不换RX/seed、不延长E20救援。

完整20轮／4440步native CSV／stdout／loss／SupCon和F clip解析通过；七视图每模型5400包、总113400个默认头预测独立复算，源物理ID／label和六场景physics元数据与C0逐包相同。读取分析0新网络前向／IQ／checkpoint。实际leo_practical路径为release的experiments/adv3b02_xuc/code，fs25MHz/residual/post_sync、EQoff G1、reference30、quality_adaptation／phase_tracking／IQcomp／AGC／reflect均保持；未修改正式测试IQ、信道、SNR或困难样本。此矩阵source-only，0target，非正式truth-last或独立盲测。

单个模型seed2026092701、一个源折，两个新模型不是两个seed。原F+CROSS_RX底座保留，关闭的是新增跨RX观测配对假设，不因负结果改变用户指定基线。真正Ag、合格As、B00/B10/B01/B11及交互、跨seed正式复现与lowurban>60%仍未证明。全部负TX/day/state/quality及间隔保留，辅助损失下降不称识别改善。

实测费用：

|arm|新增/复用|模型轮|更新|网络包|训练/冻结源V 秒|六practical 秒|bank读取 秒|partner构造 秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|C0|复用结果|20|4440|3579018|1123.572077|64.008927|0.000000|0.000000|
|SAME_RX|新增|20|4440|3579018|1182.903448|64.523674|0.751735|39.204501|
|CROSS_RX|新增|20|4440|3579018|1143.488998|63.720206|0.758507|37.759797|

两个新增模型共40模型轮／8880更新／7158036网络包，矩阵墙钟1253.321882 s；C0既有20轮／4440步／3579018网络包单列。每新行5040次bank原IQ读取、568320次partner物理曝光及额外ordinary生成；可见原L的独立ID始终5040，不称标签翻倍。partner构造计采样、旧ordinary、设备传输、波形统计／日志／cache，不含骨干前向。每行网络分解为L2273280／studentU396368／student source eval475202（含2包ownCP smoke）／EMA396368／heldclean5400／sourcepractical32400。合成本地验证1024网络包／2更新、1152合成partner观测及14主负例＋domain映射两负例单列；离线读取器另13元数据负例、0网络/IQ/CP，合成读数不是实际性能。CPU离线全分析墙钟未统一计时。

唯一P0/P1审查发现raw domain错误，发布前修复并由原审查者定向读回PASS；没有第二轮全审查。一次N607普通账户preflight、一次archive本远比较、remotecompile、worker首步真实ownCP无query smoke和PID/CWD/argv/GPU/log增长均VERIFIED，原始证据保留。预算耗尽不等于收敛。

以下是low urban的配对分层修复/新错，全集见CSV：

|比较|分层|值|包数|修复|新错|准确率Δ pp|
|---|---|---|---:|---:|---:|---:|
|CROSS_RX-C0|receiver_lock_status|degraded|965|17|28|-1.139896|
|CROSS_RX-C0|receiver_lock_status|locked|2081|16|285|-12.926478|
|CROSS_RX-C0|receiver_lock_status|unlocked|2354|77|66|+0.467290|
|CROSS_RX-C0|state_end|BLOCKED|3016|89|85|+0.132626|
|CROSS_RX-C0|state_end|LOS|1674|10|247|-14.157706|
|CROSS_RX-C0|state_end|PARTIAL|710|11|47|-5.070423|
|CROSS_RX-SAME_RX|receiver_lock_status|degraded|965|23|27|-0.414508|
|CROSS_RX-SAME_RX|receiver_lock_status|locked|2081|28|131|-4.949543|
|CROSS_RX-SAME_RX|receiver_lock_status|unlocked|2354|69|63|+0.254885|
|CROSS_RX-SAME_RX|state_end|BLOCKED|3016|84|81|+0.099469|
|CROSS_RX-SAME_RX|state_end|LOS|1674|19|103|-5.017921|
|CROSS_RX-SAME_RX|state_end|PARTIAL|710|17|37|-2.816901|

精确训练command/environment/CWD/paths/GPU/stop见experiment.json和preregistration.md。当前两个新run-owned模型均已结束，无自动扩展／target访问。

离线复算（0IQ/CP/network）：

`F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_f_realpair20.harvest --output E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final --state E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/terminal_readback.json --reference-root E:/type10-7/local_artifacts/cvs_FcrossRX_practical20_20261016/final --cached`

实际表图（三图已逐张检查）：

- [complete_training_curves.png](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/complete_training_curves.png)
- [source_scene_results.png](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/source_scene_results.png)
- [low_urban_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/low_urban_strata.png)
- [scene_results.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/scene_results.csv)
- [source_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/source_strata.csv)
- [paired_strata.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/paired_strata.csv)
- [training_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/training_curves.csv)
- [fusion_gain_curves.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/fusion_gain_curves.csv)
- [measured_costs.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_realpair20_20261016/final/measured_costs.csv)
