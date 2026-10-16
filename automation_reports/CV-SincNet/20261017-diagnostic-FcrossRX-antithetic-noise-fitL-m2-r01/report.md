# F＋跨 RX：正负噪声配对 CE 方差完整源诊断

**VERIFIED／ANALYZED，固定方差理由 PASS；识别性能提升尚未检验。** 两份既有 E20 raw8L 计算匹配基线均保持全参数、buffer 和 .grad 状态，0 新训练更新、0 额外分类头、0 target。基线为 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、默认 cosine30。

|冻结模型 seed|独立双噪声参考方差和|正负配对方差和|配对／独立比|方差下降|负协方差批|原门槛|
|---|---:|---:|---:|---:|---:|---|
|2026092701|50.806101|35.591764|0.700541|29.946%|40/40|PASS|
|2026092702|42.224399|28.801128|0.682097|31.790%|40/40|PASS|

门槛在真实源 IQ 执行前固定：两 seed 都要求 whole sum(V_ant)/sum(V_ind)≤0.80，且至少60%的40批噪声协方差为负。两 seed 都通过。某个批次的比值可高于.80，原门槛按全40批之和判断，未事后改为逐批要求或筛选批次。

独立 reader 完整读取 480 条全梯度分组记录的16×16原始 Gram、161280条默认头判决、两个模型相同的全部噪声／物理记录；从 Gram 独立计算组内无偏 Vplus、Vminus、C 和 V_ant，确认五组之和等于 whole、全部40批与5040个唯一物理 L 完整、末批48保留。没有用抽样代替完整日志或判决分析。各模型均80642 NN，含真实 checkpoint 2包 zero IQ no-query smoke；总161284 NN。

参数组仅用于完整描述，未按层挑选成功项：

|参数组|seed01 比值|seed02 比值|
|---|---:|---:|
|default_head|0.602070|0.584614|
|SIG_TIME|0.712532|0.687235|
|Sinc_Hz|0.754911|0.783635|
|time_path|0.736698|0.700507|
|remaining|0.701346|0.689678|

独立双噪声参考 V_ind=(Vplus+Vminus)/4，来自同一组边际梯度样本的无偏估计；未额外执行独立噪声双前向。配对 V_ant=Var((g+ +g−)/2)，且 V_ant=V_ind+C/2。条件是每个固定物理信道、固定 batch、8个独立原始高斯噪声；两个模型不是两个新的物理数据集。该比值只对应冻结 eval CE 导数，不是实际 AdamW 步长或完整 RDrop／SupCon 训练方差。

每个符号从原 residual 的 primitive noise 入口经过原 RX IQ、补偿、G 和各自 AGC，signal、geometry、state、quality、CFO、phase、kernel、hardware不变；逐观察检查相同。两符号各自的高斯边际分布相同，没有平均IQ、衰减噪声、提高SNR或改正式测试。原引擎 Config 与 source baseline 的全部实际值相等，完整 __file__ 和配置保存在 observation_audit.json：adv3b02_xuc 后端，25MHz／IQ256／fc2462MHz／residual／post_sync／EQoff G=1／reference30／reflect，接收质量、相位跟踪和IQ补偿按原配置。白输入功率代理和 signal_to_added_noise_db 都不冒称未知总SNR，滤波数值延迟和合成反射边界保持。

fit RX3/4/6/8 的 L5040（72 TX×RX×day单元各70包），两条历史 raw8L 模型都为合法匹配 source-only scratch final20、4440更新、同物理L/U/V角色、无未知祖先／目标选模／头拟合。完整 L6300、源V和旧父特征由既有缓存反序列化，只有fitL5040进入观察器；heldRX1、U/V不前向，U真值未读。RX/day/eq 硬件 session 不由 TX定义，六场景nominal均匀先验保留，0困难样本丢弃。原父训练来源成本分别E20／4440／5823318 NN，总40model-E／8880／11646636 NN。

以下只是已见fitL上的冻结默认头描述性读出，混合六种practical与16个噪声视图，不能称为low urban成绩、held RX泛化、训练收益或正式目标成绩。没有以某个符号的准确率晋级，也不平均两视图的logits用于部署。完整每draw／sign accuracy、macro-F1、confusion及margin/30留在 analysis_summary.json 和 predictions.jsonl.gz；全部scene/state/lock/RX/day在source_strata.json。

|状态或锁定|重复视图判决数／seed|seed01准确率|seed02准确率|
|---|---:|---:|---:|
|LOS|55472|68.279%|71.613%|
|PARTIAL|9728|56.949%|55.582%|
|BLOCKED|15440|24.424%|24.106%|
|locked|62368|67.498%|70.259%|
|degraded|8688|33.023%|33.092%|
|unlocked|9584|23.174%|22.538%|

这些分层是原观测条件下的关联，未执行新模型更新，不声称缓解了 BLOCKED／unlocked，也不能以 LOS 较高代替困难 urban 改善。

新训练成本0 model-E／0更新／0EMA／0头拟合；本地累计32个真实架构合成前向包、0更新，所有分析和绘图0NN。两个worker主体分别143.595／141.269秒，并行进程启动至最终结果约148.125秒（约1秒精度）。唯一dispatcher和两个worker自然退出，0真实执行失败／重启／热改，未干预其他owner。

验证通过34个局部配置／权限／方差负例、真实网络合成梯度和原视图一致性；合成检查覆盖EQ开启的原噪声着色。一次独立P0/P1发现原tuple与JSON list直接比较会使新run误停；已复现，双方统一规范化，真实reference_snr_db变化仍拒绝，唯一该项定向复审PASS。另有reader合成fixture最初仅4个噪声draw而固定协议为8，在真实数据访问前修正fixture，0NN；未改科学抽样预算或门槛。N607一次普通账户预检、一个release归档比较、一次remote compile、两真实CP smoke、PID/CWD/argv/GPU及日志增长、完整自然终态均有独立证据。

实际执行 commit：1bc071c9d3148de703f2ffcd6322feaa3aa33db6；取回／绘图实现：f6e5c0c172b7271fd1db8de4fc930236c5e15717，两者均已推送并独立核对远端OID。完整逐row矩阵、精确命令、环境/CWD、GPU与stop见experiment.json，启动前设计见preregistration.md。

图已实际生成并查看：[noise_gradient_variance](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair_probe_20261017/final/figures/noise_gradient_variance.png)、[source_noise_readout](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair_probe_20261017/final/figures/source_noise_readout.png)、[source_noise_strata](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair_probe_20261017/final/figures/source_noise_strata.png)；同目录各有PDF。原始日志、预测、物理观察、完整Gram留在E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair_probe_20261017/final，不提交IQ、权重或原始逐包身份元组。

下一项唯一训练假设：保持同样的As边际分布和总监督权重，正负噪声配对带来的CE估计方差下降能否改善urban身份识别，同时保护clean。最小计划是在同一seed01匹配源折从零独立E20的三行：raw重复控制、固定同信道的独立双噪声As、正负双噪声As；全部共同原三对base保留，新增两对各按原同IQ RDrop计算，再以均值乘.25，10次L前向／包全部匹配。KL仅在同一已生成IQ的dropout前向内，噪声不同视图之间不加KL。原ordinary-only SupCon、U规则和默认头联合训练保持。不得用旧raw8L绝对数充当10L匹配控制。该训练计划尚未实现、登记或启动，不把它写成已完成实验。

候选训练的拟固定证据条件（须在训练登记前冻结）：正负As相对raw的clean准确率与宏F1不降、urban均值至少+3pp且low至少+5pp；相对独立噪声As的urban均值和low各至少+1pp，clean及宏F1下降≤.5pp；全部practical及seenV下降≤.5pp，预算／覆盖健康通过。未通过则关闭这一训练采样理由，不调系数、SNR、seed、RX、噪声抽样数或追加epoch救援。通过单seed仍须独立第二seed确认，不能宣称目标完成。

当前没有合格的新Ag，没有Ag/As四格与交互项，没有新模型的正式clean＋六practical truth-last，也没有low urban>60%或suburban接近clean的联合证明。固定相干插值及旧fixed additive As的失败仍保留，不把不同实验增益相加。当前总目标ACTIVE。
