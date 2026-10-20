# R58：压缩图可学习组统计校正正式结果

VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。calibrated 276453参数，七景较父提高0/下降7，lowurban 67.383333%（较父-0.126786pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.4561/121.1858ms，G+F B1父/本版56.2264/132.8959ms；identity_control 276453参数，七景较父提高0/下降7，lowurban 66.771429%（较父-0.738690pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.1120/119.8319ms，G+F B1父/本版55.7322/130.8974ms。同图calibrated相对identity_control七景提高5/下降2，lowurban差+0.611905pp；R55仅旧归一化参考、非完整参数/计算匹配控制。新2688统计参数仅20轮/4940次；原参数80+20=100轮，所有参数联合优化，成本分别报告。原指定350521直接父保持，正负结果保留；不把容差、减参、源拟合或解析关联当作整体提升或逐细节TX真实性。原完整目标ACTIVE。

**VERIFIED / ANALYZED**。两固定行完整 E20/4940、source1800×7、正式168000×7；逐行七预测固定后独立 truth-last，所属进程实际退出。全部正负结果保留，不反馈调参或选择性重跑。

性能优先。直接初始化和冻结 IQ 教师保持人选 R46 prune_refine350521；复用 R55 source-only full-fold 统计，无 R55/R57 训练后学生继承。既有 Conv/U-Net/3prior6CG 内168个GN加入2688个可学习组统计参数；完整276453、context186959，所有原参数与新参数联合训练。两行图、参数、计算、预算相同，仅 calibrated 为源拟合初始化，identity_control 为恒等初始化。原参数继承80+20=100轮，新统计参数只训练20轮/4940次；不能把新参数也记为100轮。

源训练使用6300L_s+56700U_s的模拟受损/clean配对，E20每行L126000/U1134000，全局256/micro256，原IQ+.05谱误差+1→.1三阶段IQ教师，0TX标签/F权重/F输出/V/target训练。属于TX无标签源域自监督，使用clean配对；原冻结F50默认头仅评测同一恢复IQ及其FFT/PA/时域/统计分支。方案及核心原型在R57正式分数读取前固定。

方法 commit `3e1c14a733b2278c659c653c75a98657663cd8d4`。见[登记](experiment.json)、[完整结果](results.json)和[逐组CSV](row_metrics.csv)。单seed、已暴露且人选父的工程基准，不称新的独立盲测。

|行|完整参数|较350521减参|R17每景1pp容差|父每景1pp容差|low urban≥70%|其余五景距rawclean≤5pp|
|---|---:|---:|---|---|---|---|
|calibrated|276453|21.130831%|True|False|False|False|
|identity_control|276453|21.130831%|True|True|False|False|

容差、减参和局部拟合均不等于性能优化。raw clean同F50准确率84.6761904762%，其他五个受损场景门槛79.6761904762%。

|行|场景|准确率（%）|MacroF1（%）|较指定父（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|
|calibrated|clean|84.451786|84.148984|-0.004762|-0.135119|-377|
|calibrated|practical_high|81.282143|80.681225|-1.170238|-0.036905|68123|
|calibrated|practical_mid|80.321429|79.785090|-0.900595|+0.743452|67880|
|calibrated|practical_low_suburban|79.026786|78.559076|-0.593452|+1.030357|67195|
|calibrated|practical_high_urban|80.207738|79.637934|-1.179167|-0.145833|68056|
|calibrated|practical_mid_urban|73.819643|73.423322|-0.627381|+0.134524|66085|
|calibrated|practical_low_urban|67.383333|67.188545|-0.126786|+0.166667|62301|
|identity_control|clean|84.367262|84.057507|-0.089286|-0.219643|-519|
|identity_control|practical_high|81.537500|81.009360|-0.914881|+0.218452|68552|
|identity_control|practical_mid|80.310714|79.837029|-0.911310|+0.732738|67862|
|identity_control|practical_low_suburban|78.694643|78.326247|-0.925595|+0.698214|66637|
|identity_control|practical_high_urban|80.452976|79.968345|-0.933929|+0.099405|68468|
|identity_control|practical_mid_urban|73.651786|73.525793|-0.795238|-0.033333|65803|
|identity_control|practical_low_urban|66.771429|67.075428|-0.738690|-0.445238|61273|

主要匹配对照：同一276453参数图的 calibrated 相对 identity_control；差别仅初始化。

|场景|准确率差（pp）|MacroF1差（pp）|
|---|---:|---:|
|clean|+0.084524|+0.091477|
|practical_high|-0.255357|-0.328135|
|practical_mid|+0.010714|-0.051938|
|practical_low_suburban|+0.332143|+0.232829|
|practical_high_urban|-0.245238|-0.330411|
|practical_mid_urban|+0.167857|-0.102471|
|practical_low_urban|+0.611905|+0.113118|

R55 fold_full仅作旧归一化参考：Conv宽度相同，但273765参数与归一化计算不同，不能当作完整参数及计算匹配控制。

|行|场景|较R55 full-fold准确率差（pp）|
|---|---|---:|
|calibrated|clean|+0.063095|
|calibrated|practical_high|+0.277976|
|calibrated|practical_mid|+0.530952|
|calibrated|practical_low_suburban|+0.813095|
|calibrated|practical_high_urban|+0.361905|
|calibrated|practical_mid_urban|+0.691071|
|calibrated|practical_low_urban|+1.058929|
|identity_control|clean|-0.021429|
|identity_control|practical_high|+0.533333|
|identity_control|practical_mid|+0.520238|
|identity_control|practical_low_suburban|+0.480952|
|identity_control|practical_high_urban|+0.607143|
|identity_control|practical_mid_urban|+0.523214|
|identity_control|practical_low_urban|+0.447024|

|行|coarse/action/prior参数|权重文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|calibrated|107034/79925/89494|1460364|30761512|
|identity_control|107034/79925/89494|1460428|30761512|

|行|口径|同卡父/本版median ms|本版P95 ms|本版条/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|calibrated|G_only_B1|45.4561/121.1858|131.3246|8.2823|67194880/1084928|
|calibrated|G_only_B32|81.8886/152.3943|157.5236|210.9150|276216832/227691520|
|calibrated|G_plus_F_B1|56.2264/132.8959|143.2166|7.5326|67194880/1201152|
|calibrated|G_plus_F_B32|90.4071/163.5098|168.0914|195.4035|276216832/227691520|
|identity_control|G_only_B1|45.1120/119.8319|130.9433|8.3627|67194880/1084928|
|identity_control|G_only_B32|81.5328/153.4852|161.1659|208.1679|276216832/227691520|
|identity_control|G_plus_F_B1|55.7322/130.8974|141.7991|7.6018|67194880/1201152|
|identity_control|G_plus_F_B32|93.0624/162.9911|171.1399|195.9742|276216832/227691520|

同卡64条sourceV、B1/B32、三轮交替顺序、同步预加载GPU输入，G及G+原F50分别计时，包含手写组统计计算；排除传输/预处理/loader。驻留为三G+一F共享suite；增量峰值对应当前前向，不称独立部署显存。MAC仅Conv/Linear，排除FFT/CG/norm/激活/访存和F。参数减少不能推定提速；不同卡绝对时间不作排名。

源校准一次共用128物理样本（64L_s/64U_s），received01/20，256父+256恒等子G输入，共512，耗时12.1326s；0clean/TX/F/V/target/optimizer，父及恒等子收集期间不变。原R55同源256父G的3.9369s折叠统计复用单列，不重复记两份。

完整pre-GN组均值和log(var+eps)配对，每组256样本，无空间抽样；identity-centered ridge1e-3、斜率[.25,4]、自由截距。新参数在校准后与原参数联合训练，系数和索引不作推理输入。下表两种统计量单位不同，不合并为一种误差；局部拟合下降不证明全G或身份收益。

|统计量|module等权MSE前/后|触界斜率数|
|---|---|---:|
|mean|0.001347569/0.0003364657|75|
|logvariance|0.057365862/0.014595072|88|

calibrated首轮→末轮：loss 0.090124745→0.023504307；iq_nmse 0.057799001→0.022899272；spectral_error 0.013420906→0.0059940007；teacher_stage_nmse 0.031654699→0.0030533478；每轮中位121.590s。日志类别：[]

identity_control首轮→末轮：loss 0.085651541→0.02341032；iq_nmse 0.055311829→0.022815823；spectral_error 0.012161174→0.0059783462；teacher_stage_nmse 0.029731653→0.0029557947；每轮中位121.951s。日志类别：[]

完整20轮曲线、逐TX/RX/day、修复/损伤、全帧/主体232/尾24重建误差和F配对clean几何均保留。原组件轮次和300/串行140pass仅描述原参数继承；新2688统计参数仅20轮，新增校准及F50训练另计。追加训练预算、初始化、归一化计算和压缩分别披露。

两套独立权重已取回并CPU严格重载，合成零IQ有限输出及参数核对通过。使用experiments.pamf_r46_stat_transport.runtime.load_checkpoint，float32[B,2,256]原生幅度输入/同形输出；恢复器无需教师/F/源数据/校准文件。权重、数据和原始日志不入Git。

识别率、IQ误差和特征几何不能单独证明每个恢复细节属于输入TX；原70%/5pp和TX细节保真目标保持ACTIVE。
