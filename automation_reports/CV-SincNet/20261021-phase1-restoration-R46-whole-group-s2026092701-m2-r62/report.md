# R62：整组压缩的正式性能与成本

VERIFIED / ANALYZED：两行完整E20/4940、源1800×7、正式168000×7、逐行七预测固定后独立truth-last、全20轮/同卡成本/实际所属进程退出和独立权重CPU严格重载已核实。group_preserve 248828参数，较指定M35七景提高0/下降7，low urban 66.230952%（较M35-1.279167pp）；较R49同增量预算提高0/下降7，low差-0.591071pp；70%=False、其他五景gap≤5pp=False；同卡G B1父/本版48.9053/48.2824ms，G+F B1父/本版59.1257/57.9017ms；regroup_control 248828参数，较指定M35七景提高0/下降7，low urban 66.207738%（较M35-1.302381pp）；较R49同增量预算提高0/下降7，low差-0.614286pp；70%=False、其他五景gap≤5pp=False；同卡G B1父/本版46.9371/47.3193ms，G+F B1父/本版57.3480/57.7004ms。同初始权重/通道/预算下GN3相对GN1七景提高6/下降1，low urban差+0.023214pp；归约和实际成本不同，不称完美计算匹配。直接指定350521初始化/冻结IQ教师不变，全prior保留并更新，原pairedcleanIQ/谱/1→.1IQ教师，全组件联合；0TX/F/V/target优化/新校准/新参数/额外G前向。全部正负结果保留，不把减参或1pp容差当作性能提升或逐细节TX真实性。完整目标ACTIVE。

**VERIFIED / ANALYZED**。两固定行完整 E20/4940、source1800×7 和正式168000×7；每行七景预测固定后独立 truth-last，实际所属 controller/worker 全退出，独立权重严格重载通过。全部正负结果保留，不反馈调参或选择性重跑。

性能优先。两行同248828参数、同选组和初始权重、同训练预算，只改变 context GN3/GN1；prior完整112594参数。权重选择删除一个完整原GN组，无校准IQ。GN3保留原每组通道宽度，GN1重新合组；层级同值等价不代表整个恢复器等价，归约方式和实测成本不同。

直接指定350521父初始化和冻结三阶段IQ教师；全coarse/action/prior联合，pairedclean IQ NMSE+.05谱误差+线性1→.1教师，全63000源记录/20缓存轮/globalmicro256。L126000/U1134000，0TX/F/V/target优化，0新增参数/校准IQ/额外G前向；合法源clean模拟配对的TX无标签自监督。

方法commit `7a8fa7bd836946c3493635cbe0966591ecf14e1e`。完整命令/环境/矩阵/数据权限见[登记](experiment.json)，完整曲线、每TX/RX/day、修复损伤、NMSE/FFT/特征描述见[结果](results.json)与[CSV](row_metrics.csv)。

|行|完整参数|较M35减参|原R17逐景1pp|M35逐景1pp|low urban（%）|其他五景gap≤5pp|
|---|---:|---:|---|---|---:|---|
|group_preserve|248828|29.011956%|True|False|66.230952|False|
|regroup_control|248828|29.011956%|False|False|66.207738|False|

|行|场景|准确率（%）|MacroF1（%）|较M35（pp）|较原R17（pp）|较R49同预算（pp）|净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|group_preserve|clean|84.322024|84.041123|-0.134524|-0.264881|-0.119048|-595|
|group_preserve|practical_high|81.034524|80.513802|-1.417857|-0.284524|-0.847024|+67707|
|group_preserve|practical_mid|80.107143|79.661697|-1.114881|+0.529167|-0.319048|+67520|
|group_preserve|practical_low_suburban|78.679167|78.357848|-0.941071|+0.682738|-0.257738|+66611|
|group_preserve|practical_high_urban|79.963690|79.503637|-1.423214|-0.389881|-0.798214|+67646|
|group_preserve|practical_mid_urban|73.188690|73.162088|-1.258333|-0.496429|-0.542262|+65025|
|group_preserve|practical_low_urban|66.230952|66.686781|-1.279167|-0.985714|-0.591071|+60365|
|regroup_control|clean|84.355357|84.066338|-0.101190|-0.231548|-0.085714|-539|
|regroup_control|practical_high|80.949405|80.469675|-1.502976|-0.369643|-0.932143|+67564|
|regroup_control|practical_mid|79.954167|79.573342|-1.267857|+0.376190|-0.472024|+67263|
|regroup_control|practical_low_suburban|78.632143|78.408469|-0.988095|+0.635714|-0.304762|+66532|
|regroup_control|practical_high_urban|79.841071|79.440278|-1.545833|-0.512500|-0.920833|+67440|
|regroup_control|practical_mid_urban|73.060119|73.125889|-1.386905|-0.625000|-0.670833|+64809|
|regroup_control|practical_low_urban|66.207738|66.759798|-1.302381|-1.008929|-0.614286|+60326|

主消融为GN3相对GN1：相同初始权重、通道和预算，保留全部逐景差；同预算未裁剪R49 retain_refine只读复用，不重复训练/继承其权重。宽度和归一化同时区别于R49，不把全部收益归于单一因素。

|场景|GN3−GN1准确率（pp）|MacroF1（pp）|
|---|---:|---:|
|clean|-0.033333|-0.025215|
|practical_high|+0.085119|+0.044127|
|practical_mid|+0.152976|+0.088355|
|practical_low_suburban|+0.047024|-0.050621|
|practical_high_urban|+0.122619|+0.063359|
|practical_mid_urban|+0.128571|+0.036199|
|practical_low_urban|+0.023214|-0.073017|

|行|coarse/action/prior参数|文件字节|Conv/Linear MAC每IQ|完整训练秒|训练peak allocated字节|
|---|---|---:|---:|---:|---:|
|group_preserve|77796/58438/112594|1275740|31548897|1527.4988|1276847104|
|regroup_control|77796/58438/112594|1275740|31548897|1527.5768|1276503040|

|行|口径|父/本版median ms|本版P90/P95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---|---:|---|
|group_preserve|G_only_B1|48.9053/48.2824|63.1211/64.7224|19.2738|67194880/79946240|
|group_preserve|G_only_B32|92.5700/82.6005|98.6401/100.7848|378.6072|276216832/276216832|
|group_preserve|G_plus_F_B1|59.1257/57.9017|74.1209/76.4633|16.0559|67194880/79946240|
|group_preserve|G_plus_F_B32|96.5823/91.0821|107.6726/112.1233|346.8920|276216832/276216832|
|regroup_control|G_only_B1|46.9371/47.3193|58.3268/62.6557|19.8478|67194880/79946240|
|regroup_control|G_only_B32|91.6933/79.1482|98.6656/100.3612|388.7444|276216832/276216832|
|regroup_control|G_plus_F_B1|57.3480/57.7004|74.2679/77.2018|16.1951|67194880/79946240|
|regroup_control|G_plus_F_B32|96.5593/89.8721|113.4048/114.9340|340.3555|276216832/276216832|

成本使用同卡64条sourceV、G/G+冻结原F、B1/B32、3轮交替次序和预热。同步预加载GPU输入，不含传输/前处理/loader；三G一F共享驻留suite，增量峰值是当前前向，不当作独立部署显存。Conv/Linear MAC不含FFT/CG/norm/activation/F，训练peak包括教师/快照/优化器/激活；不同GPU时段不据绝对数值作速度排名。

group_preserve完整20轮首→末：loss 0.27823031→0.023799172；iq_nmse 0.14429862→0.023163804；spectral_error 0.04088287→0.0059758076；teacher_stage_nmse 0.13188754→0.003365775；每轮中位76.5110s；lowurban70=False；otherfive5pp=False；日志类别[]
regroup_control完整20轮首→末：loss 0.35851386→0.023941636；iq_nmse 0.18364192→0.023276095；spectral_error 0.050839055→0.0060113963；teacher_stage_nmse 0.17232999→0.0036497109；每轮中位76.2030s；lowurban70=False；otherfive5pp=False；日志类别[]

原R17逐景1pp是退化上限，性能提升、lowurban≥70%与其他五受损场景距rawclean84.6761904762%≤5pp分别判断。单seed、相关视图、已暴露且由用户指定父的工程基准；全部正负结果保留，不称新独立盲测。重建、识别率或F几何不能单独证明每个恢复细节来自输入TX。完整目标 ACTIVE。

旧每组件80+本次20=100、组件轮次和300、串行140 pass，原F50训练另计。相对M35增加20轮，另报告同增量R49控制，不能把追加预算收益全部归于减参。

独立权重位于local_artifacts/pamf_restoration/r46_group_drop_r62_20261021/models/<arm>/final.pt；运行experiments.pamf_r46_group_drop.runtime.load_checkpoint后输入float32[B,2,256]原生IQ，输出同形IQ。恢复器无需教师/F/源缓存，识别时同一恢复IQ接原冻结F50默认头与FFT/PA/时域/统计分支，无另训头。


训练期间6次被动采样覆盖约25秒：GPU2整卡利用率均值100%，GPU3为97.66666666666667%。整卡值包含其他owner，不能归于本进程或作为独占加速证明；原始PID/采样留在本地，不写入公开结果。
