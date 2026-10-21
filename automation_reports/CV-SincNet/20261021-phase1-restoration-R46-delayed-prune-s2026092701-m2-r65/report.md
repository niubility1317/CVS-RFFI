# R65：固定总预算延后裁剪的正式性能与成本

**VERIFIED / ANALYZED**。两固定日程完整 E20/4940、source1800×7 和正式168000×7；全部七预测固定后独立truth-last，实际所属进程退出，独立权重严格重载通过。所有正负结果保留，完整目标继续ACTIVE。

性能优先。两行均直接M35初始化和全程冻结原M35 IQ教师，分别dense5+compact15、dense10+compact10，最终273765/context185615。原M35 GN4通道索引在训练前固定，两行不重选；只在同row源训练阶段切取当前权重，AdamW重置一次，global LR/data/teacher日程不重启。

所有coarse/action/prior联合，原pairedclean IQ+.05谱误差+线性1→.1三阶段教师；source L126000/U1134000、teacher/student各1260000输入。0TX/F/V/target优化、额外校准IQ或新推理参数。源模拟配对clean自监督，不称cleanfree。

方法commit `4fd058f9bae79687f924e06d057e6a20083ff6dc`；完整命令/环境/数据/矩阵见[登记](experiment.json)，全20轮曲线、逐TX/RX/day、修复损伤、IQ/FFT描述量见[结果](results.json)和[CSV](row_metrics.csv)。

|行|最终参数|较M35减参|原R17逐景1pp|M35逐景1pp|low urban（%）|其他五景gap≤5pp|
|---|---:|---:|---|---|---:|---|
|prune_after5|273765|21.897689%|True|True|67.698810|False|
|prune_after10|273765|21.897689%|True|True|68.024405|False|

|行|场景|准确率（%）|MacroF1（%）|较M35（pp）|较原R17（pp）|较R49同预算（pp）|净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|prune_after5|clean|84.423810|84.121650|-0.032738|-0.163095|-0.017262|-424|
|prune_after5|practical_high|82.630952|82.219808|+0.178571|+1.311905|+0.749405|+70389|
|prune_after5|practical_mid|81.531548|81.189745|+0.309524|+1.953571|+1.105357|+69913|
|prune_after5|practical_low_suburban|79.925595|79.713118|+0.305357|+1.929167|+0.988690|+68705|
|prune_after5|practical_high_urban|81.527381|81.182139|+0.140476|+1.173810|+0.765476|+70273|
|prune_after5|practical_mid_urban|74.699405|74.846428|+0.252381|+1.014286|+0.968452|+67563|
|prune_after5|practical_low_urban|67.698810|68.418393|+0.188690|+0.482143|+0.876786|+62831|
|prune_after10|clean|84.908929|84.635704|+0.452381|+0.322024|+0.467857|+391|
|prune_after10|practical_high|82.825595|82.414014|+0.373214|+1.506548|+0.944048|+70716|
|prune_after10|practical_mid|81.620238|81.219920|+0.398214|+2.042262|+1.194048|+70062|
|prune_after10|practical_low_suburban|80.155952|79.806279|+0.535714|+2.159524|+1.219048|+69092|
|prune_after10|practical_high_urban|81.743452|81.359740|+0.356548|+1.389881|+0.981548|+70636|
|prune_after10|practical_mid_urban|75.063095|74.980802|+0.616071|+1.377976|+1.332143|+68174|
|prune_after10|practical_low_urban|68.024405|68.390564|+0.514286|+0.807738|+1.202381|+63378|

主比较为5+15相对10+10；相同初始权重、固定索引、最终图和全局预算，密集/压缩计算量及optimizer重置时机不同。R53静态裁剪与R49保留图为只读历史控制，不继承其权重。静态控制无中途重置，差异不能全归因于裁剪时机。

|场景|5+15减10+10准确率（pp）|MacroF1（pp）|
|---|---:|---:|
|clean|-0.485119|-0.514053|
|practical_high|-0.194643|-0.194205|
|practical_mid|-0.088690|-0.030175|
|practical_low_suburban|-0.230357|-0.093162|
|practical_high_urban|-0.216071|-0.177601|
|practical_mid_urban|-0.363690|-0.134374|
|practical_low_urban|-0.325595|+0.027829|

|行|dense/compact轮次|dense/compact IQ曝光|dense/compact秒|结构转换秒|
|---|---|---|---|---:|
|prune_after5|5/15|315000/945000|405.9518/1141.9514|0.0928|
|prune_after10|10/10|630000/630000|810.4122/760.0301|0.0941|

相同20轮不代表计算量相同；密集阶段被移除参数只新增5或10轮，保留参数继承80+新增20=100轮。源初始化和结构转换成本单列，完整训练墙钟包含实际阶段转换开销。

|行|场景|较R53静态同图准确率（pp）|
|---|---|---:|
|prune_after5|clean|-0.104762|
|prune_after5|practical_high|+0.222024|
|prune_after5|practical_mid|+0.182738|
|prune_after5|practical_low_suburban|+0.067262|
|prune_after5|practical_high_urban|+0.115476|
|prune_after5|practical_mid_urban|-0.361905|
|prune_after5|practical_low_urban|-0.828571|
|prune_after10|clean|+0.380357|
|prune_after10|practical_high|+0.416667|
|prune_after10|practical_mid|+0.271429|
|prune_after10|practical_low_suburban|+0.297619|
|prune_after10|practical_high_urban|+0.331548|
|prune_after10|practical_mid_urban|+0.001786|
|prune_after10|practical_low_urban|-0.502976|

|行|coarse/action/prior参数|文件字节|Conv/Linear MAC每IQ|完整训练秒|训练peak allocated字节|
|---|---|---:|---:|---:|---:|
|prune_after5|106138/79477/88150|1388956|30761512|1548.7557|1423320064|
|prune_after10|106138/79477/88150|1388956|30761512|1571.5451|1423320064|

|行|口径|父/本版median ms|本版P90/P95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---|---:|---|
|prune_after5|G_only_B1|50.0095/47.9531|61.8137/64.1438|19.4615|67194880/1084928|
|prune_after5|G_only_B32|92.3715/75.6348|88.7161/90.6989|411.7880|276216832/227691520|
|prune_after5|G_plus_F_B1|60.4939/60.8420|75.6693/77.3939|15.6153|67194880/1201152|
|prune_after5|G_plus_F_B32|102.6130/88.1807|102.5395/105.8699|351.1665|276216832/227691520|
|prune_after10|G_only_B1|48.5398/47.7989|62.2734/65.8292|19.5430|67194880/1084928|
|prune_after10|G_only_B32|87.9818/76.1956|96.6743/100.0840|403.7665|276216832/227691520|
|prune_after10|G_plus_F_B1|59.6301/58.5131|74.0820/75.9604|16.0549|67194880/1201152|
|prune_after10|G_plus_F_B32|99.4803/87.5437|102.2777/103.1969|360.0919|276216832/227691520|

成本使用同卡64条sourceV、G/G+冻结原F、B1/B32、3轮交替次序和预热。同步预加载GPU输入，不含传输/前处理/loader；三G一F共享驻留suite，增量峰值是当前前向，不当作独立部署显存。Conv/Linear MAC不含FFT/CG/norm/activation/F，训练peak包括教师/快照/优化器/激活；不同GPU时段不据绝对数值作速度排名。

prune_after5完整20轮首→末：loss 0.025517688→0.024087597；iq_nmse 0.024150952→0.023456548；spectral_error 0.0063670583→0.0061129153；teacher_stage_nmse 0.0010483832→0.0032540399；每轮中位76.3236s；lowurban70=False；otherfive5pp=False；日志类别[]
prune_after10完整20轮首→末：loss 0.025517727→0.025829523；iq_nmse 0.024150978→0.025072696；spectral_error 0.0063670612→0.006433318；teacher_stage_nmse 0.0010483961→0.004351607；每轮中位78.9501s；lowurban70=False；otherfive5pp=False；日志类别[]

原R17逐景1pp是退化上限，性能提升、lowurban≥70%与其他五受损场景距rawclean84.6761904762%≤5pp分别判断。单seed、相关视图、已暴露且由用户指定父的工程基准；全部正负结果保留，不称新独立盲测。重建、识别率或F几何不能单独证明每个恢复细节来自输入TX。完整目标 ACTIVE。

旧每组件80+本次20=100、组件轮次和300、串行140 pass，原F50训练另计。相对M35增加20轮，另报告同增量R49控制，不能把追加预算收益全部归于减参。

独立权重位于local_artifacts/pamf_restoration/r46_delayed_prune_r65_20261021/models/<arm>/final.pt；运行experiments.pamf_r46_delayed_prune.runtime.load_checkpoint后输入float32[B,2,256]原生IQ，输出同形IQ。恢复器无需教师/F/源缓存，识别时同一恢复IQ接原冻结F50默认头与FFT/PA/时域/统计分支，无另训头。


训练期间6次被动采样覆盖约25秒：GPU2整卡利用率均值98.16666666666667%，GPU3为96.33333333333333%。整卡值包含其他owner，不能归于本进程或作为独占加速证明；原始PID/采样留在本地，不写入公开结果。
