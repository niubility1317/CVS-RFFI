# R25 固定 F 源域跨记录特征诊断

状态：ANALYZED / SOURCE_ONLY_DESCRIPTIVE。五组已有source特征统计完成，0新预测／拟合／target，不改变R24或70%／5个百分点验收。

现有配对 clean 相似度使用同一个物理记录，可能受到共同波形内容影响。新增问题是：恢复的 F 特征是否仍接近不同 RX、不同日期的同 TX 源记录，而非仅接近同包 clean。跨记录／跨日不等于已解码验证的不同 payload；不得声称 payload 独立或逐个生成细节已证真。

仅复用已完整独立评分的1800条固定sourceV原F50 seed2026092701/defaulthead特征。候选为R17 unroll_unshared、R19 time_time、R23 prior_only、R24 tcn_prior／attention_prior，参考为原R10 reference_controls的clean identity特征。每个候选七视图全部prediction和source summary完成，匹配物理ID、原F及角色后，才打开sourceV标签；不读取target、权重或新IQ，不执行模型预测／梯度／拟合／分类决定更新。source_reference仅核对既有V物理角色和RX/day，不重验未变数据。

参考池按TX×RX×day分层，每格按opaque physical ID固定排序取前2条，不按特征／成绩选取。对每个评估记录，仅比较参考RX和day都不同的格，故同包／同RX／同日均不进入参考。每个有效格须包含全部6个TX的非零参考特征；缺格和零特征显式计数，不补造数据。固定单位L2余弦；逐格分别求同TX平均相似度与另外5个TX等权平均相似度之差，再对格等权。仅报告特征相似度差、正差比例、baseline/restored/clean与共同有效记录的差值，不产生额外分类预测、prototype分类器、分类头或优化阈值。按总体／TX／RX／day与全部七视图报告，跨视图相关，单seed。

该统计使用source truth作评分后的配对分组，不能称整条分析无标签。它仍依赖同一个F特征空间，不是独立识别器验证，不证明独立物理族、新TX泛化、真实在轨恢复或每个微小细节真实性。无新的源晋级条件，结果不用于重排R24；未来方法只能使用合法源证据研发。

最小执行：纯CPU、0GPU、2线程、原source缓存／固定prediction只读、新不可覆盖R25输出；本地聚焦负测、一次P0/P1、Git固定代码、登记后单归档／预检／编译。输入是已有完整prediction，不加载checkpoint，不新增模型smoke、数据重验或其他门槛。全部五行源评分完成后运行一次，并核对实际PID/CWD/argv/日志及完整输出。技术错误保留输出、不盲目重试；健康R24继续。

|ID|来源|要求|目标|状态|验证|说明|
|---|---|---|---|---|---|---|
|D25-01|身份问题|跨记录／RX／day固定成对余弦统计|分析工具|verified|正交TX合成数据／置换|不把同包匹配当身份真实性|
|D25-02|输入权限|完整source评分后标签、原F／ID匹配、0target|分析工具|verified|角色／F／缺项负测|仅复用现有预测|
|D25-03|统计口径|固定cap2、全部六TX、缺格／零特征及共同有效差值|分析工具|verified|缺格／零／重复ID测试|无分类器／拟合／重排|
|D25-04|执行|登记／Git／CPU发布／实际输出核实|登记及报告|verified|真实source预测运行|不阻断R24|
|D25-05|交付|七视图及TX/RX/day、完整结果与边界|报告|verified|独立指标复核|source非正式目标集|
|D25-06|真实性|独立物理族／新TX／多seed／逐细节|后续合格研究|deferred|当前输入不能证明|不宣称整个目标完成|

当前5verified、0pending、1deferred、0rejected、0blocked。17项检查、compile/help、一次P0/P1、真实完整source统计及独立分组加权核验、发布和进程退出均已核实。

实际本地命令：`python -X utf8 -m pytest tests/test_pamf_source_cross_record.py -q --maxfail=1`（17 passed）、`python -X utf8 -m py_compile tools/analyze_pamf_source_cross_record.py`、`python -X utf8 -m tools.analyze_pamf_source_cross_record --help`。无模型或checkpoint加载，故模型smoke不适用；真实运行将核实完整source已评分特征。

[完整源诊断报告](../automation_reports/CV-SincNet/20261020-diagnostic-restoration-source-cross-record-F50-s2026092701-r25/report.md)。源同F几何证据不等于独立TX真实性或正式性能完成。
