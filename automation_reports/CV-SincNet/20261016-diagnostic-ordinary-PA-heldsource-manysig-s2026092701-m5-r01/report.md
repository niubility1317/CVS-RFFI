# 额外 PA 对冻结默认头的源身份影响

主判据：**PASS**。零训练更新；五个N0 final20，原held L6300。

|held RX|RAW准确率|FULL准确率|NO_PA准确率|保持率变化/pp|准确率变化/pp|实际PA作用数|
|---|---:|---:|---:|---:|---:|---:|
|1|73.8095%|64.5933%|65.3869%|+1.0618|+0.7937|698|
|3|50.8730%|44.4692%|44.6478%|+1.1213|+0.1786|729|
|4|84.1270%|74.9752%|75.7540%|+0.9611|+0.7788|684|
|6|77.3016%|67.1825%|67.9415%|+1.0395|+0.7589|713|
|8|62.3016%|53.1448%|53.6359%|+1.4490|+0.4911|713|

NO_PA−FULL平均保持率+1.126556pp，事前条件逐项{'mean_retention': True, 'positive_RX': True, 'worst_RX': True}。

RAW只前向一次；FULL/NO_PA各16draw。每折1260个唯一源L样本，所有20160配对记录独立复算，RAW正确集合固定；没有按学生增强置信度筛样本。逐TX/day/PA作用子集和macroF1见CSV。

真实样本前向207910（含5个CPsmoke共10），0 optimizer/额外head更新，墙钟23.314246s。继承五个E20权重的100model-epoch另列，完整五foldN0/N1研发成本200model-epoch不隐藏。本地合成网络前向96/增强输入192包另计。

每折前级逐值相同、非PA作用包输出相同、模型参数/缓冲区冻结已实际断言，source配对预测由离线工具独立复算。没有V/U/target、星地输入或新训练。

这只证明该冻结头在当前条件下的输入效应，不能把它当成PA导致旧N1训练后RX6/8下降的因果证据，不能证明NO_PA是有效Ag，也没有正式clean/urban或跨seed结果。主判据FAIL不改强度/seed/RX或切换次要终点救援；PASS也不自动训练。

独立首个读回时23s任务已终止，运行中PID/CWD/日志增长快照缺失；发布记录、实际resolved CWD/GPU/commit、5smoke、全部draw日志、终态进程退出及结果已保存。没有补跑。最初本地观察器文件名遮蔽stdlib inspect，已改名；没有远端失败或重启。


实际运行与科学状态：**VERIFIED / 直接PA输入风险判据PASS**。实际代码b6f646090df72865599de63c19af549d14ed9445；单归档transfer对照/remotecompile通过，原release未改。五fold原N0权重来源在载入前核对，完整原图构造后strict载入成功，5个真实CP noquery smoke通过，全部16draw×160配对batch/折完成，参数/buffer逐值不变。无技术失败，无重发/训练/目标访问。

NO_PA−FULL平均准确率+0.600198pp；不过NO_PA相对RAW仍低8.209325pp，因此额外PA不能解释普通增强造成的全部冻结头损伤。不能把直接输入差额加到训练或目标成绩上。部分TX层有负差，完整CSV保留；五RX净增不等于所有类别均增。

观察缺项：首次独立读取时worker/dispatcher已自然终止，未获得运行中的独立PID/CWD/日志增长快照。现有submit/launch、实际resolved CWD/GPU/commit、全部逐draw stdout、源计数及终态进程缺失提供完成证据。没有为补观测重复运行。最初本地脚本命名inspect.py遮蔽stdlib，已改名readback_local.py；这是本地读取失败，不是远端/NumPy环境失败，未安装或修改环境。

预算：207910真实网络前向＋96本地合成网络前向＝已知208006；另有192本地合成增强输入包，不与网络前向混算。23.314246s为worker run主体实测，包含缓存/CP加载与推断，不包含Python bootstrap、发布/传输、离线分析绘图的全部时间；后者未统一计时。历史五N0权重100model-epoch及N0/N1完整200model-epoch分别披露。

复现入口：预登记immutable CWD中普通N607 Python执行experiments/cvs_ordinary_pa_source/runner.py；同root不得重复启动。离线命令：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_ordinary_pa_source.harvest --output E:/type10-7/local_artifacts/cvs_ordinary_pa_source_20261016/final --state E:/type10-7/local_artifacts/cvs_ordinary_pa_source_20261016/launch_readback.json --cached。离线仅源决策/统计，无IQ/CP/native导入。

实际输出：

- [五RX默认头结果与TX/day分层](E:/type10-7/local_artifacts/cvs_ordinary_pa_source_20261016/final/source_PA_effects.png)
- [PA前后峰值与功率变化](E:/type10-7/local_artifacts/cvs_ordinary_pa_source_20261016/final/PA_waveform_effects.png)
- [逐RX结果](E:/type10-7/local_artifacts/cvs_ordinary_pa_source_20261016/final/fold_results.csv)
- [逐TX/day及PA作用层](E:/type10-7/local_artifacts/cvs_ordinary_pa_source_20261016/final/source_strata.csv)
- [波形统计](E:/type10-7/local_artifacts/cvs_ordinary_pa_source_20261016/final/waveform_summary.csv)
