# 旧普通 IQ 增强的 RX1 留出结果

源资格：**PASS**。两个scratch E20已完成；目标数据访问0。

|模型|held RX1 V准确率|macro-F1|seen RX V均值|seen最差RX|
|---|---:|---:|---:|---:|
|N0|76.222222%|75.187958%|97.939815%|95.148148%|
|N1|82.629630%|82.547570%|97.615741%|94.407407%|

N1−N0 held accuracy +6.407407pp，macroF1 +7.359612pp；修复513、新增错误167。seen V下降+0.324074pp。

完整解析40epoch/8880step/40行nativeCSV和两份完整stdout。实际全L/U序列5994条逐条匹配。全部真实前向4819956，矩阵墙钟883.864s；额外本地合成1536网络前向另计。

准确率、macroF1、混淆和逐TX/day结果由保存的5400×2默认头决策独立复算；CE沿用执行器记录。L/U角色与seed相同；实际native args差异仅output_dir,use_aug,candidate_id。

这是已使用过的sourceRX1折、一个model seed，不是目标测试或完整DG证明。U真值始终隐藏。原ordinary配方整体的阳性不等于其中PA等每项都已证明身份保持。固定预算用尽不等于收敛。

FAIL按原规则结束本候选，不换RX/seed/参数；PASS只支持另登记其他四个源折，不自动宣称clean/urban双收益。原正式测试IQ未修改，lowurban>60%与suburban保持仍待正式证明。

图：complete_training_curves.png、held_source_strata.png、ordinary_and_cost.png；逐数值表见同目录CSV。


实际运行与科学状态：**VERIFIED / 源资格PASS**。训练实现commit96365a0aafb0ca7570b98f7e3737e3803e90d4d2，原release未热改；两个worker和dispatcher自然退出，complete=ARTIFACTS_COMPLETE，无failure。单archive比较、remotecompile、初始真实scratchCP save/reload noquery smoke和启动PID/CWD/argv/GPU/log增长均已保存。本地离线分析完整读取最终产物，没有重训或重测。

两组只训练RX3/4/6/8的L5040/U45360，source seenV21600；RX1 L1260/U11340未使用，原V中RX1的5400包只作FINAL评价。保持原单V角色，RX是评价分层。当前SIG_TIME/RAW_HZ/NoExplicitCFO/joint160/defaultcos、LS.01、两次同IQ RDrop、oldgate/ownEMA、13L+7U，所有模型与默认头正常联合训练，没有另训头/星地训练/新U机制。源有效契约不能当作fullL6300权重复用来源。

源TX与day收益必须完整保留：

|分层|N0准确率|N1准确率|差值（百分点）|
|---|---:|---:|---:|
|TX0|93.4444%|81.5556%|−11.8889|
|TX1|84.7778%|83.4444%|−1.3333|
|TX2|62.4444%|76.3333%|+13.8889|
|TX3|66.0000%|66.0000%|0|
|TX4|100.0000%|100.0000%|0|
|TX5|50.6667%|88.4444%|+37.7778|
|day1|74.3333%|82.6667%|+8.3333|
|day2|78.7222%|87.1667%|+8.4444|
|day3|75.6111%|78.0556%|+2.4444|

因此这是源RX1折的净收益，不能说所有TX都改善。seen最差RX从95.1481%降至94.4074%（−.7407pp），也如实保留；本次预定health门槛是seen均值下降<=.5pp，并非事后改用更有利指标。

训练机制与负观察：N1从早期保持更高的增广L CE/KL及梯度，而N0训练损失更低；不能用更低一致性损失代替泛化成绩。U在E14按原规则启用，N1接受量没有增加；U真值始终隐藏，不能计算其伪标签精度。默认W、原融合gain、SIG_TIME投影参数的实际变化已按20轮记录，没有新增部署参数。两行关键loss/gradient均有限，跳过更新0；L阶段总目标与mean两CE+symKL最大差分别2.3842e−7/3.5763e−7。原eval.py固定返回probe_dom_acc=NaN，故每轮nonfinite_val_metric_count=1对应未运行探针空值，不是身份预测/梯度非有限；未启用拟合probe。

实际普通增强：DAC概率0，class_signature关闭。nominal base cfg保留原PA.14、time_shift.35、amp/phase.45、CFO.35、phase_noise.30、AWGN.40、多径.18、DC.30、taper.25；实际epoch概率经原schedule缩放，E20 PA=.035084、AWGN=.100240、multipath=.045108。所有完整范围、实际args与每epoch概率保存在ordinary_base_config/actual_native_args/mechanism_progress。累计波形MSE=.181549，记录到的最大分量绝对值204.893；不能因整体均值阳性把每项都称为轻度或身份保持，也不能仅凭峰值认定某个操作是收益/损伤原因。

成本：两行各20epoch/4440更新，合计40model-epoch/8880更新；没有加载父模型。每行L1136640、studentU396368、EMA396368、student源eval475202（含smoke2）、heldV5400，合计每行2409978真实网络前向、两行4819956。N0/N1训练主体计时854.587/875.203s；矩阵真实墙钟883.864s。heldV及收尾差额包含在矩阵墙钟，源主体与整行不能混称。额外本地合成1536网络前向另计；完整已知网络前向4821492。离线解析/绘图及失败CP读取的CPU时间未统一计时，不伪造其墙钟。

REJECTED_EXTRA_GATE：离线工具先尝试了非必需的CP全张量比对，读取初态成功、读finalpayload因baseline_origin_sat_view反序列化依赖缺失失败。该附加项已取消阻塞且未重试/修复，没有网络前向或训练更新。实际同初始化规则/seed/nativeargs、本地198个初始state张量一致与真实scratchCP smoke已核实；本轮没有独立完成两个实际initialCP的逐张量比较，不作该项断言。该分析故障不更改训练/预测或源资格。

复现：预登记immutable CWD下普通N607 Python运行 `experiments/cvs_ordinary_fold20/runner.py --row N0-RX1-s2026092701` / N1；已完成root不得重新提交。离线完整采集/分析命令 `F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_ordinary_fold20.harvest --output E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final --finish`；使用已保存cache的分析不重读IQ或checkpoint。

未完成：其余源heldRX3/4/6/8、跨model seed、full-source普通增强与As匹配组合、正式原clean+6practical。当前结果仅允许按原方案另行登记其余4源折，不由该单fold直接启动target或宣称lowurban>60%。即使五fold通过，正式联合收益仍需独立固定预算训练和逐row冻结后truth-last验证。整体目标保持active。

实际图与表：

- [完整六面板训练/seenV/LR曲线](E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final/complete_training_curves.png)
- [held RX1逐TX/day结果](E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final/held_source_strata.png)
- [实际普通增强与前向成本](E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final/ordinary_and_cost.png)
- [逐分层数值](E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final/source_strata.csv)、[完整训练曲线数值](E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final/training_curves.csv)、[成本](E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final/measured_costs.csv)
