# 普通增强关闭 PA：RX1 独立 E20 训练结果

源资格：**FAIL**。一个新scratchE20，复用N0/N1结果不继承权重。

|模型|heldRX1准确率|macroF1|seenV均值|seenV最差RX|
|---|---:|---:|---:|---:|
|N0|76.222222%|75.187958%|97.939815%|95.148148%|
|N1|82.629630%|82.547570%|97.615741%|94.407407%|
|N2|79.277778%|79.035528%|97.652778%|94.611111%|

固定门槛：{'vs_N0_accuracy': True, 'vs_N0_macroF1': True, 'seen_health': True, 'N1_accuracy_retained': False, 'N1_macroF1_retained': False, 'complete_health': True}。N2−N0 accuracy +3.055556pp；N2−N1 -3.351852pp。

新20epoch/4440step/nativeCSV与完整stdout已解析；全部5994个真实L/U批次与两旧控制逐条匹配。PA guard实际零调用，其余ordinary保留。真实网络前向2409978，矩阵墙钟899.753s；原控制40modelepoch/8880update/4819956前向另列，本地512合成网络前向另计。

源only，0target/新头/新U机制，heldL不入训练；只在FINAL评价源heldV。默认头决策与macroF1独立复算，逐TX/day正负结果完整保存。E20预算用尽不是收敛。

PA输入诊断PASS不能替代本训练结果；本单fold也不能证明完整Ag或clean/urban双收益，跨seed未验证。FAIL按原规则结束NoPA候选，PASS仅允许另登记其余固定四RX；不自动进入target或混合训练。


实际运行与科学状态：**VERIFIED / RX1源资格FAIL**。实际训练代码2b916b8892fce0dd1ae80ff9c8f1d649445b245b；单archive transfer对照/remotecompile、原scratchCP noquery smoke、PID/CWD/argv/GPU/log增长均已保存，模型与dispatcher自然退出，无failure。原immutable release未热改，不重训/重测旧N0/N1。

当前863730参数SIG_TIME/RAW_HZ/NoExplicitCFO/joint160/defaultcos，25MHz/IQ256/equalized1/nativeRMS；LS.01/RDrop同IQ两前向/AdamW/cos/oldgate/ownEMA正常执行，默认头和骨干联合更新。每行只训练RX3/4/6/8的L5040/U45360，seenV21600；heldRX1 L1260/U11340不用，原V中RX1 V5400仅FINAL。无As/新G/独立头/U真值/目标；N2从零，未加载两旧控制模型。

actualnativeargs只差PA开关和run/candidate/output标识，nativePA flagfalse、simulate_pa调用guard实际零次，其余ordinary4440call保持。原p_pa仍是nominal正值，不把概率字段误读为实际执行；真实开关和guard证明PA关闭。包装resolved顶层33/17是遗留报告描述，实际nativeargs13/7、20epoch/4440成功更新及全日志证明本次总预算E20，没有执行E50或延长。

完整20epoch/4440step/nativeCSV及stdout已解析，有限关键loss/grad与完整6TX×4RX×3day覆盖通过。原未运行probe_dom_acc的NaN空值不是身份数值失败，未拟合probe。原控制的完整40epoch/8880step分析已完成并复用。本地原args比较fixture误判相同NaN占位发生在模型前0fwd，已修正测试对等，没有改native值。模型不同于对照的实际initialCP未做额外逐张量审计；同初始化规则/seed、native图、本地198初态tensor和自身CPsmoke已核实，不作超出证据断言。

成本：新增20model-epoch/4440updates/真实网络前向2409978，含L/studentU/EMA/sourceeval和smoke2/heldV；原对照40model-epoch/8880updates/4819956真实前向另列，没有权重继承。新增本地512合成网络前向/256增强输入包另列；矩阵墙钟899.753173s，CPU分析/绘图时间未统一计时。预算耗尽不等于收敛。

逐层正负结果：

|分层|N0准确率|N1准确率|N2准确率|N2−N1/pp|
|---|---:|---:|---:|---:|
|TX0|93.4444%|81.5556%|82.4444%|+0.8889|
|TX1|84.7778%|83.4444%|84.4444%|+1.0000|
|TX2|62.4444%|76.3333%|80.0000%|+3.6667|
|TX3|66.0000%|66.0000%|65.8889%|-0.1111|
|TX4|100.0000%|100.0000%|100.0000%|+0.0000|
|TX5|50.6667%|88.4444%|62.8889%|-25.5556|
|day1|74.3333%|82.6667%|78.9444%|-3.7222|
|day2|78.7222%|87.1667%|84.3333%|-2.8333|
|day3|75.6111%|78.0556%|74.5556%|-3.5000|

PA冻结输入诊断PASS仅是本训练的来源依据，不能将其+0.6002pp直接加入训练收益；本RX1模型也不是完整Ag/跨seed/正式clean或urban证据。FAIL按原规则结束NoPA候选，不以换强度/RX/seed或另换主终点救援；PASS只允许独立另登记固定剩余4RX，再按原五fold规则处理。当前无目标预测与正式practical新成绩；完整clean净增/lowurban>60%/suburban保持/跨seed目标仍active。

复现入口：预登记immutable CWD中普通N607 Python运行experiments/cvs_ordinary_noPA20/runner.py；已完成root不能重复启动。离线收尾：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_ordinary_noPA20.harvest --output E:/type10-7/local_artifacts/cvs_ordinary_noPA20_20261016/final --state E:/type10-7/local_artifacts/cvs_ordinary_noPA20_20261016/terminal_readback.json --prior-output E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final --cached。离线不读IQ/CP或加载native。

实际输出：

- [完整三组训练/seenV/LR曲线](E:/type10-7/local_artifacts/cvs_ordinary_noPA20_20261016/final/complete_training_curves.png)
- [逐TX/day默认头结果](E:/type10-7/local_artifacts/cvs_ordinary_noPA20_20261016/final/held_source_strata.png)
- [全部分层数值](E:/type10-7/local_artifacts/cvs_ordinary_noPA20_20261016/final/source_strata.csv)
- [完整训练曲线数值](E:/type10-7/local_artifacts/cvs_ordinary_noPA20_20261016/final/training_curves.csv)
