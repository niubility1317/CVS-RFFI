# 普通 IQ 增强的五个源 RX 留出结果

源资格：**FAIL**。一个model seed，五个源RX折；RX1复用已有结果，其余8个独立E20完成。

|held RX|N0准确率|N1准确率|增益/pp|macroF1增益/pp|seenV下降/pp|
|---|---:|---:|---:|---:|---:|
|1|76.222222%|82.629630%|+6.407407|+7.359612|+0.324074|
|3|48.888889%|53.314815%|+4.425926|+5.933065|+0.226852|
|4|83.018519%|85.851852%|+2.833333|+3.109929|+0.282407|
|6|77.425926%|76.500000%|-0.925926|-0.686051|+0.185185|
|8|64.222222%|61.629630%|-2.592593|-4.140521|+0.250000|

平均准确率差+2.029630pp，macroF1差+2.315207pp。源门槛逐项：{'mean_accuracy': True, 'positive_folds': False, 'worst_fold': False, 'mean_macroF1': True, 'seen_health': True, 'complete_health': True}。

新8模型共160model-epoch/35520updates/真实前向19279824；复用RX1 pair40model-epoch/8880updates/真实前向4819956。新矩阵墙钟1096.709s。本地合成前向另计。

所有新行完整20epoch/4440step/nativeCSV/stdout已解析；所有四对实际L/U物理样本序列逐条相同。默认头联合更新、源TX/RX/day覆盖和U真值隐藏按实际日志核对；保存的heldV默认头决策独立复算。

五个fold训练集相互重叠，不是五个model seed；只有FINAL heldV读出，不存在heldV训练曲线。该结果不等于正式跨RX clean/urban收益，不认证每个旧PA等算子的身份保持。E20用尽不是收敛。

无target、无As、无旧权重。FAIL结束普通增强作为Ag的本候选，不换RX/强度/seed救援；PASS也不自动启动target或扩seed。原正式practical IQ与信道保持。


实际状态：**VERIFIED / 源资格FAIL**。训练代码2cbaee17215772bc50ec63d77b771675894991ec；原immutable release无热改，8个worker与dispatcher自然退出，无failure。发布归档单次比较、remotecompile、真实scratchCP noquery smoke及启动PID/CWD/argv/GPU/log增长通过；本地离线收尾不读取IQ/CP或执行网络。

模型保持863730参数的当前SIG_TIME/RAW_HZ/NoExplicitCFO/joint160/defaultcos，25MHz/IQ256/equalized1/nativeRMS。每行留出一个源RX，只训练其余4RX L5040/U45360，seenV21600为原V评价分层；heldRX原V5400仅FINAL。held L1260/U11340不使用，U TX始终隐藏，无full-source/跨fold/EMA/其他checkpoint继承，无As/target/新损失/分支/头。全部参数和默认头按原训练联合更新，13L+7U。

N0/N1每对实际nativeargs只允许candidate_id/output_dir/use_aug差异；每对全部5994条L/U批次物理样本序列逐条相同，覆盖6TX×4训练RX×3day。所有新160epoch/35520step/nativeCSV和完整stdout已解析，RX1既有40epoch/8880step证明复用。原probe_dom_acc固定NaN空值不属于身份梯度/决策失败，未另训probe。完整逐项proof见final_execution；E20预算耗尽不能证明收敛。

旧ordinary仍含PA、AWGN、多径等，仅整体配方进行确认，不将单项操作或TX级效果预先称为身份保持；DAC/class_signature仍关闭。逐TX/day负差保留在source_strata.csv与五fold图，不用平均数抹去类别损失，也不按heldRX/分数改强度或选择重跑。五fold共享大量训练样本，只有一个model seed，不能声称跨seed复现。

成本：新8×20=160model-epochs/35520updates/实际19279824真实网络前向；复用RX1旧pair40model-epochs/8880updates/4819956前向，累计24099780。每行计入L、studentU、ownEMA、seenV与smoke2、heldV；详见measured_costs.csv。新矩阵墙钟1096.709s，旧RX1矩阵883.864s另列。新本地2048个合成网络前向/256增强包另计，旧RX1本地1536/384另列；CPU离线采集/绘图未统一计时，不伪造墙钟。

复现：在预登记immutable CWD中，以普通N607 Python执行 experiments/cvs_ordinary_confirm20/runner.py，或--row加对应登记row。已完成root不得重复提交。离线收尾使用本工作树：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_ordinary_confirm20.harvest --output E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final --first-output E:/type10-7/local_artifacts/cvs_ordinary_fold20_20261016/final --finish。

未完成：跨model seed、full-source匹配Ag/As组合和原clean+6practical正式truth-last。当前源结果不回答哪种urban退化得到改善，不宣称lowurban>60%或clean/星地双突破。全局FAIL结束此普通增强作为Ag的候选，不换RX/强度/seed救援；PASS仅提供源证据，后续组合需独立固定方案。整体goal保持active。

实际输出：

- [五fold准确率及TX/day分层](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/fivefold_source_results.png)
- [RX3完整训练曲线](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/training_RX3.png)
- [RX4完整训练曲线](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/training_RX4.png)
- [RX6完整训练曲线](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/training_RX6.png)
- [RX8完整训练曲线](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/training_RX8.png)
- [逐fold数值](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/fold_results.csv)
- [逐TX/day数值](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/source_strata.csv)
- [逐模型成本](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/measured_costs.csv)
- [全部新模型训练曲线数值](E:/type10-7/local_artifacts/cvs_ordinary_confirm20_20261016/final/training_curves.csv)
