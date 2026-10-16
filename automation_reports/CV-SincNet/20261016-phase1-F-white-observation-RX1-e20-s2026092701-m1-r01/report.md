# F/E20：有界接收观测噪声源RX1结果

源资格 **FAIL**，0目标访问；一个新scratch20，A0结果复用不载权重。

|模型|heldRX1准确率|MacroF1|seenV均值|seenV最差RX|
|---|---:|---:|---:|---:|
|A0|81.148148%|80.603394%|98.041667%|95.537037%|
|W30|81.166667%|80.429806%|98.074074%|95.648148%|

W30−A0 accuracy +0.018519pp，macroF1 -0.173588pp；修复80/破坏79；seenV损失-0.032407pp。

新20epoch/4440step/nativeCSV/fullstdout与复用A0日志完整解析，全部5994个L/U批次逐条相同。实际新增噪声比29.999998至30.000002dB，wholeordinary/As调用0。

新20模型轮/4440updates/3546618真实前向，墙钟1010.793s；复用控制20模型轮/4440updates/3546618前向另列。本地合成检查另计。

一个已使用过的源RX折、一个modelseed，不能替代完整DG/正式clean/practical/跨seed。FAIL关闭WHITE30且不扫SNR/概率或换RX/seed救援；PASS只支持另登记固定其他4RX，不直接进target或As。

固定能量Gaussian方向不等于精确AWGN；30dB只是新增噪声比，不是原录音总SNR。所有有标签视图参与CE，零功率包保留；默认头和F按原联合方式训练。完整曲线/逐TX/day正负结果保存，预算耗尽不等于收敛。


实际运行与科学状态：**VERIFIED / FAIL_SOURCE_RX1_ONLY**。实际训练代码b30b94ce3d448d8746f7e3b41a647b766828bbf6。worker和dispatcher自然退出，无failure；单archive transfer/remotecompile、真实own scratchCP no-query smoke及PID/CWD/argv/GPU/log增长均独立保存。原release未热改，不触碰其他健康任务。

本轮F三路SIG_TIME/CENTER_25K/CENTER_UNITS、900722参数、25MHz/IQ256/equalized1/原nativeRMS/NoExplicitCFO/joint160/defaultcos全部保持，默认头和骨干正常联合训练，未拟合诊断头。新算子仅训练IQ端的固定能量噪声，.5RD(raw)+.5RD(G)，四L前向，与匹配A0一致；每RD复用同IQ，0跨视图一致性。原U/ownEMA/旧门控/strong_noise_std不变，U TX始终隐藏。没有PA/DAC/IQ失衡/CFO/multipath/shift/classsignature硬件重写，0wholeordinary/As。

有效角色fitRX3/4/6/8的L5040/U45360/seenV21600，heldRX1 L/U不使用，原heldV5400仅FINAL。全部6TX×4RX×3day训练覆盖和实际5994L/U批次与原A0逐条一致。A0来自同F/seed/split/fourforward/E20的已完成source控制，只复用其metrics/trace，不载任何旧CP/EMA/optimizer/统计。包装resolved顶层33/17为遗留描述，实际nativeargs13/7与全20epoch/4440成功更新证明总预算E20。

新成本20模型轮/4440updates/3546618真实前向，矩阵墙钟1010.793286s；复用控制20模型轮/4440updates/3546618真实前向另列，0父权重成本。完整L/studentU/sourceeval(含smoke2)/EMA/heldV分项保存在analysis_summary.json。本地累计1024合成网络前向/1536噪声输入包/2合成optimizer更新另列，0realIQ/CP；唯一限定P0/P1与两次各11负例通过。CPU分析/绘图未统一计时，E20预算耗尽不等于收敛。

新增噪声方向逐step/view刷新、独立于标签/RX身份；按实际波形能量精确0.1%生成，不用白输入功率代理。30dB是新增噪声比，原录音总SNR未知；原网络RMS在噪声之后按原规则执行，未修改practical received IQ或测试SNR。归一化Gaussian方向是固定能量各向同性噪声，不能称为精确Gaussian AWGN。零功率包不丢弃，不加噪，仍参与CE。

完整20epoch/4440step/nativeCSV/fullstdout和复用A0日志已解析，全部关键loss/gradient有限、无跳步或U真值。F每行4440个物理clip/单位映射按step核对，默认W/gain、三路投影和q有实际运动；机制证据不替代识别收益。原未运行probe的空值仍为占位，未拟合新probe。

|源分层|A0准确率|W30准确率|W30−A0/pp|
|---|---:|---:|---:|
|TX0|96.2222%|96.2222%|+0.0000|
|TX1|86.0000%|85.0000%|-1.0000|
|TX2|56.7778%|52.0000%|-4.7778|
|TX3|65.5556%|65.7778%|+0.2222|
|TX4|100.0000%|100.0000%|+0.0000|
|TX5|82.3333%|88.0000%|+5.6667|
|day1|80.0556%|81.4444%|+1.3889|
|day2|88.2222%|86.1667%|-2.0556|
|day3|75.1667%|75.8889%|+0.7222|

这是一个已使用过的sourcefold和一个modelseed。F由用户在暴露A–F历史基准后指定，不声称新独立盲测；固定FINAL20，新目标分数不参与调参、选模或晋级。FAIL关闭WHITE30，不扫描SNR/概率、换RX/seed救援；PASS仅允许另登记同算子的固定其余四RX，不直接进target或混合As。旧raw+wholeordinary近阈值失败保持关闭，不改用其A1主行。

正式clean净收益、low urban>60%、suburban保持和跨seed均未完成。当前没有新practical成绩，不由源收益推断缓解了某类urban损伤；整体goal保持active。

复现训练入口见experiment.json：immutable CWD中运行experiments/cvs_white_observation20/runner.py；已完成root不可重复启动。离线复算：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_white_observation20.harvest --output E:/type10-7/local_artifacts/cvs_F_white_observation20_20261016/final --state E:/type10-7/local_artifacts/cvs_F_white_observation20_20261016/terminal_readback.json --control-root E:/type10-7/local_artifacts/cvs_F_raw_anchor20_20261016/final/A0-RX1-s2026092701/source --cached。离线不读取IQ/CP或加载native。

实际输出：

- [完整两组训练曲线](E:/type10-7/local_artifacts/cvs_F_white_observation20_20261016/final/complete_training_curves.png)
- [逐TX/day默认头结果](E:/type10-7/local_artifacts/cvs_F_white_observation20_20261016/final/held_source_strata.png)
- [全部分层数值](E:/type10-7/local_artifacts/cvs_F_white_observation20_20261016/final/source_strata.csv)
- [完整训练数值](E:/type10-7/local_artifacts/cvs_F_white_observation20_20261016/final/training_curves.csv)

## 负结果与下一项独立诊断

W30相对A0修复80、破坏79，净改善仅1/5400，不形成预定1pp收益；macroF1下降0.173588pp。TX5+5.6667pp与TX2−4.7778pp大体抵消，TX1−1.0000pp、TX3+0.2222pp、TX0/4持平；day1/2/3分别+1.3889/−2.0556/+0.7222pp。保留这些正负分层，不能由单TX或单day涨分宣称身份泛化增强成功。

完整曲线已查看。W30与A0的平均CE/KL、physical grad_before_clip、源V及U接受量轨迹接近；早期CE/KL下降，后期大体平台且有限，E14按计划启用旧U，cos LR一致。实际新增噪声强度正确、层和默认头确有更新，但这些执行证据未转化为跨RX净识别收益。U没有TX真值，不推断其伪标签准确率；不把E20预算用尽称作充分收敛。

本结论限定于本固定0.1%能量噪声、F/E20、一个已使用过的源RX折及seed2026092701，不否定所有噪声正则。按预定规则关闭WHITE30，不扫描SNR/概率，不扩其余四折/seed，不将其加入联合As。原raw+wholeordinary候选也继续保持关闭。

下一项F/practical源复核已在WHITE30最终成绩产生前固定于d92fd04fbf66d7896ee26194e76b782aa4fe079a，见analysis/cvs_F_practical_source_reaudit_20261016.md。固定A0和W30两个final20默认头，在未参与两模型训练的held源L1260上比较原六practical与总效应/固定接收状态反事实；0optimizer/头拟合/U/V/target。W30即使失败也保留为负诊断，不能由反事实高分重新晋级或修改其强度。该诊断目前只是设计，尚未实现、登记、发布或生成成绩/图；不是本轮遗漏的正式测试，也不是新的审批门槛。
