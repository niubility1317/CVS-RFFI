# 源相位重复性结果

源规则：**FAIL**；真实运行与产物：VERIFIED。实际代码 `1a96162bda2026419cbadd52dc4bb475d38ba646`。

|窗口|risk gain|正收益折/18|最低cell坐标可用数/70|最小拟合resultant|结论|
|---|---:|---:|---:|---:|---|
|w0_80|-0.01994330|3|0|0.000000|FAIL|
|w80_160|0.00054227|6|0|0.000000|FAIL|
|w160_240|-0.05280389|2|0|0.000000|FAIL|

主窗口规则逐项：

- coverage：False
- risk_gain：False
- fold_repeatability：False
- all_RX_nonnegative：False
- wrong_RX_separation：True
- fitted_direction_concentration：False
- cycle_repeatability：False

全量6300源L、18900窗口、75600周期；0模型/前向/更新/checkpoint/目标预测。诊断主体计时 0.739 秒（含缓存读取，不含Python/Torch导入，完整进程墙钟未单独计时）。

这里的risk gain是源RX相位统计的留出误差差，不是准确率或百分点。周期重复比不是总SNR；RX关联不能区分硬件、传播、均衡或采集条件。相位共轭、CFO混叠和模板时移规范仍未唯一识别。

冻结规则对空cell坐标以0占位，并独立判coverage失败；因此最低resultant=0和未加权cycle均值含不可估计项，不能把它们直接解释为实测相位混乱。图中的周期一致性另按可用数加权，只作描述，不改主决定。可用性不足也不证明不存在任何RX相位因素。

失败后不换窗口、降阈值或调幅启动训练。通过也仅支持另做有限IQ/defaulthead源安全验证，不能跳过身份保持和同预算训练对照。当前没有任何新clean/urban性能或跨seed结果。

图：E:/type10-7/local_artifacts/cvs_source_phase_20261016/final/phase_source_repeatability.png；逐cell表：E:/type10-7/local_artifacts/cvs_source_phase_20261016/final/source_cells.csv。


实际发布与执行：VERIFIED。单归档传输对照、普通账户/主机、远端compile与独立PID/CWD/argv读回通过；启动快照log0字节，终端日志及完整三窗口结果随后落地。实际统计代码commit1a96162bda2026419cbadd52dc4bb475d38ba646；发布后没有修改远端实现或规则。后续本地harvest只修正图中可用样本加权描述，不改变固定空cell占位、阈值或FAIL决定。

主结果仅0.00054227圆周risk改善，6/18折正向，RX1/3为负；部分cell坐标可用数0，未建立跨TX/day稳定RX相位方向。可用频点自身的奇偶周期一致性很高（各RX约.987至.999），因此不能将失败统一归因为相位噪声。覆盖不足、可用频点几乎共用相位形态、RX关联不能跨折稳定重现共同限制该候选；不证明真实RX不存在相位效应。

该相位增强假设按预定规则结束。不通过换主窗口、提高扰动、删除RX或另训诊断头挽救。统计没有训练任何识别网络，不能据此报告新的默认头准确率、clean代价或跨seed提升。整体clean正收益＋low urban>60%＋suburban保持目标仍未完成。

复现命令（在本run immutable release CWD，使用预登记普通N607 Python）：`python -u -m experiments.cvs_source_phase.run`。不可向已经完成的run根重发。离线采集命令：`F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_source_phase.harvest --output E:/type10-7/local_artifacts/cvs_source_phase_20261016/final`。

本地实际验证：同一6336包合成检查执行2次，独立审查另执行1次，总19008个合成IQ检查包，网络前向0；包含4协议负例。原模型/正式测试IQ/其他任务均未修改。
