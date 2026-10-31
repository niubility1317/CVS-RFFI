# MILKR 类间/类内散度描述符加权

预登记，未启动。唯一矩阵/seed/命令/环境/owner/路径/stop rule见experiment.json。完整Tri-SIG-TIME SSL及DURA冻结，F/G训练与更新0。

## 方法及前置源选择

R11不中心化descriptor。对当前support中三112维块分别估计类均值的方差B及组内残差方差W（除n-C），取r=B/(W+view_mean(W)+eps64)，r再除view_mean(r)，无类间证据的view取均匀r=1。坐标权重为(1-eta)+eta*r，乘sqrt(weight)后逐view单位化。eta固定网格0/0.25/0.5/0.75，原五块保持R9。K1无物理类内估计，严格恢复R9；不伪造额外K。

只复用R9的source-only K1核规则。源L五RX×两episode，四K等权选择一个eta，source held至少+0.5pp且每K不低超过0.5pp才进目标；否则SOURCE_ONLY_SCIENTIFIC_STOP、target0。源六类已见，不能替代新类证据；权重依完整support标签，条件LOO不是无偏nestedCV。

## 条件目标

源PASS后P03/MIX01、seed2026092701，七RX×K1/5/10/20×N2/5/10/20、全补集query、四严格S/Q。每个worker共享同一次F/G特征提取，拟合R11与eta0的R9C控制，各自独立系数、预测文件和arm指标键，共1120行。两臂标签额度相同，不把特征共享当部署免费成本。原MILKR同split固定预测复用。所有预测完成才独立truth-last评分。

主R11 S0Q0平均宏H至少+5pp才满足主目标；仅全部16格均+5pp才称各K/N均达。另给R11−R9C同池配对效应与逐RX负差。P03已有曝光，不声称盲测。源选择或目标适应均不读取query真实组别/配额/统计。

两个定向测试及一次独立P0/P1审查PASS：权重公式、eta0/K1精确R9、统计冻结/查询分批独立。真实checkpoint无query smoke核对实际336及F原输出不变。GPU0/2/4/6每卡最多2新增worker，启动核实显存、不触碰其他owner。不可覆盖run root，科学低收益不算技术失败，技术失败保留且不原地盲重启。

## 已核实源端结果

**SOURCE_ONLY_SCIENTIFIC_STOP / VERIFIED**。四强度全部完成，eta0原R9胜出，未通过预登记+0.5pp门槛，目标IQ/预测/评分0。源目标分别{"0.0": 0.7228550536894782, "0.25": 0.7136107691347389, "0.5": 0.7015141791456654, "0.75": 0.6980128522905621}。源数据处理用时17.577秒。

真实checkpoint无query smoke PASS，归档比较与远端compile PASS，实际commit `cbb91b43da036c03b4aa5bd244f399c6d6a8f312`；首次读取时PID1412355已退出，完整source_selection、四模式日志及正常完成artifact证明执行结束，不声称曾观察到存活进程。没有技术失败、未降低原门槛。

该源代理六类均被F见过，负结果限全局强度选择，不能证明当前目标support真实物理留出选择也无效。原候选保留，不据query调强度或补跑目标。见[source_summary.json](source_summary.json)。
