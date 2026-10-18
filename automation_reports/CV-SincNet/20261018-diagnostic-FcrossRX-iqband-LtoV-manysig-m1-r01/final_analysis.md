# I/Q 频率相关增益差：源统计反证结果

本次技术执行 **VERIFIED**；固定科学假设 **FAIL_CLOSED**。完整源统计不支持本次有界二次频率形状的共享 RX I/Q 增益差模型，不进入增强 IQ 生成或 E30 网络训练，不扩大拟合范围或换基函数挽救。

## 范围与方法

run：`20261018-diagnostic-FcrossRX-iqband-LtoV-manysig-m1-r01`。代码 commit：`0f52826b50ce0ef55ba95971ff4f29da0e689643`。原不可覆盖源输入为 P50 的 observations.npz，仅访问既有 L6300/V27000 派生 periodogram 与原可见元数据。保留全部 RX1/3/4/6/8、day1/2/3、6TX、split392005，90 格各 L70/V300；没有留出训练 RX、读取 U TX、合并 V 或访问目标 IQ/truth。

原表示为 EQ1、IQ256、25 MHz、RMS center=false。新统计为每单元 10log10(mean PI/mean PQ)，再去除频率均值；保持原 127 个正 rFFT 点。固定形状 b(k)=(k/127)^2−mean((k/127)^2)，仅拟合相对 RX1 的一个幅度系数，网格 −1 至 +1 dB、101 点。该范围仅用于本次统计模型，未证明是安全增强幅度。

18 个 held TX×日期检验均只使用其他 5TX×其他 2day 的 L 拟合，全部源 RX 保留；V 完全只作评价。另比较零 RX 差异、任意加性 RX 曲线及同 TX 其余日期曲线。没有分类器、身份头或部署模块拟合。

此候选与已关闭静态 RXMIX 不同：先消去了常数 I/Q 增益；与共同复 FIR 不同：统计的是两路相对增益；与 P50 相对时延不同：本次只检验幅度比而非交叉谱相位。物理可能性的文献依据仅为 [IEEE TSP 的频率相关 I/Q 失衡研究](https://ieeexplore.ieee.org/abstract/document/9314287)，不据此宣称当前源数据已识别出 RX 硬件效应或可逆未知 TX 效应。

## 8 项预登记条件的实际结果

| 固定条件 | 结果 | 判定 |
|---|---|---|
| V RX 差异非零 | identity SSE 127809.736234 | PASS |
| L 双半样本 RX 对比功率均值≥0.01 dB²、各 RX>0 | 五 RX 依次 12.162527、10.925620、10.758473、8.923955、12.481448 dB² | PASS |
| shared≤0.8identity | 比值 0.998664，仅降低 0.1336% | FAIL |
| shared≤0.9additive | 比值 1.186259，增加 18.6259% | FAIL |
| 至少 15/18 检验优于 additive | 6/18 | FAIL |
| 每个 RX 不劣于 additive | 五 RX 全部更差 | FAIL |
| shared≤1.05TXconditional | 比值 4.521849 | FAIL |
| 非参考参数无边界解 | 66/72；拟合系数 +0.02 至 +1.00 dB | FAIL |

合计 **2 PASS、6 FAIL**。总 SSE：shared 127638.982300、identity 127809.736234、additive 107597.913609、TXconditional 28227.166739。SSE 是该源统计的平方风险，绝不是分类准确率。边界命中不授权扩大幅度；局部 6 个改善检验不授权挑 TX、日期或 RX 重新晋级。

结论只否定这个固定低维模型。不能据此断言所有频率相关 I/Q 不平衡不存在，也不能把强烈的源统计差异直接解释成 RX 专属成分。当前没有证据支持它作为身份保持 A_g，未启动真实 IQ 变换或训练。

## 验证、运行与成本

本地及启动首步 7 项测试均 PASS，远端 0.255 s。完整规模合成加载/独立复算通过，16916400 个派生实值、72 解一致；本地合成绘图被替代，真实源运行使用已安装 Matplotlib。8 个 Python 文件 AST、2 个生成脚本 AST、10 成员模拟归档检查通过，单次独立 P0/P1 审查 PASS。

发布经一次预检、单包 SHA 比较及远端 compile PASS。run 输出根保持不可覆盖。进程 PID 1074277、start ticks 153575946 的 dispatch 与 active 记录一致；active 保存 CWD、模块文件 argv 和 CUDA 隐藏。首次独立读取时该短进程已自然退出，故没有存活期 /proc cmdline 快照；日志已增长至 stdout810/stderr105字节，完成文件、真实统计及独立图表提供执行后证据。没有 failure.json，没有重启、终止或修改其他任务。

复用 33300 条不同源记录的 16916400 个派生实值；主程序和独立复算各读一次这些派生统计。新增直接 IQ 读取、checkpoint 读取、神经网络前向、训练更新、教师前向、分类头拟合、新 IQ 视图和目标访问均为 0，未消耗 E30 模型训练预算。源统计/拟合墙钟 0.930987 s，启动测试后完整墙钟 3.296480 s，其中独立复算 2.145694 s。

独立复算重新从原 periodogram 生成全部单元与半样本统计，最大单元差 1.065814e−14；72 个网格解索引完全一致，所有 SSE、逐检验与逐 RX 风险及 8 判据一致。实际三联图 PNG/PDF 已生成并目视检查，清楚展示风险、边界拟合和全部 18 个检验，不标注任何神经网络准确率。

## 当前目标与未完成项

本轮没有新默认头识别成绩，无法报告本候选的 urban 改善或 clean 代价。保留 clean 监督份额的 As 三种子结果沿用原冻结报告，不重测、不回流调参：low urban 均值 50.7698%（相对同预算无 As +17.2849 个百分点）；urban 三场景均值 +22.3079 个百分点；clean 均值 −0.3210 个百分点，第三 seed −0.9030 个百分点。As 的星地收益重复，clean 保持未稳定，不能归为新 A_g 收益。

新增 A_g 独立身份收益、Ag×As 交互、clean 净提升、low urban>60% 及 suburban 各场景接近 clean 仍未完成。本轮按原规则关闭固定候选，不把统计可重复或执行正确替代身份性能；完整研究目标保持活动。

## 产物与代码

- [逐 RX SSE](E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018/retrieved/independent_analysis/per_RX_SSE.csv)、[18 个检验与系数](E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018/retrieved/independent_analysis/held_TX_day_folds.csv)
- [真实图 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018/retrieved/independent_analysis/source_iqband_effects.png)、[真实图 PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018/retrieved/independent_analysis/source_iqband_effects.pdf)
- [主结果](E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018/retrieved/result.json)、[独立复算](E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018/retrieved/independent_analysis/result.json)、[完成证据](E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018/retrieved/completion.json)、[实际配置](E:/type10-7/local_artifacts/cvs_FcrossRX_iqband_source_20261018/retrieved/resolved_config.json)
- [运行程序](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqband_source51/run.py)、[统计模型](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqband_source51/core.py)、[独立复算程序](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqband_source51/recount.py)、[测试](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqband_source51/checks.py)、[发布程序](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqband_source51/publish.py)
- [上一轮三种子正式结果](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02/final_analysis.md)

原派生输入与私有拟合 NPZ 均保留在原 run；不提交数据、运行状态、日志、权重或生成图表。公开报告、登记及实际代码随 Git 交付。
