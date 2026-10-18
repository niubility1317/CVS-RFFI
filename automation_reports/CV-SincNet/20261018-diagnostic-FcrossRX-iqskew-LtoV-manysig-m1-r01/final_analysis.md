# I/Q 相对时延源域检验结果

本次运行技术执行 **VERIFIED**，科学假设 **未通过**。固定的 RX 共享 I/Q 相对时延相位模型被关闭；不进入网络训练，不调整频段、拟合范围、参考 RX、划分或 seed 挽救。

## 数据与方法

- run：`20261018-diagnostic-FcrossRX-iqskew-LtoV-manysig-m1-r01`；代码 commit：`6c3cd4c3856df0b8aa64ac6a3eda233ad3070bc4`。
- 仅复用既有合法源数据：L6300、V27000，6 个 TX、5 个 RX（1、3、4、6、8）、3 个日期；每个 TX×RX×日期单元 L70/V300。未留出源 RX，未读取 U 的 TX 真值，未访问目标测试。
- 固定 IQ256、25 MHz、原 RMS 归一化、center=false、eq1。没有生成增强 IQ，没有训练或加载身份网络、教师或新分类头。
- 在全部 127 个正频率点统计 I/Q 归一化交叉谱。每个 held TX×日期检验均仅用其余 5 个 TX、其余 2 个日期的 L 数据拟合；参考 RX1 的 V 可观测统计仅作为固定预测关系的输入。此检验不是部署预测，也没有把不同 RX 的独立样本当作逐包配对。
- 18 个 TX×日期检验、72 个非参考 RX 参数；相对时延拟合网格固定为 −0.5 至 +0.5 样本。L 的不重叠奇偶半样本用来检验统计量是否可复现。该有限窗口关系是必要统计假设，不是实际硬件时延的识别结果。

## 固定阈值的实际结果

| 条件 | 实际结果 | 判定 |
|---|---|---|
| V 存在 RX 差异 | identity SSE 2302.146968 | PASS |
| L 交叉半样本相干功率 | 五 RX 为 0.855493、0.374368、0.746249、0.757942、0.415001 | PASS |
| 共享相位模型比 identity 降低至少 20% SSE | shared/identity=0.999474，仅降低 0.0526% | FAIL |
| 比通用 RX 加性差异参照降低至少 10% SSE | shared/additive=1.283629，增加 28.3629% | FAIL |
| 至少 15/18 检验优于加性参照 | 0/18 | FAIL |
| 每个 RX 均不劣于加性参照 | 5 个 RX 全部更差 | FAIL |
| 不超过同 TX 条件参照风险的 1.05 倍 | shared/TXconditional=3.989538 | FAIL |
| 无参数落在拟合边界 | 0/72；拟合数值范围 −0.04 至 +0.07 样本 | PASS |

合计 3 PASS、5 FAIL。源 V 总 SSE：共享模型 2300.935733，identity 2302.146968，加性 RX 参照 1792.523951，同 TX 条件参照 576.742438。各 RX 和全部 18 个检验的原始统计分别保存在 CSV。不能把失败解释为“只需换频段或加大扰动”；这个固定模型没有得到所需的跨 TX、日期证据。

这些结果也不证明真实数据中没有频率相关 I/Q 失衡。它们只否定本次明确限定的共享相对时延相位规律；没有验证有限 IQ 窗口上的身份保持变换，更没有证明任何识别收益。

## 实际执行与独立验证

启动首先完成 9 项协议负例与数学单元测试，PASS（远端 1.970 s）。本地完整规模合成复算 PASS（16916400 个实值；本地仅绘图被替代，真实远端绘图使用实际 Matplotlib）。单次独立 P0/P1 审查 PASS。单包发布完成一次本地/远端 SHA 比较，远端编译 PASS。

启动后独立核对 PID 1069080、start ticks、CWD、cmdline 和 CUDA 隐藏状态，并观察到日志增长。最终 PID 已退出，completion 为 SOURCE_ANALYZED，无 failure.json。技术完成与科学失败分别记录。

实际使用 33300 个源 IQ 包，生成 16916400 个交叉谱充分统计实值。神经网络前向、训练更新、教师前向、分类头拟合、新 IQ 视图及目标测试访问均为 0。源读取/统计/拟合墙钟 3.241623 s；启动测试后完整运行墙钟 11.043917 s，包含独立复算 2.245272 s。没有消耗 E30 模型训练预算。

独立复算不重新读取 IQ、不加载模型；从保存的派生统计重建全部单元、72 个网格解和风险。全部 72 个解索引一致，单元交叉谱最大差为 0，8 项判据一致。真实三联图 PNG/PDF 已生成并目视检查：文字清楚，展示五 RX 风险、全部拟合值和 18 个检验风险，未标注神经网络准确率。

## 与当前联合目标的关系

本次没有新增模型或正式测试成绩。既有同预算、保留 clean 监督份额的 As 对照仍为后续训练组织的参照。上一轮完整三种子结果：urban 三场景均值增加 22.3079 个百分点，low urban 增加 17.2849 个百分点至 50.7698%，clean 平均下降 0.3210 个百分点；逐 seed clean 差为 +0.0935、−0.1536、−0.9030 个百分点。As 的 urban 收益可跨这 3 个 seed 复现，clean 保持尚不稳定；这些不是本次 I/Q 假设带来的收益。

当前不能声称 A_g 已增强跨 RX 身份提取，不能声称 clean 与星地双提升，low urban 超过 60% 的目标仍未达到。本次按预登记关闭失败路线，保留全部数据角色与原模型，不新增训练损失或诊断头。

## 可查产物

- [逐 RX SSE](E:/type10-7/local_artifacts/cvs_FcrossRX_iqskew_source_20261018/retrieved/independent_analysis/per_RX_SSE.csv)
- [全部 18 个 TX×日期检验](E:/type10-7/local_artifacts/cvs_FcrossRX_iqskew_source_20261018/retrieved/independent_analysis/held_TX_day_folds.csv)
- [真实三联图 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_iqskew_source_20261018/retrieved/independent_analysis/source_iqskew_effects.png)
- [真实三联图 PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_iqskew_source_20261018/retrieved/independent_analysis/source_iqskew_effects.pdf)
- [主结果](E:/type10-7/local_artifacts/cvs_FcrossRX_iqskew_source_20261018/retrieved/result.json)、[独立复算](E:/type10-7/local_artifacts/cvs_FcrossRX_iqskew_source_20261018/retrieved/independent_analysis/result.json)、[完成证据](E:/type10-7/local_artifacts/cvs_FcrossRX_iqskew_source_20261018/retrieved/completion.json)、[实际配置](E:/type10-7/local_artifacts/cvs_FcrossRX_iqskew_source_20261018/retrieved/resolved_config.json)
- [实现](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqskew_source50/run.py)、[统计模型](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqskew_source50/core.py)、[独立复算程序](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqskew_source50/recount.py)、[发布程序](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_iqskew_source50/publish.py)
- [上一轮三种子正式结果](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02/final_analysis.md)

私有派生 observations.npz 保留在不可覆盖的原 run，未提交 Git。真实 IQ、checkpoint、日志、运行状态和生成图表均未提交 Git；公开登记和分析报告随交付提交。
