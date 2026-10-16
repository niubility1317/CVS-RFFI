# F＋跨 RX：周期差异残差相关形状源检验

状态：ANALYZED／VERIFIED，固定源依据检验失败并关闭。基线为 F＋跨 RX 同 TX 对比学习，后续模型预算 E20；当前不构造模型、不修改 IQ、不训练。

唯一矩阵与固定配置见 [experiment.json](experiment.json)。Run 为 20261017-diagnostic-F-crossRX-cyclecov-fitL-manysig-m1-r01，owner 为 codex/root/F-cycle-covariance-source-20261017。

固定源统计问题：IQ[20:180] 的八个20点周期，经分析专用 lag20 公共相位抵消及周期均值移除，其周期内部 lag1至4 归一化复相关形状是否有跨 TX/day 的 RX 关联。残差不预设为纯接收机热噪声，也可能含 TX 随机误差、传播、均衡和重采样影响。已核查的原文与推导见 [研究说明](../../../analysis/cvs_cycle_residual_covariance_basis_20261017.md)。区别于另一对话 practical 正负噪声采样方向。

输入仅复用原 VALIDATED_ONCE 源载体的 fit L5040，RX3/4/6/8、day1/2/3、6TX、每格70包，split392005。原缓存反序列化 L6300、未使用 V 和历史特征；RX1原L1260在转换统计数组前排除，U仅角色ID元数据，没有U IQ/TX标签。U/V/heldRX1/target均不参与统计，不消费父权重或缓存特征；不是完整L6300或内部L_train5580训练结果。

18折同时留出一个TX和一个day，其余5TX×2day给每RX估计均值，训练和留出分别去除四RX公共均值；以零RX对比为MSE参照，另设三个固定错误RX循环指派。已知源RX条件统计，不拟合身份或接收机分类器。全部5040包、72格、18折参与，没有置信筛选或删包。

原固定条件必须全部通过：每格lag20 coherence中位数≥.90、q10≥.80；RX形状对比基准RMS≥.02；合并风险改善≥.20；至少15/18折改善；每个RX风险改善≥0；正确RX风险改善超过最佳错误指派≥.02。风险改善不是身份准确率。任一失败即关闭固定窗口、周期、lag、阈值、RX、seed依据问题，不改条件救跑；PASS也只支持源统计依据，不自动训练或证明Ag。

本地22个针对性非法配置/输入及源条件反例检查PASS，包含TX-only、day反转、低周期一致性、弱RX差异失败。两次本地检查累计296个合成IQ观察、0真实IQ、0NN/更新。一次独立P0/P1审查PASS，见review.json。无模型/CP，真实CP无查询smoke不适用；无重复数据验证、额外审批或签名。

发布入口：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_cycle_covariance17.publish，CWD为当前受管codex/cvs-iq-timing-e20-20261016工作树。远端普通N607既有CVS-RFFI Python，命令 CVS-RFFI/bin/python -u -m experiments.cvs_cycle_covariance17.run；CWD PROJECT/releases/cvs_F_cycle_covariance_fitL_20261017_r01。实际Python绝对路径、机器私有信息仅在私有runtime与submit中保存。代码标识是本次实际Git commit，由release_commit.txt和resolved_config记录，不新增报告/配置哈希。

CUDA_VISIBLE_DEVICES为空，CPU/OMP/MKL/OpenBLAS各2线程，GPU none。不可变输出 PROJECT/runs/20261017-diagnostic-F-crossRX-cyclecov-fitL-manysig-m1-r01；独立发布根 PROJECT/releases/cvs_F_cycle_covariance_fitL_20261017_r01；日志source_cycle_covariance.stdout.log。一次普通账户/存在性/空间预检，一份release archive单次传输比对，一次远端编译与启动后PID/CWD/argv/CUDA/日志或终态读回。0模型、0CP、0NN、0更新、0教师、0另训头。

角色不匹配、缺缓存、非有限值、错矩阵或异常时保存failure和原产物停止，无自动重试，不干预健康旧任务。科学FAIL不等于技术失败。

预期产物：active/resolved_config/completion、全部72格/18折与wrongRX的cycle_covariance.json、私有5040逐包shape/coherence/residualratio/metadata、fit物理ID、stdout、独立全量复算、同row报告和登记。只取回统计与日志，不取回IQ或权重。生成输出及私有机器状态不进入Git。

尚缺有效新Ag、独立As稳定收益、同契约B00/B10/B01/B11与交互项、跨seed和正式clean＋六practical。low urban>60%与suburban各场景距自身clean≤2pp探索目标均未完成。本检验不能替代真实身份识别收益。


## 实际源结果

执行 commit 0b0db6d038ac44668cd9dac30e96d8bad6262dc5，已独立核对远端代码OID。一次普通账户预检、单份归档传输比较、远端编译及初始PID/CWD/argv/CUDA检查完成。worker 0.623498秒自然结束，完整stdout2行、无failure；结束早于独立日志增长采样，采用实际completion和完整日志终态核实，不重跑补采样。

| 固定指标 | 实际 | 判定 |
|---|---:|---|
| 72格周期相关中位数≥.90 | 50/72格未达，范围0.501817至0.945911 | FAIL |
| 72格周期相关q10≥.80 | 43/72格未达，范围0.294854至0.942537 | FAIL |
| RX形状对比基准RMS≥.02 | 0.203119 | PASS |
| 合并风险改善≥20% | 35.211296% | PASS |
| 改善折数≥15/18 | 17/18 | PASS |
| 每RX风险改善≥0 | RX6 −17.668361% | FAIL |
| 正确RX较最佳错RX改善≥2风险百分点 | 91.190197 | PASS |

| 源RX | 周期中位数未达格数 | q10未达格数 | 留TX/day风险改善 |
|---|---:|---:|---:|
| 3 | 18/18 | 18/18 | 52.041331% |
| 4 | 5/18 | 4/18 | 49.480090% |
| 6 | 9/18 | 3/18 | -17.668361% |
| 8 | 18/18 | 18/18 | 43.309050% |

风险改善是源条件形状的MSE变化，不是身份准确率。正确RX对比的基准MSE为0.0412574125、实际MSE为0.0267301427；三个错RX风险改善为−55.978900%、−130.839768%、−55.978900%，这些风险比不限制在0至100%。合并值通过不能覆盖周期解释与RX6失败。

独立reader未导入core.analyze，使用标量/中位数/线性分位数及显式RX指派，从全部5040条已保存观察重建72单元和18折，核对全部八维单元形状、coherence分位、残差功率比、每折风险、每RX及全部错RX控制，逐项与运行产物相符。它没有再次读取原始IQ或重做观察器；原观察器由已提交实现、本地解析不变性与负例覆盖。全部物理ID仅私有检查唯一数5040，不在Git保存。

结论：这个固定周期残差不能作为可迁移RX观测增强的充分依据。结果保留了部分RX关联，但周期一致性不满足假设，且RX6跨TX/day预测误差增加；不据此断言已分离TX/RX或估计纯热噪声，不把程序PASS/源关联代替身份收益。固定窗口20:180、cycle20、lag1至4和原条件关闭，不更改窗口、lag、阈值、RX、seed或剔除低coherence包救跑，不启动colored-noise Ag。

## 成本与完整产物

实际0模型、0checkpoint、0NN/更新/教师/另训头、0target；仅5040个fitL IQ观察，源缓存反序列化L6300和未使用V/历史特征，U只有角色ID、没有IQ或TX标签。原L6300中RX1的1260包在统计数组转换前排除。0新增训练epoch；后续预算仍为scratchE20和F＋跨RX同TX默认头联合训练。工程两次局部检查共296合成IQ观察，独立复算和绘图0IQ/NN/更新。

[全量聚合复算](final_analysis.json)、[执行摘要](execution_summary.json)与原始预登记保留。私有逐包观察、物理ID及完整日志留在 local_artifacts/F_cycle_residual_covariance_20261017/final，不取回IQ/权重。图已生成并查看：[周期源条件全量图](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_cycle_residual_covariance_20261017/final/figures/cycle_covariance_source.png)，同目录保留PDF。

本地交付工具曾出现镜像缺少路径、导入根缺失、encoding拼写和跨平台路径表达错误，均按实际失败层更正；真实worker没有失败、重启或热改。publisher预检run路径的同一Windows反斜杠问题已由拦截真实发布入口、0远端动作的测试复现并修正；一次限定该行的P0/P1定向复审PASS，见path_fix_review.json。修复不改变本次已执行commit或统计结果，没有再次发布实验。

## 研究状态

本轮最小检验已完整回答：固定短周期残差RX依据未通过。没有产生新的IQ算子或A_g识别收益；不回退到同身份插值、已关闭MC2或其他旧失败配方。下一项需先取得独立的身份保持观测依据，再冻结一个可证伪的有界IQ算子；目前未选择幅度、登记或启动后继候选。联合任务继续ACTIVE：有效Ag、As跨seed稳定、同完整契约四格与交互项、正式clean＋六practical、lowurban>60%和suburban接近自身clean均仍缺证据。
