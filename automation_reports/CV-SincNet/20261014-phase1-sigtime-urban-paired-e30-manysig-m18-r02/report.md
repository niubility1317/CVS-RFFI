# 固定 SIG_TIME 的 Phase1 urban 星地增强：最终 E30 验证

**18/18 row 完成；low urban 60% 目标未达到。** 固定原 CVS/默认头、固定正式 practical IQ，主方案 C3 low urban 为 53.157% ± 0.513%，相对普通拼接 B0 +0.659 个百分点；clean 平均代价 0.108 个百分点。以下包含全部三 seed、负结果、源归因、成本和原预登记。

## 最终三 seed 结果

18/18 模型完成固定 E30、七视图预测和独立 truth-last 评分。所有模型使用同一固定 SIG_TIME 架构与 checkpoint 中的默认分类头。主方案 C3 预先指定，目标分数未用于更改采样、参数、资格或选择性重跑。

| 臂 | urban 均值 (%) | low urban (%) | mid urban (%) | clean (%) | suburban (%) | urban Macro-F1 (%) | 最差 urban RX (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 46.704 ± 2.153 | 37.164 ± 1.806 | 45.713 ± 2.146 | 83.362 ± 1.762 | 57.300 ± 2.740 | 44.684 ± 2.183 | 34.365 ± 0.834 |
| B0 | 64.063 ± 0.478 | 52.499 ± 0.449 | 63.117 ± 0.516 | 81.709 ± 1.198 | 77.619 ± 0.481 | 63.763 ± 0.379 | 57.815 ± 1.593 |
| C1 | 63.997 ± 0.673 | 52.431 ± 0.460 | 63.057 ± 0.636 | 81.446 ± 1.402 | 77.521 ± 0.962 | 63.811 ± 0.778 | 57.715 ± 1.538 |
| C2 | 64.332 ± 0.720 | 53.050 ± 0.566 | 63.497 ± 0.708 | 81.437 ± 1.499 | 77.282 ± 0.939 | 64.153 ± 0.764 | 58.078 ± 1.386 |
| C3 | 64.456 ± 0.654 | 53.157 ± 0.513 | 63.622 ± 0.637 | 81.601 ± 1.457 | 77.416 ± 0.822 | 64.288 ± 0.707 | 58.307 ± 1.404 |
| C4 | 64.518 ± 0.680 | 53.327 ± 0.499 | 63.701 ± 0.657 | 81.612 ± 1.811 | 77.373 ± 0.851 | 64.227 ± 0.831 | 58.351 ± 1.428 |

以上为均值±样本标准差，实际每臂 3 seed。

| 臂/seed | low urban (%) | mid urban (%) | urban 均值 (%) | clean (%) | suburban (%) | urban Macro-F1 (%) | 最差 RX (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0/2026092701 | 36.126 | 44.309 | 45.160 | 84.308 | 55.028 | 43.630 | 33.692 |
| C0/2026092702 | 36.115 | 44.646 | 45.789 | 81.329 | 56.530 | 43.229 | 34.106 |
| C0/2026092703 | 39.249 | 48.183 | 49.163 | 84.450 | 60.343 | 47.194 | 35.297 |
| B0/2026092701 | 52.770 | 63.346 | 64.228 | 81.407 | 77.622 | 63.749 | 59.015 |
| B0/2026092702 | 51.980 | 62.526 | 63.524 | 80.691 | 77.137 | 63.391 | 56.008 |
| B0/2026092703 | 52.746 | 63.479 | 64.435 | 83.029 | 78.099 | 64.149 | 58.422 |
| C1/2026092701 | 52.605 | 62.947 | 63.780 | 80.625 | 76.925 | 63.331 | 58.386 |
| C1/2026092702 | 51.909 | 62.483 | 63.458 | 80.648 | 77.006 | 63.392 | 55.956 |
| C1/2026092703 | 52.779 | 63.740 | 64.751 | 83.065 | 78.630 | 64.709 | 58.804 |
| C2/2026092701 | 53.286 | 63.464 | 64.186 | 80.545 | 76.824 | 63.736 | 58.950 |
| C2/2026092702 | 52.404 | 62.806 | 63.697 | 80.598 | 76.660 | 63.689 | 56.479 |
| C2/2026092703 | 53.461 | 64.221 | 65.114 | 83.167 | 78.361 | 65.035 | 58.804 |
| C3/2026092701 | 53.314 | 63.540 | 64.265 | 80.773 | 76.908 | 63.857 | 59.193 |
| C3/2026092702 | 52.584 | 63.030 | 63.918 | 80.746 | 76.975 | 63.903 | 56.689 |
| C3/2026092703 | 53.574 | 64.296 | 65.184 | 83.283 | 78.363 | 65.104 | 59.040 |
| C4/2026092701 | 53.329 | 63.474 | 64.217 | 80.626 | 76.873 | 63.642 | 59.179 |
| C4/2026092702 | 52.827 | 63.186 | 64.041 | 80.509 | 76.890 | 63.861 | 56.703 |
| C4/2026092703 | 53.825 | 64.441 | 65.296 | 83.702 | 78.355 | 65.179 | 59.172 |

| 配对对比 | urban 差 (百分点) | low 差 | clean 差 | urban/low 正差 seed 数 |
|---|---:|---:|---:|---:|
| B0 − C0 | +17.359 | +15.335 | -1.653 | 3/3，3/3 |
| C1 − B0 | -0.066 | -0.068 | -0.263 | 1/3，1/3 |
| C2 − C1 | +0.336 | +0.619 | -0.009 | 3/3，3/3 |
| C3 − C2 | +0.123 | +0.107 | +0.164 | 3/3，3/3 |
| C4 − C3 | +0.063 | +0.170 | +0.012 | 2/3，3/3 |
| C2 − B0 | +0.270 | +0.552 | -0.272 | 2/3，3/3 |
| C3 − B0 | +0.393 | +0.659 | -0.108 | 3/3，3/3 |
| C4 − B0 | +0.456 | +0.828 | -0.097 | 2/3，3/3 |

主方案 C3 low urban=53.157%，60% 目标未达到。相对 B0，urban +3、low +5、clean 最大下降 0.5 个百分点的探索条件逐项判断为：False、False、True。这些不是选模门槛。
C1 的双视图 CE 在相同 logits 下与 B0 数学等价；不把前向拆分宣称为新的判别方法。课程与间隔的增益以完整配对差报告，包含源状态课程轨迹的交互。没有 KL 对照，也没有证明新的创新或优于 KL。

## 冻结后的源域机制证据

固定 E50 clean 正确集合有 718/720 条，所有 row 使用同一 ID 集合统计保持率。所有验证 draw 相同；另对首 seed 六臂使用同一原诊断 draw 进行 11 条控制条件。无 optimizer 更新、无拟合头、无 U 真值、无 query。源控制效应不等同于目标测试的因果证明。

| 首 seed 源 low urban | 完整 residual (%) | 移除新增噪声固定态 (pp) | 移除新增噪声重算态 | 平坦 H 固定态 | 移除 carrier 固定态 | 移除 RX IQ 固定态 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 45.000 | +6.806 | +7.778 | +3.194 | +0.417 | -0.139 |
| B0 | 63.472 | +9.444 | +9.444 | +1.806 | +0.556 | -0.139 |
| C1 | 63.750 | +9.444 | +9.583 | +2.917 | +0.000 | -0.139 |
| C2 | 63.333 | +11.250 | +11.806 | +3.472 | +1.806 | +0.000 |
| C3 | 63.611 | +11.111 | +11.528 | +4.306 | +1.111 | -0.139 |
| C4 | 63.333 | +10.972 | +10.972 | +3.333 | +0.972 | +0.000 |

物理审计已全量解析 47520 条原源诊断观察。正式 no-EQ 残余核长 62、数值延迟 24，保守有效内部从第 61 点开始，共 195 点；reference 收到窗口 [80,160)，名义数值移位对应 [56,136)，完整 FIR 输入支持 [19,160)。reflect/edge 的内部误差为 0；原模型仍对人工前缀敏感。正式 IQ、边界和测试 SNR 未改动。
source low urban 的 added-noise 比值中位数 7.250 dB，不能冒充总 SNR 或 reference_snr_db=30 dB。白输入代理替代为实际波形功率在 mid/low urban 改变 19/27 个接收状态；EQ=false 下噪声着色动态范围 0 dB，原 720 low 样本无 AGC 限幅。
首 seed 30 轮末实际批次梯度审计：C3 加权间隔梯度相对加权 satellite CE 的中位强度在所记录参数上为 1.714% 至 2.067%，说明此次系数下额外约束偏弱，不构成增大系数的事后实验。原分支输出、既有增益和头梯度均保留；没有强制全部特征或门控等于 clean。

## 实际成本与验证

| row | 更新数 | 主训练前向样本量 | 墙钟 (s) | eta_final | U urban 接受数/前向数 |
|---|---:|---:|---:|---:|---:|
| B0-s2026092701 | 6660 | 2838960 | 2908.865 | 0.000 | 0/0 |
| B0-s2026092702 | 6660 | 2838960 | 2883.979 | 0.000 | 0/0 |
| B0-s2026092703 | 6660 | 2838960 | 2860.872 | 0.000 | 0/0 |
| C0-s2026092701 | 6660 | 1986480 | 1399.203 | 0.000 | 0/0 |
| C0-s2026092702 | 6660 | 1986480 | 1448.135 | 0.000 | 0/0 |
| C0-s2026092703 | 6660 | 1986480 | 1545.217 | 0.000 | 0/0 |
| C1-s2026092701 | 6660 | 2838960 | 3022.855 | 0.000 | 0/0 |
| C1-s2026092702 | 6660 | 2838960 | 3063.974 | 0.000 | 0/0 |
| C1-s2026092703 | 6660 | 2838960 | 3122.094 | 0.000 | 0/0 |
| C2-s2026092701 | 6660 | 2838960 | 3629.760 | 0.400 | 0/0 |
| C2-s2026092702 | 6660 | 2838960 | 3884.022 | 0.450 | 0/0 |
| C2-s2026092703 | 6660 | 2838960 | 3849.473 | 0.400 | 0/0 |
| C3-s2026092701 | 6660 | 2838960 | 3842.922 | 0.450 | 0/0 |
| C3-s2026092702 | 6660 | 2838960 | 3614.438 | 0.400 | 0/0 |
| C3-s2026092703 | 6660 | 2838960 | 3880.415 | 0.400 | 0/0 |
| C4-s2026092701 | 6660 | 3405960 | 5285.457 | 0.400 | 8554/567000 |
| C4-s2026092702 | 6660 | 3405960 | 5818.129 | 0.500 | 8601/567000 |
| C4-s2026092703 | 6660 | 3405960 | 5552.197 | 0.500 | 8724/567000 |

总实际更新 119880，总主训练网络前向样本量 50244840，各 row 墙钟之和 61612.005 秒（并行不等于整批历时）。每 row 父训练成本为 0、新增/累计 30 轮；E50 历史训练 2119.459 秒仅作诊断参考。另有每 row 86400 个源 panel 前向、30 次 native source V 验证及 final source V 验证、初始化/诊断开销；墙钟已经包含训练流程这些成本。源受控机制分析实际观察前向数 434160，耗时 967.426 秒，无训练更新。
所有 119880 条更新与全部 stdout 已全量解析；无非有限跳步、技术错误或 U TX 可见记录。所有增强 row 的 TX×源 RX×三训练场景×遮挡状态 270 个 cell 都有实际样本。固定预算结束不等于收敛。
9 项聚焦测试通过，一次独立 P0/P1 审查及一次原问题定点复核完成。source branch hook 输出等价/恢复、六原 residual 分支梯度、11 原控制条件的冻结状态测试通过。真实 E50 checkpoint 无 query smoke、r02 单归档传输比较、远端 compile、PID/CWD/命令/GPU/log 增长核查通过。r01 因跨平台路径在提交前失败，产物保留，r02 修复后正常完成。

## 交付、负结果与边界

实际训练代码提交 f70d84672046309a0bf3d3a44ab23f5ab286b0de；源控制分析与报告提交见本分支历史。起点为 7fceadc5020c2970ae31b7502942f0453abff88f。主命令 python -u -m experiments.cvs_urban_sigtime.dispatch，冻结分析 python -u -m experiments.cvs_urban_sigtime.mechanism，分别从 r02 与 r02_analysis 的不可变 release 执行。
18 row、全部七场景/RX/day/TX 评分、逐 seed 表、完整训练曲线与分层图位于 E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/；权重、逐包 prediction、原始日志保存在 N607 同 run。它们没有进入 Git。
保留未达到 60%、未达到 +3/+5 探索目标、个别 seed clean 代价、C1 不优于拼接、额外间隔作用偏弱、U 弱门控低接受率及全部负差。未开展 KL 对照、超 E30 训练、结构/头替换、目标反馈调参或选择性重跑；不声称预算耗尽即收敛。
该证据属于固定 practical 代理信道与已暴露 benchmark 上的 Phase1；跨模型 seed 的重复小增益不能替代新数据或真实在轨信道确认。源损伤控制提供机制线索，正式收益只能由同 row 冻结默认头评分证明。


## 退化改善与负结果判断

主要收益来自真实 IQ 增强：B0 相对 C0 的 low urban +15.335、urban +17.359 个百分点，clean −1.653 个百分点。主方案 C3 相对用户指定 B0 的 low urban +0.659、mid urban +0.505、urban +0.393 个百分点，三 seed 均为正；urban Macro-F1 +0.526、最差 urban RX +0.492 个百分点。clean 平均 −0.108、suburban 平均 −0.204 个百分点。三个 seed 的 low 差分别 +0.545、+0.604、+0.828；clean 差分别 −0.634、+0.055、+0.254 个百分点。因此 clean 保持仅在均值上满足 0.5 个百分点条件，不能宣称逐 seed 全满足。

课程 C2 相对 C1 的 low +0.619 个百分点；间隔 C3 相对 C2 的 low +0.107 个百分点。三 seed 的这两项差均为正，但间隔引入后课程轨迹也会改变，不能把差全部当作损失的独立直接效应。C4 相对 C3 的 low +0.170、urban +0.063 个百分点，urban 仅 2/3 seed 为正，urban Macro-F1 反而 −0.061 个百分点。C4 low 均值 53.327%，也未达到 60%。不能按目标成绩把 C4 选作已证实的最优模型。

源域分层支持“遮挡后仍可恢复”的小幅改善：在同一 720 条验证 IQ 与命名流中，C3 相对 B0 的 low urban BLOCKED/degraded、added-noise 比值 3 至 8 dB 区间 +2.703 个百分点（每 seed 同一 74 条）；BLOCKED/unlocked、3 至 8 dB 区间 +3.292 个百分点（81 条）。BLOCKED/unlocked、低于 3 dB 区间 −0.280 个百分点（238 条）。mid urban BLOCKED/unlocked、3 至 8 dB 区间 +7.547 个百分点（53 条）。这些是三模型 seed 的源域均值，不能直接当作正式 target 分层因果效应；1 至 3 条的小分层不用于机制结论。

仍未解决的损伤主要包括新增噪声和传播核影响。首 seed C3 源 low urban 移除新增噪声后的固定态/重算态效应分别 +11.111/+11.528 个百分点；平坦 H 固定态 +4.306，移除 carrier +1.111，移除注入 RX IQ −0.139。因此证据不支持“噪声敏感性已消除”或“Doppler/IQ 是唯一失败原因”。换 edge 边界仍下降 10.694 个百分点，说明人工前缀敏感性保留；正式 reflect IQ 不作任何修改。

固定 E50 clean 正确集合的源 low urban 保持率：B0 为 61.746%，C3 为 62.303%，差 +0.557 个百分点。mid urban 差 +1.207 个百分点。默认头归一化最近竞争类间隔的 low 均值 B0=0.076087、C3=0.075149；C3 相对 C2=0.074296 有小幅增加，却未超过 B0。不能把准确率小增益宣称为所有样本或所有竞争类间隔全面增大。

原分支可允许不同漂移。真实分支观察显示信道增强后 reference/SIG_TIME 的 clean–satellite cosine 比 C0 高，而课程/间隔在 B0 之上的差小；这只是冻结源前向相关证据。已有 PA/reference/signature 增益随训练联合更新，不是额外部署门控，也未被强制等于 clean。C4 在三 seed 的 U urban 接受率仅为 1.509%、1.517%、1.539%；源 L 高置信代理不代表未知 U 精度，弱门控覆盖不足是本轮 SSL 的实际限制。

| 原冻结 E50 源场景 | 残余 CFO P10 / 中位数 / P90 (Hz) | 残余核能量 P10 / 中位数 / P90 |
|---|---:|---:|
| practical_mid_urban | -267.454 / 7.702 / 334.246 | 0.002135 / 0.256905 / 0.670371 |
| practical_low_urban | -483.512 / -5.294 / 398.911 | 0.000841 / 0.005129 / 0.343855 |

## 可检查的交付文件

- [18 row 逐 seed 汇总](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/per_seed_metrics.csv)
- [126 个逐 seed / 场景结果](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/per_scene_seed_metrics.csv)
- [2268 条 RX / day / TX 等分层评分](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/all_stratified_metrics.csv)
- [原独立评分与 confusion](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/scores.json)
- [119880 条更新的完整日志审计](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/training_audit.json)
- [26 次冻结源域分析](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/source_mechanism.json)
- [源域分析分层摘要](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/source_mechanism_summary.json)

- [三种子正式结果](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/three_seed_outcomes.png)
- [各 seed 配对差](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/paired_three_seed_changes.png)
- [全部三 seed 训练曲线](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/training_all_seeds.png)
- [已有增益训练曲线](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/existing_gate_training.png)
- [源域质量分层：线为均值，阴影为三 seed 最小至最大](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/source_quality_three_seeds.png)
- [固定 clean 正确集合保持率：横轴标注每 seed 样本数](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/fixed_reference_state_retention.png)
- [首 seed 固定态与重算态控制效应](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/trained_source_controlled_effects.png)
- [原分支漂移](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/final/figures/existing_branch_source_drift.png)

[FIR 有效观察区域](E:/type10-7/local_artifacts/cvs_urban_sigtime_20261014/source_diagnostic/effective_observation_region.png)。源图不能替代正式七环境结果。绘图脚本保存在同一本地 artifacts 根目录，输入为上述不可变结果。

代码变更包括 experiments/cvs_urban_sigtime/ 下的训练包装、信道桥接、目标函数、分层观察、源诊断、预测评分、队列、发布和机制分析，以及原 train_ssdg.py 的可选 callback。旧调用在 callback=None 时保持原行为。18 个实际配置、命令与路径见同目录 experiment.json/configs/。

## 原预登记、执行审计与协议边界

本轮按用户最新要求，从头 E30；对比基线为 SIG_TIME＋旧门控＋cos＋普通拼接星地增强。训练环境仅为 practical_low_suburban、practical_mid_urban、practical_low_urban。主验收目标是 low urban 准确率至少 60%；urban +3 个百分点、low urban +5 个百分点、clean 下降不超过 0.5 个百分点是相对 B0 的探索目标，不是结果保证，也不参与训练、扩种子或重跑选择。

## 代码审计与边界

起点 Git：7fceadc5020c2970ae31b7502942f0453abff88f。实际训练器与 practical 实现来自 experiments/adv3b02_xuc/code/。运行时保存 train_ssdg、leo_practical.channel 和 leo_practical.residual 的 __file__。固定单 SIG_TIME：t2 Dropout 后的 6 维局部投影、4 个时间窗、112 维描述子、joint160；特征参数 195041，原默认 ResidualPhysicalHead 6×160 cosine 头、scale=30 联合训练。参考响应和已有增益不增加、不替换。关闭额外拟合分类头。

ManySig 实际 equalized=1；25 MHz，256 点，native center crop 和逐包 RMS 归一化。星地增强再按原 practical 输入归一化，通过原 residual 引擎得到 IQ，最后由原 AGC 处理，不追加输出 RMS 归一化。录音的 equalized=1 不代表新增 practical 信道被均衡。本轮 residual/post_sync、EQ=false，因此 G=1；reference_snr_db=30，质量适配、相位跟踪和注入 RX IQ 补偿均启用，unlocked 状态由原引擎关闭精细跟踪，边界 reflect，数值 FIR 延迟 24 点。新增噪声 SNR 不是录音总 SNR，不能反演未知 TX 硬件效应。

合法源角色保持 L/U/V=6300/56700/27000，source RX=1/3/4/6/8，day=1/2/3，split=392005；native 逐物理角色 ID 与现有 source_contract 精确匹配，U TX 隐藏。源 V 仅健康监测，不梯度更新。RX 硬件按 RX/day/eq session 稳定，session 不含 TX；命名随机流区分训练 epoch/step/view、固定源验证与诊断。所有合法有标签 satellite IQ 都参与 CE，无学生置信度筛选。

已完成的源诊断参考是同一单 SIG_TIME 的固定 final50 checkpoint（源 run 20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01，实际执行提交 512c10a267b5e654cfd2c983ab1debdeebc7f626）。其 scratch、11100 更新、source-only、默认头来源已读回；真实 strict-load smoke 和源域检查由 launcher 第一阶段执行。它只用于冻结诊断，不作为本轮初始化。新模型与自己的 EMA 均从零开始，父训练成本为 0，新增/累计为 E30。SIG_TIME 是用户指定的已暴露 benchmark 起点，本轮不声称新的独立盲测。

## 可执行矩阵与归因

矩阵和每 row 配置唯一维护于 experiment.json。原预登记 18 row，现全部完成。首 seed=2026092701，先运行以下 5 臂，随后仅按源证据冻结条件原样扩展 02/03；C4 须源标签代理 precision≥.95、coverage≥.1 后独立从零运行。

| 臂 | 唯一差异 |
|---|---|
| C0 | clean 归因控制，原旧门控＋EMA＋cos |
| B0 | 用户指定对比基线，clean/satellite 普通拼接一次前向，三环境等先验 |
| C1 | 同样 IQ 信道 draw，双视图分别前向，clean/satellite CE 各自 mean 后 1:1；无额外一致性 |
| C2 | C1＋有界源域状态课程，eta≤.5；每个已登记场景和状态仍有 nominal 支持 |
| C3 | C2＋单侧默认头间隔，rho=.5、m_max=1、固定 scale=30；clean 正确参考 stop-gradient，q 连续且只调额外约束 |
| C4 | C3＋原 EMA clean 软标签→同一 U 的 mid/low urban 视图；不要求 urban 学生先高置信 |

B0 拼接 CE 和 C1 分开 mean 的 CE 在相同 logits 下数学等价；二者执行与 Dropout 随机实现不同，不能把归一化本身宣称为创新。旧半监督和 EMA 在所有臂中保留为共同训练配方：20 轮 label、10 轮 pseudo，原弱/强门控不变，lambda_u=.16、entropy=.01、label_smoothing=.01；C4 是独立新增 U urban 软监督。所有网络与默认头正常联合更新。

每臂 30×222=6660 更新，L batch128，U batch256（尾批按原 loader）。AdamW lr=.0002，weight_decay=.0001，min_lr=.000001，原 cosine 压缩至 E30，固定 final30。记录实际更新、各类前向样本量、源验证成本、墙钟、完整损失曲线、门控和梯度变化；固定预算不能当作收敛证据。

源诊断固定 720 条 TX×RX×day 平衡 L 样本和同一命名流，比较完整 residual、移除新增噪声、平坦能量匹配 H、移除 carrier 残差、移除注入 RX IQ；分别记录重算接收状态的总效应和固定接收状态/频率/相位/EQ/AGC 的受控效应。另比较白输入代理与实际波形功率、reflect/edge 的人工边界及 FIR 有效区域。仅源 IQ，原默认头，无参数更新或拟合诊断头。

源扩种条件预先固定于 dispatch.py：各 row exact6660、合法角色、真实 labeled satellite 覆盖；候选源 L panel clean 比 B0 下降不超过 .02、hard margin 比早期三轮均值下降不超过 .01、final L CE<2。它们是早期可执行性/学习健康条件，不是目标效果证明。source_qualification.json 冻结在汇总目标成绩之前，目标不影响晋级或选择性重跑。

## 正式评价、命令与产物

复用 20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views 的 VALIDATED_ONCE 固定数组和现有 clean capsule：每场景 168000 条，7 个未见 RX、4 day、6 TX。保持完整 clean＋六 practical，保留全部困难样本，不调测试信道、SNR、状态或归一化。每 row 训练冻结后立即自动预测全部 7 视图；独立 exec scorer 完成后才打开 truth，并独立复算 accuracy/Macro-F1/confusion。报告逐 seed、urban 均值、low/mid、最差 urban RX、clean/suburban，以及全部负结果和缺项。

远端 CWD 为 PROJECT/releases/cvs_sigtime_urban_paired_20261014_r02；环境为 N607 既有 CVS-RFFI Python。精确主命令：`python -u -m experiments.cvs_urban_sigtime.dispatch`，worker 为同模块 `--mode worker --row <row>`，随后同 PID exec 到 predict/score。唯一 owner 为 codex/root/urban-sigtime-20261014，每 GPU 包含其他 owner 最多 3 个实验进程；不修改或重启其他健康任务。

代码/config 的实际 Git commit 由 release_commit.txt 固定，并写入远端 experiment.json 与运行证据；不要求设计、报告或单成员 hash。一次 release 归档、一次本机/远端 archive SHA 对比、一次远端 compile。launcher 第一步真实 checkpoint 无 query smoke，PASS 后直接源诊断和训练；不存在 smoke token 或额外审批。

技术停止仅限协议/角色/预算错误、输出冲突、非有限、无法执行或无法产生合法 prediction。保留全部产物，停止 pending 派发并让其他健康 row 收尾；低性能不技术停止，不自动重启。启动后核对 PID/CWD/argv/GPU/log 增长。

期望产物：source_diagnostic/results.json、各 row 的 final_ssdg.pth、完整 step_metrics、source_learning_progress、training_coverage、gradient_gate_history、七视图 predictions、独立 scores、summary、resources、analysis 和分层图/训练曲线。权重、原始日志、逐样本预测留在既有实验存储，不进入 Git。

## 已执行验证

本地 `python -m experiments.cvs_urban_sigtime.checks`：9 项通过。包括 margin stop-gradient/scale、不筛除真实标签 CE、旧头/实际特征数值更新、隐标签元数据/session、跨 epoch 刷新、原 residual 全数值对齐、EQ 噪声着色、固定/重算状态控制、预算/目标配置负测、真实普通拼接及 U urban 约束。修改模块和发布脚本语法编译通过。

一次独立 P0/P1 审查指出 baseline_reuse 缺字段和 C4 U 混入 suburban，已修复；一次对应问题定点复核确认无遗留。没有额外全量审查、文档 gate、hash 链或目标成绩选模。最终三种子证据见正文；预登记和源规则原样保留。

## 技术恢复

r01 在提交进程前因本机生成的 Linux manifest 路径含反斜杠而失败，独立读回无 submit、worker 和运行输出。失败归档保留。r02 只修复跨平台路径，不改变科学矩阵、物理信道、预算或数据。

## 已核实的启动与源域证据

实际训练提交 f70d84672046309a0bf3d3a44ab23f5ab286b0de 已 push 并读回一致。r02 的传输、远端编译、PID/CWD/命令及五 worker GPU/log 增长已独立核实。源诊断完成 720×6×11 条观察，耗时 130.450 秒，无模型状态改变。E50 参考训练历史成本为 2119.459 秒，只供诊断，本轮继承成本为 0。

| 源域控制条件 | mid urban 准确率 (%) | low urban 准确率 (%) | low 相对 full (百分点) |
|---|---:|---:|---:|
| 完整 residual | 53.472 | 44.722 | 0 |
| 移除新增噪声，固定接收状态 | 59.444 | 53.333 | +8.611 |
| 移除新增噪声，重算接收状态 | 60.417 | 53.611 | +8.889 |
| 平坦传播核，固定接收状态 | 58.472 | 47.778 | +3.056 |
| 移除 carrier 残差，固定接收状态 | 53.472 | 46.389 | +1.667 |
| 移除注入 RX IQ，固定接收状态 | 53.472 | 44.583 | -0.139 |
| 实际波形功率替代白输入代理 | 53.611 | 45.000 | +0.278 |
| reflect 改为 edge，仅源诊断 | 43.194 | 37.361 | -7.361 |

mid/low urban 的实际波形功率代理分别改变 19/27 个接收状态；移除新增噪声后重算状态分别改变 294/448 个。以上是冻结 E50 在合法 L_s 的同样本控制效应，不能解释为增强收益或目标准确率。边界效应说明整个前向对人工前缀敏感，正式 IQ、数值延迟和边界保持不变。

训练后补充仅源域冻结分析：CPU 命令 python -u -m experiments.cvs_urban_sigtime.mechanism；从独立不可变 releases/cvs_sigtime_urban_paired_20261014_r02_analysis 执行，输出同 run/source_mechanism。观察真实 time/frequency/PA/reference/SIG_TIME 分支、3 类已有增益与默认头梯度，不拟合头，不更新参数，不读取 query/truth。所有 E30 row 使用同一验证随机流，并使用固定 E50 clean 正确 ID 集合计算保持率；E50 的原控制随机流另行报告，避免跨随机流直接归因。已通过实际网络 branch hook 输出等价/恢复检查和六 residual 分支梯度、冻结状态检查。此分析不参与训练、扩种子或选择。
