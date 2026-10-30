# Phase3 JOINT-FEATURE 结果（2026-10-31）

**唯一主方法 `joint_predicted_ratio` 不晋级（NO_PROMOTION）。** 相对原 `predicted_ratio`，星地未知拒对净增 4 条，但正确接收净少 4 条；H 仅增加 0.143614 个百分点，不能抵消 A_K 下降、已知保护和覆盖失败。保留原 predicted 重点候选及 pooled 对照，全目标 ACTIVE。

Run：`20261031-phase3-joint-feature-s2026092701-r01`；实际代码 `b7f81eca9a2a0bb55afc361acd50f3fbd739b68a`。本轮三条件及四方法均事前固定，没有按 Q 换主。

## 方法和数据权限

原 F0/default head、DURA、P1 signal r03 和原 160 维 density11 全部冻结。新表示将同 P1 输入的原 F0 单位 u160 与 r03 单位 z160 拼接并归一化为 320 维；仅以源 L6300/OE6134 独立拟合联合白化、90 个已知格和 32 个背景分量，不是局部完整协方差，也不是零统计拟合。
源 L 的六旧类×五 RX×三日各 70 条；L 标签合法，OE 仅使用 outside-six 成员监督和公共 RX/day，不读 OE TX/K。原 Phase1 来源为 L6300/U_s56700/V27000，L 占 L+U_s 的 10%；本轮 U_s 不加载，V27000 是额外已知校准标签。
源 u160 复用合规缓存并与既有 G10/P1 L/OE IQ 的 ID/元数据对齐，新做 signal z 共 12434 行。源 V/NV 的 u/z、身份输出和两旧控制复用；两新方法各用 V 校准六个预测类阈值，四组阈值全部冻结后才访问 NV/Q。
源 RX1/3/4/6/8 对应 1-19/18-2/19-2/2-19/3-19，日期 2021-03-08、03-15、03-23。源 NV3150 为历史已暴露的 19-15、19-16、7-1，各 1050 条，仍是已见源 RX。
clean/star Q 各 864，均为 RX24-1、2021-03-15 的相同物理记录：六旧类×96+三未知类×96=576+288。两条件均经 DURA；本轮新算同 G→F0 的 u，核对 logits 数值与 closed 一致，复用原 z，不新前向 Q signal。源与目标跨 RX，但日期不是源未见；全部预测完整后才独立连接三份 truth。

## 十二个固定运行点

A_K/C_K/R_U 分别为逐真实类别宏平均正确接收率、接收率、未知拒识率，H=2A_KR_U/(A_K+R_U)。下表为百分比；F0 损失为同 DURA 输入的原 F0 闭集减 A_K，单位为百分点。

| 条件 | 方法 | A_K | C_K | R_U | H | F0 损失 | AUROC | OSCR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| source DURA | pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 | 1.2111 | 94.9853 | 87.9447 |
| clean DURA | pooled_ratio | 68.9236 | 85.4167 | 99.6528 | 81.4874 | 11.1111 | 95.9919 | 77.4637 |
| star DURA | pooled_ratio | 63.0208 | 84.8958 | 76.3889 | 69.0639 | 9.8958 | 84.7283 | 62.6049 |
| source DURA | predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 | 1.2370 | 94.7343 | 87.9376 |
| clean DURA | predicted_ratio | 70.8333 | 87.3264 | 100.0000 | 82.9268 | 9.2014 | 96.1583 | 77.7705 |
| star DURA | predicted_ratio | 64.0625 | 85.5903 | 77.0833 | 69.9723 | 8.8542 | 85.0279 | 62.9823 |
| source DURA | joint_pooled_ratio | 89.7000 | 96.9852 | 66.5397 | 76.4032 | 1.2815 | 95.3670 | 88.2053 |
| clean DURA | joint_pooled_ratio | 62.5000 | 78.9931 | 100.0000 | 76.9231 | 17.5347 | 97.0703 | 78.5530 |
| star DURA | joint_pooled_ratio | 61.4583 | 83.3333 | 79.8611 | 69.4615 | 11.4583 | 85.1984 | 62.9382 |
| source DURA | **joint_predicted_ratio** | 89.7222 | 96.7852 | 66.6667 | 76.4950 | 1.2593 | 95.0848 | 88.1101 |
| clean DURA | **joint_predicted_ratio** | 67.7083 | 84.2014 | 100.0000 | 80.7453 | 12.3264 | 97.0528 | 78.5687 |
| star DURA | **joint_predicted_ratio** | 63.3681 | 84.7222 | 78.4722 | 70.1159 | 9.5486 | 85.7271 | 63.5483 |

同路径原 F0 闭集始终为 source 90.981481%、clean 80.034722%、star 72.916667%，身份预测没有变化。主方法 source/clean/star 正确接收 24225/27000、390/576、365/576，未知拒对 2100/3150、288/288、226/288。clean 的 R_U100% 不能补救 C_K84.201389%/损失12.326389pp；star 的 C_K84.722222%/损失9.548611pp 同样失败。

## 同 ID 配对与身份损失

以下为主方法减原 `predicted_ratio`。gain/loss/net 区分新增与丢失事件，不能只看净 H。

| 条件 | 正确接收 gain/loss/net | 接收 net | 未知拒对 gain/loss/net | ΔA_K / ΔR_U / ΔH（百分点） |
|---|---:|---:|---:|---:|
| source | 42/48/−6 | +7 | 223/74/+149 | −0.022222 / +4.730159 / +3.203530 |
| clean | 3/21/−18 | −18 | 0/0/0 | −3.125000 / 0 / −2.181488 |
| star | 5/9/−4 | −5 | 11/7/+4 | −0.694444 / +1.388889 / +0.143614 |

星地主方法相对原 pooled 的正确接收/未知拒对净数为 +2/+6，相对 joint pooled 为 +11/−4；这些局部比较不能替代事前 predicted 控制或唯一主方法的门槛。两新分数分别按已知 V 校准，其运行点差异不能全归因于未校准的拼接算子。
接收率不等于身份准确率。主 clean 接收 485 条但只正确 390 条，71 条原本闭集正确记录被拒；star 接收 488 条但只正确 365 条，55 条原本正确记录被拒。14-7 在 star 接收 96 条但仅正确 32 条。所有差异来自拒绝集合，不是默认身份头修正。

## 分组和缺失覆盖

四方法 source known12 均过，known42 均未过。主方法有五个 RX×scene 诊断组超过 2pp；它们仍完整报告，但不冒称 hard42。主方法现存 next-q 在 RX8 绑定，下一点 F0 损失为 2.018519pp；未据 NV/Q 扫描或另选阈值。source 总体 R_U90 未过，仅作开发诊断。

| 星地主场景 | known/unknown | A_K | C_K | R_U | F0 损失（pp） |
|---|---:|---:|---:|---:|---:|
| practical_high | 33/6 | 67.3016 | 87.7778 | N/A | 8.8889 |
| practical_mid | 42/25 | 66.3889 | 83.8889 | 91.6667 | 16.1111 |
| practical_low_suburban | 203/117 | 65.7642 | 85.7444 | 90.0807 | 10.3610 |
| practical_high_urban | 38/16 | 75.3307 | 91.0714 | 93.3333 | 6.5476 |
| practical_mid_urban | 45/15 | 74.9802 | 93.9153 | 81.1111 | 6.0847 |
| practical_low_urban | 215/109 | 53.8573 | 79.2809 | 58.1720 | 9.6723 |

high 的未知三类数为 0/4/2，缺 19-15，因此宏 R_U、H、曲线为 N/A；不能以 6/6 pooled 拒对补认证。high urban/mid urban 的 C_K 虽过 90%，身份损失仍超 2pp；六景身份保护全部失败，low urban 宏 R_U 仅 58.172043%。总体不能代替各必需主景。
主方法对 19-15/19-16/7-1 的源拒对数为 651/652/797（各1050），clean 为 96/96/96，star 为 77/74/75（各96）。全部方法的 TX/RX/day/scene、42 组和逐真实类 Wilson 区间见原 score 和私有 analysis；单开发 RX、同物理条件重复不构成独立样本或全 RX 置信证明。

## 新增统计与实际执行

新联合密度确实拟合一次，known90/background32 均在 V 前冻结；背景固定 seed2026103019、25 次 Lloyd，全部有限且 SSE 非增：339.557396→296.237209，32 个活跃分量、无空模。known 白化协方差数据/各向同性权重为 .9/.1，D320 的方差收缩系数为 70/390=.179487，而原 D160 为 70/230；维数、白化、体积和收缩共同改变，不伪称沿用同一 160 维分布。
joint 聚合数组共 1136236 B（NPZ1139110 B）：白化320×320占819200 B、known90×320均值230400 B、background32×320均值81920 B；无逐源行部署 bank。原 density 数组364340 B保持精确，四组六阈值192 B；新增仅其中12个阈值。
NN 更新0，原 F0 232993+signal334721+DURA266951=834665 参数不变。新增真实 signal12434 行（L+OE）、Q F0/G各1728行；Q signal 新前向0、source F0/G新前向0；真实 checkpoint 合成三模型各2行。源 V/NV 的旧结果复用不增加物理记录，原训练和恢复缓存/Q signal 成本仍继承。
原 F0 的11100更新、r03复制特征新增2220/投影2220、DURA4940更新及原OE监督背景均单列为祖先成本，不能把本轮称零统计拟合或无OE监督。predict24.475319 s、CPU score12.593357 s、pipeline37.150677 s为整进程时间，不是单包时延。
两阶段均 COMPLETE/returncode0，pipeline ANALYZED_DEVELOPMENT。三次 observations 独立捕获 driver1124895及predictor1124984的CWD/完整argv/CUDA4，GPU UUID95d925a1单次444 MiB；predict.log549→695 B。此为离散样本，不是显存峰值。
CPU scorer1125157仅有pipeline gpu=None/returncode0和代码CUDAempty设定，未捕获其独立live遥测。终态live/owned GPU均空、foreign_operations=0；登记锁竞争属于元数据收尾，不是远端实验技术失败。
完整三份运行stdout共759 B/9行，无warning/error/traceback或非有限提示。29unique+44sub PASS：core16项含新joint5与旧兼容11，root10、score3；持久score测试日志保留fixture作用域NameError及定向修复、旧np.trapz警告，测试中注入NaN不是实际运行非有限失败。唯一独立P0/P1 review PASS。
单包616files/data0，archive SHA `11baaf461ca37e9e402619e7bfeaa76eb5a21ccc68275c1d9cdce49a830c94fa` 比较、远端compile/runtime均 VERIFIED。无新训练、阈值扫描或补跑用于解释本结果。

本轮 U0、正式RX24-18访问0，无DA/匿名REG，A_N/H_OW=N/A。源RX已见、源NV及Q历史曝光、单seed/单开发RX、V曾用于校准和high缺类边界保持；H微增或某控制局部收益不构成独立确认，目标仍ACTIVE。

完整证据：[score/results.json](../local_artifacts/pamf_phase3_joint_feature_20261031_r01/score/results.json)、[私有analysis.json](../local_artifacts/pamf_phase3_joint_feature_20261031_r01/analysis.json)、[pipeline](../local_artifacts/pamf_phase3_joint_feature_20261031_r01/pipeline_state.json)、[本run报告](../automation_reports/CV-SincNet/20261031-phase3-joint-feature-s2026092701-r01/report.md)。
