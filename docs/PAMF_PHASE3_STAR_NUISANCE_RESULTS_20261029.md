# G10 源端 RX/day 硬投影证伪结果（2026-10-29）

**预定主候选 `rxday::RMD::P1`：NO_PROMOTION；本轮未访问 target，总 Phase3 目标保持 ACTIVE。** 本轮只检验源域开放集能力，不能证明跨 RX 泛化、无标签适应或匿名注册成功。

性能 run `20261029-phase3-star-nuisance-s2026092701-r01`；数据 run `20261029-phase3-star-source-novel-g10-r01`。实际代码 `fccbde38a407891db5acce776d6ad7af891f1341`，预登记 `9b90bc912cf1aa738a345174478339b31a04e1b7`。三个变体、两种距离、两条路径共12列全部报告，结果不改变主候选。

## 数据、权限与匹配控制

原 F0 为 Tri-SIG-TIME E50/11100、232993 参数，保留完整六类默认头；Phase1 祖先 L6300/U56700/V27000，源训练有标签比例 L/(L+U)=10%。本轮只用 L6300 标签拟合统计，V27000 的额外 known 标签只校准阈值；不加载 U_s IQ，不消费 OE 训练 IQ，也不使用 E51 或目标数据；novel builder复用的parent metadata整文件读取不等于消费OE训练IQ。旧类顺序为14-10、14-7、20-15、20-19、6-15、8-20。

source RX index1/3/4/6/8＝1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。novel3150为19-15、19-16、7-1各1050条，每TX×RX×day70条；它们此前已有源开发评测暴露，且与known同RX/day，不是新TX盲测或未见RX验证。

known L/V按已验证 metadata 的命名TX、rx/day/eq/sig_i索引 raw ManySig；sig_i不等于base_index。整PKL载入有全文件IO，但实际模型/统计只消费L/V，U_s缓存未打开。Full novel只回读既有15个成员与原3150条映射，不重采样、不读旧归一化IQ/旧truth或按分数选记录。

全部新模型输入均为G10 MIX01星地：scene2026103001/channel2026104001、2.45GHz、receiver_seed2027、25MHz、residual/post_sync/no EQ；raw256点转换布局后由observe完成一次FP32 unitRMS，输出不再按样本归一化。P0直接F0；P1为同一received经固定DURA v1再F0，两者都不是clean。

33300条known是上一轮相同physicalID、相同episode的确定性G10重建；marker中的new_star_observations=33300只是本进程生成数。novel3150为既有物理记录的新G10实现。两者都不增加独立物理样本；本轮36450物理记录产生72900条F、36450条G计算曝光，另有合成smoke。

## 固定机制与指标

每path仅用星地L拟合自己的八branch geometry，control与既有MD/RMD数学一致。rx或rxday在五个源标准化单位分支上计算同类RX或RX×day均值残差，先移除六类between-class均值span分量，再SVD取90%残差能量、rank上限16，真正删除所选方向后重新单位化。不是可逆缩放后重拟合协方差。

投影后的六类/背景均值、0.8协方差＋0.2 trace/d等方差shrinkage、MD/RMD源分数均值/标准差都只来自L。默认头与身份选择完全不变，投影仅改变拒识分数。类间span只保证重新单位化之前在源L上保留；重新单位化及未知身份方向不受此保证，诊断分别记录。

V只为每列求一个最大合法全局τ：overall＋5RX＋6scene＋30RX×scene共42组，FRR≤5%、同输入同path原F0 closed−open≤2pp，ties整组不拆。缺known类不可认证；零拒绝仍无解时接受全部并保留false。全部L聚合及τ冻结后才打开novel公共预测，12列预测完成后独立scorer连接truth。

主性能为A_K（正确身份且接受）、R_U（未知拒识）、H_OS及宏加权OSCR；C_K接收率只作保护，错身份接收仍在分子。AUROC/OSCR概括固定预测，不用于选择阈值。所有TX先各自计算再等权；缺必需类为N/A且不能认证。源gate要求42known组及overall/5RX/6scene未知≥90%，RX×scene未知仅诊断。

## 十二列总体结果

| 候选 | C_K% | A_K% | R_U% | H_OS% | macroOSCR% | macroAUROC% | F0损pp | 源gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control::MD::P0 | 99.0519 | 75.1370 | 2.7937 | 5.3870 | 58.6502 | 75.5740 | 0.6741 | False |
| control::MD::P1 | 99.3296 | 90.7037 | 0.8889 | 1.7605 | 77.2587 | 83.3026 | 0.2778 | False |
| control::RMD::P0 | 99.0852 | 75.3778 | 4.0000 | 7.5969 | 62.5673 | 75.1695 | 0.4333 | False |
| control::RMD::P1 | 98.8370 | 90.4519 | 6.4444 | 12.0317 | 78.1555 | 83.8635 | 0.5296 | False |
| rx::MD::P0 | 99.0815 | 75.2815 | 4.6667 | 8.7885 | 61.4871 | 77.3238 | 0.5296 | False |
| rx::MD::P1 | 99.1037 | 90.5963 | 1.9048 | 3.7311 | 77.6677 | 83.6363 | 0.3852 | False |
| rx::RMD::P0 | 98.7296 | 75.2481 | 5.1429 | 9.6277 | 62.5699 | 75.2145 | 0.5630 | False |
| rx::RMD::P1 | 99.2111 | 90.6667 | 2.0952 | 4.0958 | 77.1439 | 82.6866 | 0.3148 | False |
| rxday::MD::P0 | 99.2444 | 75.4593 | 3.9365 | 7.4827 | 62.8788 | 77.3791 | 0.3519 | False |
| rxday::MD::P1 | 99.1963 | 90.6333 | 2.1270 | 4.1564 | 77.9211 | 83.9053 | 0.3481 | False |
| rxday::RMD::P0 | 98.9741 | 75.3593 | 3.8413 | 7.3099 | 62.4770 | 75.0298 | 0.4519 | False |
| rxday::RMD::P1 | 99.1741 | 90.6296 | 2.3492 | 4.5797 | 77.2395 | 82.7789 | 0.3519 | False |

主候选：A_K=90.6296%，R_U=2.3492%，H_OS=4.5797%，macroOSCR=77.2395%；known42组=True，known校准=True，未知12个必需组=False。

主候选known接受/正确接受/原F0正确=[26777, 24470, 24565]/27000；unknown拒绝=74/3150。总体各TX等量，Wilson95%记录计数区间：接收99.06–99.28%、正确接受90.28–90.97%、未知拒识1.88–2.94%。这不是跨RX总体置信区间，重复信道/path不独立；不对不等量scene套pooled区间冒充macro区间。

## 校准、配对增益与分组

每列使用自己事前规定的known-only校准τ；运行点差值同时包含分数变化与合法阈值迁移。AUROC/OSCR可帮助辨别排序变化，不能把不同τ的拒识差值全部归因于投影。

| 候选 | τ | 零拒绝可行 | 42组通过 | 缺类组 | 最坏FRR% | 最坏损pp | 下一τ仍合法 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control::MD::P0 | -1.707161 | True | True | 0 | 2.8423 | 1.9735 | False |
| control::MD::P1 | -2.435494 | True | True | 0 | 2.4501 | 1.8138 | False |
| control::RMD::P0 | -1.388806 | True | True | 0 | 2.6993 | 1.9939 | False |
| control::RMD::P1 | -2.099503 | True | True | 0 | 3.7930 | 1.7859 | False |
| rx::MD::P0 | -1.660043 | True | True | 0 | 2.6200 | 1.6761 | False |
| rx::MD::P1 | -2.41076 | True | True | 0 | 3.2191 | 1.8928 | False |
| rx::RMD::P0 | -1.321636 | True | True | 0 | 3.4125 | 1.9677 | False |
| rx::RMD::P1 | -2.34438 | True | True | 0 | 3.0458 | 1.8138 | False |
| rxday::MD::P0 | -1.689592 | True | True | 0 | 3.0846 | 1.7766 | False |
| rxday::MD::P1 | -2.516184 | True | True | 0 | 2.9385 | 1.8138 | False |
| rxday::RMD::P0 | -1.431942 | True | True | 0 | 2.7714 | 1.7766 | False |
| rxday::RMD::P1 | -2.377502 | True | True | 0 | 3.0711 | 1.7793 | False |

| 相对匹配control | ΔA_K pp | ΔR_U pp | ΔH_OS pp | ΔOSCR pp | ΔAUROC pp |
| --- | --- | --- | --- | --- | --- |
| rx::MD::P0 | 0.1444 | 1.8730 | 3.4015 | 2.8369 | 1.7498 |
| rxday::MD::P0 | 0.3222 | 1.1429 | 2.0957 | 4.2286 | 1.8052 |
| rx::RMD::P0 | -0.1296 | 1.1429 | 2.0308 | 0.0025 | 0.0451 |
| rxday::RMD::P0 | -0.0185 | -0.1587 | -0.2869 | -0.0904 | -0.1397 |
| rx::MD::P1 | -0.1074 | 1.0159 | 1.9706 | 0.4090 | 0.3338 |
| rxday::MD::P1 | -0.0704 | 1.2381 | 2.3959 | 0.6624 | 0.6028 |
| rx::RMD::P1 | 0.2148 | -4.3492 | -7.9358 | -1.0116 | -1.1769 |
| rxday::RMD::P1 | 0.1778 | -4.0952 | -7.4520 | -0.9160 | -1.0847 |

下表比较三种P1/RMD；单元格为A_K / R_U / H_OS（%），coverage为known6/novel3是否齐全。完整12列×42组含RX×scene保留在score/results.json::candidates::<列>::groups。

| 组 | control | rx | rxday | 主coverage |
| --- | --- | --- | --- | --- |
| rx/1 | 93.0185 / 1.9048 / 3.7331 | 93.0926 / 0.7937 / 1.5739 | 93.0926 / 0.4762 / 0.9475 | True/True |
| rx/3 | 83.9259 / 5.7143 / 10.7000 | 84.2593 / 3.4921 / 6.7062 | 84.1667 / 4.7619 / 9.0138 | True/True |
| rx/4 | 91.4444 / 8.2540 / 15.1413 | 91.5926 / 2.3810 / 4.6413 | 91.5741 / 2.2222 / 4.3391 | True/True |
| rx/6 | 92.6667 / 11.7460 / 20.8493 | 92.7593 / 0.6349 / 1.2612 | 92.7407 / 0.4762 / 0.9475 | True/True |
| rx/8 | 91.2037 / 4.6032 / 8.7640 | 91.6296 / 3.1746 / 6.1366 | 91.5741 / 3.8095 / 7.3148 | True/True |
| scene/practical_high | 97.4745 / 7.1577 / 13.3360 | 97.7429 / 2.2796 / 4.4553 | 97.6524 / 2.7738 / 5.3944 | True/True |
| scene/practical_mid | 96.9390 / 9.6592 / 17.5678 | 96.9928 / 3.0977 / 6.0037 | 97.0452 / 2.7146 / 5.2815 | True/True |
| scene/practical_low_suburban | 94.9977 / 8.0563 / 14.8530 | 95.2659 / 2.8780 / 5.5872 | 95.2179 / 3.0694 / 5.9472 | True/True |
| scene/practical_high_urban | 96.7717 / 9.3782 / 17.0993 | 96.7717 / 1.4654 / 2.8870 | 96.7717 / 2.2061 / 4.3139 | True/True |
| scene/practical_mid_urban | 91.0974 / 4.6688 / 8.8824 | 91.0936 / 1.2607 / 2.4869 | 91.1962 / 1.2607 / 2.4870 | True/True |
| scene/practical_low_urban | 83.1351 / 4.0327 / 7.6923 | 83.3835 / 1.2539 / 2.4706 | 83.3166 / 1.6797 / 3.2931 | True/True |

主候选known保护失败/不可认证组：无；未知90%失败/不可认证必需组：overall, rx/1, rx/3, rx/4, rx/6, rx/8, scene/practical_high, scene/practical_mid, scene/practical_low_suburban, scene/practical_high_urban, scene/practical_mid_urban, scene/practical_low_urban。N/A不填0，也不当通过。

主分数macroAUROC=82.7789%<85.5%。同一宏加权known/unknown分布若能在一个τ同时达到C_K≥95%和R_U≥90%，至少0.95×0.90=0.855的成对分数严格有序，因此AUROC≥0.855是必要条件。低于此界的必需组为overall、rx/3、rx/8、scene/practical_mid_urban、scene/practical_low_urban；仅移动本分数τ无法同时达标。高于此界也不保证存在合法τ。本判断不扫描oracle阈值，且不是所有模型都不可能的结论。

## 逐TX与投影实测

| TX | N | known接收% | 正确接受% | F0closed% | 未知拒识% |
| --- | --- | --- | --- | --- | --- |
| 14-10 | 4500 | 98.8889 | 88.7333 | 89.0444 | — |
| 14-7 | 4500 | 99.4222 | 80.1556 | 80.3333 | — |
| 20-15 | 4500 | 99.6889 | 95.8444 | 95.9111 | — |
| 20-19 | 4500 | 98.6444 | 84.6444 | 85.4667 | — |
| 6-15 | 4500 | 98.6222 | 95.6667 | 96.3333 | — |
| 8-20 | 4500 | 99.7778 | 98.7333 | 98.8000 | — |
| 19-15 | 1050 | — | — | — | 2.7619 |
| 19-16 | 1050 | — | — | — | 1.8095 |
| 7-1 | 1050 | — | — | — | 2.4762 |

以下五分支单元格为rank / 实际能量覆盖% / reunit后类间均值相对变化%；无残差覆盖为N/A。删除rank达到16也不等于覆盖90%；实际energy_target_met和全部范数异常均读取，不以事前预算代替实测。

| path/variant | fused160 | time160 | frequency160 | pa160 | pa_moments48 |
| --- | --- | --- | --- | --- | --- |
| P0/control | 0/N/A/0.00 | 0/N/A/0.00 | 0/N/A/0.00 | 0/N/A/0.00 | 0/N/A/0.00 |
| P0/rx | 11/91.10/22.35 | 10/91.25/5.70 | 7/91.68/20.10 | 4/90.05/9.08 | 4/94.67/9.92 |
| P0/rxday | 16/89.56/35.97 | 16/90.91/8.06 | 9/90.48/31.47 | 8/91.09/11.76 | 4/91.93/10.11 |
| P1/control | 0/N/A/0.00 | 0/N/A/0.00 | 0/N/A/0.00 | 0/N/A/0.00 | 0/N/A/0.00 |
| P1/rx | 9/91.56/24.69 | 10/90.54/3.95 | 7/90.64/16.60 | 4/92.09/8.69 | 3/93.82/6.52 |
| P1/rxday | 12/90.27/31.29 | 15/90.77/4.96 | 9/90.87/19.46 | 5/90.88/9.28 | 3/92.17/6.53 |

| path/variant | 最大preunit类间相对损伤 | 投影剩余范数范围 | reunit异常 | 输入零范数 | 达到90%的分支数 |
| --- | --- | --- | --- | --- | --- |
| P0/control | 0 | 1–1 | 0 | 0 | 0 |
| P0/rx | 4.15e-15 | 0.198474–0.999809 | 0 | 0 | 5 |
| P0/rxday | 2.87e-15 | 0.197999–0.997751 | 0 | 0 | 4 |
| P1/control | 0 | 1–1 | 0 | 0 | 0 |
| P1/rx | 9.75e-16 | 0.353744–0.999802 | 0 | 0 | 5 |
| P1/rxday | 9.5e-16 | 0.34823–0.999726 | 0 | 0 | 5 |

主P1/rxday五branch确实删除[12, 15, 9, 5, 3]个方向，各自达到90%残差能量，非identity fallback。preunit类间相对损伤最大9.5e-16，但fused重新单位化后类间均值变化31.29%；这说明线性span保护不等于最终几何或未知身份保真。P0/rxday的fused被cap16限制，仅覆盖89.56%，未写成达成90%。

源L类均值span的数值保护不证明未知身份信息仍在；V/novel从未参与投影、Gaussian或score scale拟合。control固定I，rx/rxday若无可用残差则I回退并如实记录。所有路径共享各自原F0闭集预测，身份正确数差异只来自拒绝，不是另训head或改变身份argmax。

## 成本、完整证据与结论

神经参数为单F0 232993＋冻结G266951＝499944，NN/optimizer更新均0；P0只用F0，P1加G。三个metric共用同一次F/G特征，12列不是12次独立神经前向；Gaussian、投影和geometry为额外FP64聚合状态，不能隐去为“单F成本”。

| path/variant | 共享geometry FP64系数 | metric FP64系数(含投影) | metric数组净字节 |
| --- | --- | --- | --- |
| P0/control | 244273 | 318983 | 2552417 |
| P0/rx | 244273 | 318983 | 2552437 |
| P0/rxday | 244273 | 318983 | 2552485 |
| P1/control | 244273 | 318983 | 2552417 |
| P1/rx | 244273 | 318983 | 2552437 |
| P1/rxday | 244273 | 318983 | 2552449 |

这些是数组净字节，非NPZ压缩文件大小、峰值显存或NN参数；部署不保留source逐行bank。各variant保存自己的Gaussian/scale/projection，仅诊断读数复用，不增加源标签。

| job | 实际PID | driver指派GPU | 整子进程秒 | 终态 |
| --- | --- | --- | --- | --- |
| source_predict | 2849941 | 2 | 135.694866 | COMPLETE |
| source_score | 2851375 | CPU | 3.626930 | COMPLETE |

source内部133.009709s、全pipeline139.404101s，均为含加载/信道/统计/IO的整过程，不是逐包时延或投影加速证据。独立live PID=[2849846, 2849941]，owned GPU采样(pid,MB)=[(2849941, 798), (2849941, 832)]，source日志bytes=[614, 10543, 22165]；采样不是峰值。终态owned/live为空，foreignstop=0；其他owner的GPU列表不计本run。

数据builder实际PID=12828、日志bytes=[515, 1262]，完整18行，终态VERIFIED_COMPLETE、GPU新增0/foreignstop0；3150条完成VALIDATED_ONCE。完整模型run 14份JSON、4条JSONL、4份完整log均解析；12×42校准/评分、known27000×12及novel3150×12分数、两path geometry和六metric完整读取。

新代码组件34个unique tests＝6指标＋6数据＋7core＋13runner/scorer＋2launcher。已有分组件stdout为数据6＋14subtests、core7＋13subtests、runner/scorer13＋14subtests；子项不混成unique，措辞修正后的单项复核不另增unique。数据和性能候选各完成一次独立P0/P1审查；性能审查的6项＋2subtests定向验证也不重复计入34项。一次发布preflight/archive比较/remote compile及真实checkpoint合成2条starIQ smoke见原始记录，未为报告重测或补采资源。

全证据索引：[本轮local artifacts](../local_artifacts/pamf_phase3_star_nuisance_20261029_r01)，`score/results.json::candidates`含12列/42组/逐TX，`prediction/source/{P0,P1}/{control,rx,rxday}/metric.npz`含五分支全诊断，`prediction/source/calibration/complete.json`含42×12校准；`known_predictions.npz`/`novel_predictions.npz`仅为完成预测，报告没有打开原truth、IQ、checkpoint或旧run score。

主rxday/RMD/P1相对匹配control：ΔA_K=0.1778pp、ΔR_U=-4.0952pp、ΔH_OS=-7.4520pp、ΔOSCR=-0.9160pp、ΔAUROC=-1.0847pp。主候选少拒绝129条unknown，仅多正确接受48条known；同path三variant原始closed逐样本完全相同，F0闭集90.9815%不变，+48/27000来自τ和拒绝集合变化，不是编码器或身份判别改善。所有五个source RX及六scene的R_U都低于control。known保护通过，主要失败在开放集分离与拒识，排序也退步；open微增不能抵消。该差值只支持本源域固定G10/已暴露novel的机制判断；不按单项或最高诊断列换主。

结论为NO_PROMOTION。源资格即便通过也不等于跨RX、DA、注册或完整Phase3达成；本轮没有U适应、发现簇、匿名原型或未来Q，A_N/H_OW均为N/A而非0。下一假设尚未实施，当前不改τ、rank、能量比例、候选或seed。
