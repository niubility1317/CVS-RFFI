# Phase3 NUISANCE-MARGINAL 结果（2026-10-31）

**唯一主方法 marginal_predicted_ratio 不晋级（NO_PROMOTION）。** 相对原 predicted_ratio，星地正确接收逐条完全相同，额外接收1条错误身份记录，未知拒对净少3条；R_U/H下降1.041667/.432362个百分点。保留原predicted重点候选与pooled冻结对照，目标ACTIVE。

Run：`20261031-phase3-nuisance-marginal-s2026092701-r01`；实际代码 `93fc96a25b929c534e9b67570d4f2b38bc054c3e`。全四方法、三DURA条件均已固定并完成，没有按Q换主。

## 解析边缘化及权限

原F0/default head、P1 signal r03及DURA全部冻结。将原known90均值按六旧类×15个公共RX/day格排列：每格先去除各类均值，再对六类平均得o_d，取总体协方差C=Σ_d o_d o_dᵀ/15。实测rank14，basis正交误差4.715987e−15；仅从已冻结源聚合推导，不新读L/OE或重新拟合原GMM/normalizer。
每个known及OE背景Gaussian均从vI改为vI+C，解析积分共享源域扰动；两项密度使用同一C。它不硬删除方向或重新投影特征，也不在目标U拟合一个点式δ。没有U/未知TX/K/配额进入先验。
源祖先L6300/OE6134及原F0的L6300/U_s56700/V27000合同保留：L占L+U_s的10%，OE含outside-six成员监督，V是额外已知标签。本轮L/OE/U_s实际读取为0，仅V27000对两个新分数各拟合六τ；两旧模型/τ/calibration/raw/margin及共享身份字段精确复用。
源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，源日为2021-03-08、03-15、03-23。旧六类为14-10、14-7、20-15、20-19、6-15、8-20。NV3150为历史已暴露19-15/19-16/7-1各1050条，仍是已见源RX。clean/star各Q864为RX24-1、2021-03-15相同576 known+288 unknown物理记录，六旧类/三未知类各96条；日期非源未见。
沿既有G10 MIX01缓存：600 km、25 MHz、2.45 GHz、residual/post_sync/noEQ，郊区/城区各0.5，六分量按原覆盖面积概率固定抽样。三主测试都已经过DURA；本轮不新读IQ、不重前向Q。先验在V前冻结，四组τ在NV/Q前全冻，三条件全部预测后才独立连接三份truth。

## 十二个固定运行点

A_K/C_K/R_U分别为逐真实类别宏平均正确接收率、接收率、未知拒识率；H=2A_KR_U/(A_K+R_U)。指标为百分比；F0损失为同输入原F0闭集减A_K（百分点）。

| 条件 | 方法 | A_K | C_K | R_U | H | F0损失 | AUROC | OSCR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| source DURA | pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 | 1.2111 | 94.9853 | 87.9447 |
| clean DURA | pooled_ratio | 68.9236 | 85.4167 | 99.6528 | 81.4874 | 11.1111 | 95.9919 | 77.4637 |
| star DURA | pooled_ratio | 63.0208 | 84.8958 | 76.3889 | 69.0639 | 9.8958 | 84.7283 | 62.6049 |
| source DURA | predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 | 1.2370 | 94.7343 | 87.9376 |
| clean DURA | predicted_ratio | 70.8333 | 87.3264 | 100.0000 | 82.9268 | 9.2014 | 96.1583 | 77.7705 |
| star DURA | predicted_ratio | 64.0625 | 85.5903 | 77.0833 | 69.9723 | 8.8542 | 85.0279 | 62.9823 |
| source DURA | marginal_pooled_ratio | 89.8037 | 97.1889 | 59.8095 | 71.8000 | 1.1778 | 94.9912 | 87.9419 |
| clean DURA | marginal_pooled_ratio | 69.4444 | 86.1111 | 99.6528 | 81.8503 | 10.5903 | 96.1625 | 77.6476 |
| star DURA | marginal_pooled_ratio | 63.8889 | 86.2847 | 74.3056 | 68.7046 | 9.0278 | 85.0019 | 62.8430 |
| source DURA | **marginal_predicted_ratio** | 89.7630 | 96.7963 | 61.0794 | 72.6940 | 1.2185 | 94.7013 | 87.9111 |
| clean DURA | **marginal_predicted_ratio** | 71.3542 | 87.8472 | 100.0000 | 83.2827 | 8.6806 | 96.1607 | 77.7585 |
| star DURA | **marginal_predicted_ratio** | 64.0625 | 85.7639 | 76.0417 | 69.5400 | 8.8542 | 85.0568 | 63.0076 |

原F0闭集source/clean/star始终为90.981481/80.034722/72.916667%，没有身份判别改变。主clean虽拒对全部288未知，C_K仅87.847222%、身份损失8.680556pp；star的C_K85.763889%、R_U76.041667%、身份损失8.854167pp均未过。source的R_U61.079365%只作开发诊断。

## 同ID配对

下表为主方法减固定控制，gain/loss/net保留正确接收与未知拒对的完整事件差；Δ为百分点。

| 控制 | 条件 | 正确接收gain/loss/net | 接收net | 未知拒对gain/loss/net | ΔA_K / ΔR_U / ΔH |
|---|---|---:|---:|---:|---:|
| predicted | source | 12/7/+5 | +10 | 9/36/-27 | +0.018519 / -0.857143 / -0.597455 |
| predicted | clean | 4/1/+3 | +3 | 0/0/0 | +0.520833 / +0.000000 / +0.355846 |
| predicted | star | 0/0/0 | +1 | 1/4/-3 | +0.000000 / -1.041667 / -0.432362 |
| pooled | source | 31/33/-2 | -84 | 75/69/+6 | -0.007407 / +0.190476 / +0.132653 |
| pooled | clean | 15/1/+14 | +14 | 1/0/+1 | +2.430556 / +0.347222 / +1.795233 |
| pooled | star | 8/2/+6 | +5 | 4/5/-1 | +1.041667 / -0.347222 / +0.476036 |
| marginal_pool | source | 22/33/-11 | -106 | 80/40/+40 | -0.040741 / +1.269841 / +0.893957 |
| marginal_pool | clean | 12/1/+11 | +10 | 1/0/+1 | +1.909722 / +0.347222 / +1.432344 |
| marginal_pool | star | 4/3/+1 | -3 | 7/2/+5 | +0.173611 / +1.736111 / +0.835329 |

主star对原predicted正确0增0失，接收+1完全是错误身份接收；未知1增4失/net−3。clean净多3条正确且未知不变，仍未满足保护。marginal pooled的局部A_K收益不能替换唯一主，也不能补救其拒识损失；两新方法分别按同一合法V规则校准，差异含分数和运行点共同变化。
接收不等于身份正确：主clean接收506但正确411，另50条原本正确记录被拒；star接收494但正确369，另51条原本正确被拒。原闭集本身的错误与拒识新增损伤须分开，不能降低原F0基线遮盖。

## 分组与覆盖

四方法source known12均通过，known42均未通过。主有五个RX×scene诊断组超过2pp；现存next-q绑定RX3，下一点损2.018519pp。本轮hard12与30个cross诊断明确分开，且V本身用于校准，不称独立确认或分布外保证。

| 星地主场景 | known/unknown | A_K | C_K | R_U | F0损失（pp） |
|---|---:|---:|---:|---:|---:|
| practical_high | 33/6 | 72.8571 | 93.3333 | N/A | 3.3333 |
| practical_mid | 42/25 | 69.7222 | 87.2222 | 91.6667 | 12.7778 |
| practical_low_suburban | 203/117 | 66.2404 | 86.6372 | 87.0965 | 9.8848 |
| practical_high_urban | 38/16 | 75.3307 | 91.0714 | 86.6667 | 6.5476 |
| practical_mid_urban | 45/15 | 75.5093 | 94.4444 | 81.1111 | 5.5556 |
| practical_low_urban | 215/109 | 54.2402 | 80.1846 | 57.1640 | 9.2894 |

high未知三类数为0/4/2，缺19-15，宏R_U/H及曲线为N/A，不能以6/6 pooled拒对补认证。六景身份损失均超2pp，low urban宏R_U仅57.163978%。主对19-15/19-16/7-1源拒对672/517/735（各1050），clean96/96/96，star74/70/75（各96）。
所有方法的TX/RX/day/scene、42组、覆盖和三项完整配对保留于原score及analysis；逐真实类Wilson区间仅描述当前样本，不把重复方法/条件当独立物理样本或未见RX证明。

## 成本与实际证据

本轮无训练steps：新增IQ、真实F0/G/signal前向、NN更新均0。实际三个checkpoint合成烟测各2行后立即继续；从源聚合导出rank14先验一次，再拟合两个新六τ。原GMM/normalizer不变，不把“无NN更新”写成“无新增统计与校准”。
prior仅三数组：FP64 basis14×160为17920 B、variances14为112 B及schema172 B，合计18204 B（NPZ18962 B）；原density11数组364340 B，全24τ192 B。主部署只需一组六τ48 B，无逐源行bank。F0232993+signal334721+G266951=834665神经参数及原F011100/r03增量2220、投影2220、G4940和既有缓存/GMM成本全部继承。
predict25.616272 s、独立CPU score14.970574 s、pipeline40.674194 s，均COMPLETE/returncode0，pipeline ANALYZED_DEVELOPMENT；这是整进程成本，不是B1单包时延。
四次观测独立捕获driver1180081及producer1180175的CWD/fullargv/CUDA7，GPU UUID9c4eb614两次406 MiB离散样本；两次live时predict.log均512 B，末态674 B，不能虚报live期间增长或显存峰值。CPU1180467仅pipeline gpuNone/代码CUDAempty/完成产物，未独立捕获live；终态live/ownedGPU空、foreign0。
三份完整运行stdout共742 B/7行，无warning/error/traceback/非有限提示。13unique+34sub PASS、唯一P0/P1 review PASS；测试中的mock缺reference_config已仅定向修复一次，1792条旧np.trapz警告只属测试，未重复已过用例。
单包636files/data0，SHA `68d9775c6a0de76f69b9e2ecfdaeb1cce1f82edb885436f9cf6c8ac3675b4aee` 比较、compile/runtime VERIFIED。本报告仅整理既有产物，没有新fit、score、阈值扫描或遥测补跑。

源RX已见、NV/Q历史曝光、单seed/单开发RX与high缺类边界保持；U、正式RX24-18访问为0，无DA/发现/匿名REG，A_N/H_OW=N/A。本次解析源nuisance边缘化未取得既定联合收益，原predicted/pooled保留，完整目标ACTIVE。

证据：[score/results.json](../local_artifacts/pamf_phase3_nuisance_marginal_20261031_r01/score/results.json)、[analysis.json](../local_artifacts/pamf_phase3_nuisance_marginal_20261031_r01/analysis.json)、[pipeline](../local_artifacts/pamf_phase3_nuisance_marginal_20261031_r01/pipeline_state.json)、[本run报告](../automation_reports/CV-SincNet/20261031-phase3-nuisance-marginal-s2026092701-r01/report.md)。
