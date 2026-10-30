# Phase3 DENSITY-BOUNDARY 结果（2026-10-31）

**唯一主方法 BOUNDARY_predicted 不晋级（NO_PROMOTION）。** 相对同预算 REPLAY，边界损失明显提高星地拒识，但正确接收仍下降；相对原 E0，联合表现未胜出。保留原 E0_predicted 重点候选和 E0_pooled 冻结对照，全目标 ACTIVE。

Run：`20261031-phase3-density-boundary-s2026092701-r01`；实际代码 `e3194bc6ed46ce1580967f036a35fcb138e1c906`。两臂及六方法×三 DURA 条件均固定，没有按源 NV/Q 换主。

## 方法与数据

REPLAY、BOUNDARY 分别从同一 P1 signal r03 独立复制，fresh CENTER_UNITS 各更新 2220 步；原 F0/default head、DURA、r03 教师、共享 GMM11 和 point teacher 不更新。两臂共有 source crossset/OE topology 与 point anchor，BOUNDARY 另加固定条件 GMM 的成员 BCE；REPLAY 的同项仅记录诊断、权重为 0。
每步目标为 source_loss+point_anchor+boundary_weight×boundary_loss。条件 log density ratio 先除以 160，再分别计算 L 正成员/OE 负成员 BCE，两组各半；不是将 BCE 输出再除 160。路由类别来自原 F0 对源缓存 u160 的 head-only 预测，不能换成真实 L y。原 L 标签仅用于既有合法源监督，OE 不使用真实 TX/K。
固定 source contract split392005：L6300/U_s56700/V27000，L 占 L+U_s 的 10%，V 是额外校准标签。训练仅读 L6300/OE6134 的既有 G10 MIX01/P1；OE6134 提供 outside-six 成员监督。源 RX1/3/4/6/8 对应1-19/18-2/19-2/2-19/3-19，日期为2021-03-08、03-15、03-23。
NV3150 为历史已暴露的19-15/19-16/7-1各1050条，仍来自已见源RX。clean/star Q各864，均为RX24-1、2021-03-15同一576 known+288 unknown物理记录，六旧类/三未知类各96条；日期与源重合，不是严格未见日期。三主测试均经DURA，训练无clean/Q/U。
训练不读取旧τ/cal/V/NV/Q。全部神经状态冻结后，V仅拟合四组新六τ，E0两组旧τ精确复用；GMM及normalizer均不重拟合。E0 source/Q的z、两旧ratio和共享身份字段精确复用，全部预测固定后才独立连接三份truth。

## 完整训练轨迹

已完整读取4440条steps：每臂1至2220连续、全部数值有限，L192/OE64逐步索引完全匹配且每批有效无重复；两臂实际各覆盖L6300、OE6134。每臂L曝光426240、OE曝光142080，总学生signal前向1136640；这是重复计算曝光，不是新增物理样本。source_loss从step2首次分歧。
下表为全部2220步均值及四个连续555步区间均值，未用首尾样本替代全程。BOUNDARY含额外BCE，不能按两臂total loss高低直接比较训练优劣。

| 臂/分量 | 全程均值 | 1–555 | 556–1110 | 1111–1665 | 1666–2220 |
|---|---:|---:|---:|---:|---:|
| REPLAY/total | 0.280827 | 0.342201 | 0.295600 | 0.259515 | 0.225994 |
| REPLAY/source | 0.272455 | 0.337004 | 0.288602 | 0.250274 | 0.213939 |
| REPLAY/anchor | 0.008373 | 0.005196 | 0.006998 | 0.009241 | 0.012055 |
| REPLAY/BCE（REPLAY权重0） | 0.369810 | 0.372511 | 0.370481 | 0.368759 | 0.367490 |
| BOUNDARY/total | 0.541256 | 0.622169 | 0.557190 | 0.513035 | 0.472629 |
| BOUNDARY/source | 0.278861 | 0.341762 | 0.294981 | 0.257593 | 0.221108 |
| BOUNDARY/anchor | 0.022558 | 0.016716 | 0.021385 | 0.024437 | 0.027693 |
| BOUNDARY/BCE | 0.239837 | 0.263691 | 0.240824 | 0.231005 | 0.223827 |

总损失起末：REPLAY .356812→.226495，BOUNDARY .731422→.473986。末步BCE为主.212239、REPLAY诊断.352864；末步anchor为.029158与.011705。总目标方程误差≤2.99e−8，anchor分量误差≤1.87e−9，BCE两组均值公式误差0。训练代理确有学习，较大anchor表示离教师更远，不单独证明泛化改善或失败因果。
Sinc/浅层/中层/时频/PA/参考/签名/MLP hidden/last九模块仅step1/2记录梯度，观察到均非零，不能推广为全2220步逐模块证明。全程global physical norm有限；REPLAY/BOUNDARY均值4.562630/4.944999，最大14.332270/16.141039，超过clip5的步数674/863。两臂每步Sinc与physical clip计数正确至2220；AdamW lr1e−4、wd1e−4、Sinc lr.1，student buffers保持不变。

## 十八个固定运行点

A_K/C_K/R_U为逐真实类别宏平均正确接收率、接收率和未知拒识率，H=2A_KR_U/(A_K+R_U)。指标均为百分比，F0损失为同输入原闭集减A_K（百分点）。

| 条件 | 方法 | A_K | C_K | R_U | H | F0损失 | AUROC | OSCR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| source DURA | E0_pooled | 89.7704 | 97.1074 | 60.8889 | 72.5613 | 1.2111 | 94.9853 | 87.9447 |
| clean DURA | E0_pooled | 68.9236 | 85.4167 | 99.6528 | 81.4874 | 11.1111 | 95.9919 | 77.4637 |
| star DURA | E0_pooled | 63.0208 | 84.8958 | 76.3889 | 69.0639 | 9.8958 | 84.7283 | 62.6049 |
| source DURA | E0_predicted | 89.7444 | 96.7593 | 61.9365 | 73.2914 | 1.2370 | 94.7343 | 87.9376 |
| clean DURA | E0_predicted | 70.8333 | 87.3264 | 100.0000 | 82.9268 | 9.2014 | 96.1583 | 77.7705 |
| star DURA | E0_predicted | 64.0625 | 85.5903 | 77.0833 | 69.9723 | 8.8542 | 85.0279 | 62.9823 |
| source DURA | REPLAY_pooled | 89.6519 | 96.8926 | 56.8889 | 69.6079 | 1.3296 | 94.3705 | 87.5446 |
| clean DURA | REPLAY_pooled | 73.6111 | 90.6250 | 97.5694 | 83.9137 | 6.4236 | 95.7706 | 77.3703 |
| star DURA | REPLAY_pooled | 63.3681 | 85.7639 | 55.9028 | 59.4018 | 9.5486 | 80.1595 | 59.3346 |
| source DURA | REPLAY_predicted | 89.6259 | 96.4704 | 56.8571 | 69.5763 | 1.3556 | 94.0420 | 87.5643 |
| clean DURA | REPLAY_predicted | 73.9583 | 90.7986 | 98.2639 | 84.3960 | 6.0764 | 95.6386 | 77.2509 |
| star DURA | REPLAY_predicted | 63.1944 | 84.3750 | 63.8889 | 63.5398 | 9.7222 | 80.9172 | 60.0899 |
| source DURA | BOUNDARY_pooled | 89.6259 | 97.0407 | 60.6032 | 72.3111 | 1.3556 | 88.8177 | 82.2158 |
| clean DURA | BOUNDARY_pooled | 73.6111 | 90.6250 | 88.5417 | 80.3890 | 6.4236 | 92.7294 | 75.6764 |
| star DURA | BOUNDARY_pooled | 62.8472 | 84.7222 | 75.6944 | 68.6752 | 10.0694 | 80.8419 | 60.4962 |
| source DURA | **BOUNDARY_predicted** | 89.6111 | 96.6963 | 62.7619 | 73.8210 | 1.3704 | 89.4521 | 83.0283 |
| clean DURA | **BOUNDARY_predicted** | 74.6528 | 91.6667 | 97.2222 | 84.4557 | 5.3819 | 96.3469 | 78.5777 |
| star DURA | **BOUNDARY_predicted** | 62.6736 | 84.0278 | 78.1250 | 69.5515 | 10.2431 | 82.8728 | 61.9617 |

原F0闭集始终为source90.981481%、clean80.034722%、star72.916667%，身份预测没有改变。主clean虽C_K91.666667%、R_U97.222222%分别过单项，原闭集至A_K仍损5.381944pp；star损10.243056pp，C_K84.027778%、R_U78.125%均失败。

## 同ID配对

表内为主方法减对应控制；gain/loss/net分别保留事件增加、丢失和净数，差值单位为百分点。

| 控制 | 条件 | 正确接收gain/loss/net | 接收net | 未知拒对gain/loss/net | ΔA_K / ΔR_U / ΔH |
|---|---|---:|---:|---:|---:|
| E0_pred | source | 91/127/-36 | -17 | 224/198/+26 | -0.133333 / +0.825397 / +0.529560 |
| E0_pred | clean | 27/5/+22 | +25 | 0/8/-8 | +3.819444 / -2.777778 / +1.528839 |
| E0_pred | star | 14/22/-8 | -9 | 12/9/+3 | -1.388889 / +1.041667 / -0.420845 |
| REPLAY_pred | source | 95/99/-4 | +61 | 339/153/+186 | -0.014815 / +5.904762 / +4.244710 |
| REPLAY_pred | clean | 13/9/+4 | +5 | 5/8/-3 | +0.694444 / -1.041667 / +0.059666 |
| REPLAY_pred | star | 13/16/-3 | -2 | 49/8/+41 | -0.520833 / +14.236111 / +6.011710 |
| BOUNDARY_pool | source | 30/34/-4 | -93 | 125/57/+68 | -0.014815 / +2.158730 / +1.509899 |
| BOUNDARY_pool | clean | 8/2/+6 | +6 | 25/0/+25 | +1.041667 / +8.680556 / +4.066660 |
| BOUNDARY_pool | star | 4/5/-1 | -4 | 8/1/+7 | -0.173611 / +2.430556 / +0.876320 |

相对REPLAY，星地R_U/H增加14.236111/6.011710pp，但A_K少.520833pp；相对E0，R_U仅多1.041667pp，A_K少1.388889pp、H少.420845pp。边界项有匹配的拒识效应，却没有胜过旧基线联合性能，不能以较高clean成绩或控制救主。
源运行点R_U较E0仅增.825397pp，宏AUROC反降5.282118pp；源L/OE的BCE下降不能当作历史NV分离排序改善。接收也不是身份正确：主clean接收528但正确430，另31条原本正确被拒；star接收484但正确361，另59条原本正确被拒。14-7在star接收96但仅正确32。

## 分组与覆盖

六方法source known12均过、known42均未过。主有六个RX×scene诊断组超2pp；现存next-q绑定low urban，下一点损2.000659pp，未据未知/Q另选阈值。source R_U仍仅62.761905%，仅为诊断；不因源失败取消既定跨RX矩阵。

| 星地主场景 | known/unknown | A_K | C_K | R_U | F0损失（pp） |
|---|---:|---:|---:|---:|---:|
| practical_high | 33/6 | 63.8095 | 84.2857 | N/A | 12.3810 |
| practical_mid | 42/25 | 70.2778 | 87.7778 | 91.6667 | 12.2222 |
| practical_low_suburban | 203/117 | 68.3180 | 87.7426 | 90.9816 | 7.8072 |
| practical_high_urban | 38/16 | 70.5688 | 86.3095 | 86.6667 | 11.3095 |
| practical_mid_urban | 45/15 | 72.5992 | 91.5344 | 81.1111 | 8.4656 |
| practical_low_urban | 215/109 | 51.1249 | 77.1504 | 57.4059 | 12.4046 |

high未知三类数0/4/2，缺19-15，宏R_U/H/曲线为N/A，不能用6/6 pooled拒对补认证。六景F0身份损失均超2pp；low urban宏R_U仅57.405914%。主对19-15/19-16/7-1源拒对650/538/789（各1050），clean93/96/91，star76/71/78（各96）。
全部18点的42组、TX/RX/day/scene、覆盖和逐真实类Wilson区间及三项完整配对见原score与analysis。单开发RX/单seed、同物理条件重复和源NV/Q历史曝光均不能当成独立确认；V还曾用于阈值校准。

## 成本与执行证据

两新完整signal各更新2220，不是仅MLP训练：每个复制特征累计15540，投影累计4440；原F0仍11100。单主F0232993+open334721+G266951=834665神经参数；三open加F0/G的架构活跃参数合计1504107，不是全native实际驻留内存测量。固定GMM11数组364340 B共享、全部36τ288 B；teacher逐行bank不部署且未fetch。
真实训练学生signal1136640＋parentreference12434；源F0 head-only12434单列，不能冒充12434次完整backbone（该阶段backbone/G为0）。eval新增signal63756=2×30150+2×1728、F0/G各1728，E0新真实前向0；原训练和缓存/Q E0成本继续继承。
合成烟测单列：train原parent open/F0/G各2，student gradient16；eval三open合计6、F0/G各2，均无optimizer更新。神经训练后新密度/normalizer fit0，仅四新六τ由V拟合；不得把新NN4440更新与旧GMM冻结混为“零新训练”。
训练462.670735 s、predict26.080005 s、独立CPU score16.561103 s，pipeline505.483873 s；三阶段COMPLETE/returncode0。时间为完整进程成本，不是B1单包时延。
六次观测独立捕获driver1155253和train1155342的CWD/fullargv/CUDA4，GPU UUID95d925a1离散样本758 MiB；train.log670→1345→2405→3098→4016 B。不是显存峰值。predict1161117/GPU4和CPU1161391均未捕获独立live，仅有pipeline分配/终态与代码CUDAempty证据；末态live/ownedGPU空，foreign0。
四份完整运行stdout共4754 B/61行，无warning/error/traceback/非有限提示。23unique+34sub PASS、唯一P0/P1 review PASS；旧np.trapz警告仅在持久测试stdout。单包627files/data0，SHA `7c3a16c9797125c6e198a9dc1679419d413dcb858a08c5a09bc558172b6192f9` 比较、compile/runtime VERIFIED。

本轮U0、正式RX24-18访问0，无DA/发现/匿名REG，A_N/H_OW=N/A。source-only新边界目标具有匹配控制拒识效应，但未过既定综合门槛，也不完成Phase3；原E0_predicted/E0_pooled保留，目标ACTIVE。

完整证据：[score/results.json](../local_artifacts/pamf_phase3_density_boundary_20261031_r01/score/results.json)、[analysis.json](../local_artifacts/pamf_phase3_density_boundary_20261031_r01/analysis.json)、[pipeline](../local_artifacts/pamf_phase3_density_boundary_20261031_r01/pipeline_state.json)、[本run报告](../automation_reports/CV-SincNet/20261031-phase3-density-boundary-s2026092701-r01/report.md)。
