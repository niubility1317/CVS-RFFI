# Phase3 同记录 received／恢复信号开放表征结果（2026-10-31）

**结论：NO_PROMOTION，原 PCD 研究基线保留，完整目标 ACTIVE。** 唯一主 TARGET_PAIR_predicted_ratio 的星地 AK/CK/RU/H 为 27.910624/48.533058/49.552823/35.708460%。相对旧 PCD，AK 下降 0.041098 个百分点、RU 增加 0.245125 个百分点，未满足 AK 严格提高要求；相对匹配 SOURCE_PAIR，AK 增加 1.870704 个百分点，却以 RU 下降 1.837017 个百分点为代价。跨接收机绝对目标均未通过，不换 pooled、不扩 seed。

Run：20261031-phase3-paired-signal-s2026092701-r01；实际代码 00e86deb914ff02f54fe57c101491220aee0a1a0。N607 train/predict/score 全部 COMPLETE/return0，pipeline 为 ANALYZED_DEVELOPMENT；科学判定与执行成功分开。

## 本轮改变了什么

三个学生独立从纯源 r03 E_SIGNAL2220 初始化，不从已适配 PCD 或另一源学生续训。一个 SignalOpen334721 共享权重、对两个输入槽各前向一次，得到单位 z0/z1；并非两个独立编码器。新 adapter 为含 bias 的零初始化 Linear320→160，共 51360 参数：

`delta = 0.1/sqrt(160) × tanh(Linear(concat[z0−z1, z0×z1])); z = normalize(z1+delta)`。

delta 的数学 L2 上界为 0.1；初始等价于 P1 表示，但 FP32 再单位化不宣称逐 bit 相同。原 F0/default6、r03、历史 PCD、DURA 不改；仅新学生的非 Sinc body、projection 与 adapter 更新，Sinc48 和 buffers 冻结。原 F0 身份始终只接收 P1，配对信息只改变开放表征，不能纠正其闭集误分。

| 新臂 | 两输入槽 | 源训练 | 目标 U | 科学比较用途 |
|---|---|---|---|---|
| SOURCE_P1 | P1/P1，模型内部屏蔽 P0 | L192/OE64，444 步 | 0 | 同结构 P1 控制 |
| SOURCE_PAIR | 同一记录 P0/P1 | 同源索引，444 步 | 0 | 对 SOURCE_P1 检验配对输入作用 |
| TARGET_PAIR，唯一主 | 同一记录 P0/P1 | 同源索引，444 步 | U576、原 PCD 目标 | 对 SOURCE_PAIR 检验适应作用 |

源目标为 crossset metric＋OE topology KL＋10×原 r03 源 P1 z 锚定。SOURCE_P1 与 SOURCE_PAIR 先封存，随后才打开 U。它们名义结构与两次 base 前向相同，但 SOURCE_P1 的有效输入信息只有 P1；不声称信息容量完全相同。

TARGET_PAIR 保持已有 PCD global/local：原 r03 parent_z 初始化16容量原型与固定非 self 邻居；eps0.05、20轮 log Sinkhorn 得到停止梯度分配，学生两强 U64 以温度0.1预测自身及邻居分配，global/local各乘0.1。16不是未知真实K；仍含固定邻接假设，本轮没有声称修复该假设，没有 EMA、真目标 TX 或 query 早停。

U P0 取同一 public received，P1及其强视图复用 fresh 缓存；对两槽同步重放 phase[-π,π]/gain[0.95,1.05]（seed2026103114），重放 P1 与原强视图一致。新消费 P0 不等于第二次物理观测或重抽信道，也不使用 clean 恢复参考。具体算法、冻结顺序与合法句柄见[预登记计划](PAMF_PHASE3_PAIRED_SIGNAL_PLAN_20261031.md)。

## 数据、校准与测试边界

源合同392005：L6300、OE6134用于源重放及冻结后的密度，Us56700仅祖先、本轮读取0；源五 RX 为1-19/18-2/19-2/2-19/3-19（索引1/3/4/6/8），日期2021-03-08/15/23。六旧 TX 为14-10/14-7/20-15/20-19/6-15/8-20，L真实标签可用；OE只有 outside-six 成员监督与公共RX/day，没有真实TX/K。

V27000只在学生与统计冻结后校准阈值，未训练神经或normalizer；这是额外已知校准标签，不把源V称独立已知测试。NV3150为19-15/19-16/7-1各1050，只冻结预测与独立评分。新源统计输入合法不使含目标U祖先的 TARGET_PAIR 变成纯源系统。

目标开发 RX3-2/index34：03-01星地U576；未来03-15/23的Q5525/4713，共10238条物理记录，旧4508／四未注册5730。旧六类分别1491/1182/150/406/1186/93，5-5/10-7/2-19/7-11分别1454/1439/1427/1410。后来专有7-11整类不在U；真实早到/后来角色、类别数和配额仅builder/scorer可见。源/U/Q物理互斥，clean/star使用相同canonical Q及顺序。

三条件全部先过固定DURA v1R69（E20/4940、266951参数、reuse/graph、非native、FP32/TF32off）；F0闭集保护始终同P1。源/未来G10为600 km、25 MHz、2.45 GHz、residual/post_sync/noEQ/refSNR30 dB/reflect、一次input RMS且输出无RMS；源scene/channel seed2026103001/2026104001，未来2026103133/34，receiver2027。clean仅冻结测试，没有clean U或学习。

新三学生各仅源L/OE拟合known90/OE32、L残差白化、固定10%收缩及背景32容量/25次Lloyd/seed2026103019。随后V拟合6组新六τ（36值），旧SOURCE与OLD_PCD四组24值精确复用，总60个FP64/480 B。全部模型先于V、全部τ先于NV/Q冻结；十读出×三条件完整后CPU才连接NV/cleanQ/starQ三truth各一次，Utruth0。

## 全部 1332 步训练

完整解析 SOURCE_P1/SOURCE_PAIR/TARGET_PAIR 各1至444步，连续、全数值有限，非stdout尾部抽样。训练端在每臂逐步检查旧SOURCE444流；本地分析独立核对三条新L/OE流完全相同，没有再次读取旧SOURCE步文件。TARGET U采样与固定seed2026103112一致。每臂L曝光85248、OE28416，unique L6300/OE5913（OE池中221条未抽到）；两源臂U0，TARGET U64×444=28416次、unique576，双强视图56832条次，另每步全U弱前向255744条次。

actual metadata/实现保证两个源臂完成先于U打开；stdout也显示两个源完成行在首次TARGET进度前，但没有独立的U-open时间戳，不把进度文本当该IO时刻的直接遥测。三臂均 AdamW lr1e−4/wd1e−4/eps1e−8、ordinary norm clip5，各444次optimizer更新；不串接状态。

以下为全部步骤连续三段均值；anchor为乘10前，global/local为各乘0.1前。源臂没有U项，三个总loss不能直接按大小解释科学优劣。

| 臂 | 步段 | 总loss | source | crossset | OE KL | anchor | global | local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SOURCE_P1 | 1–148 | 0.36874061 | 0.36874061 | 0.28713623 | 0.06335827 | 0.00182461 | 0.00000000 | 0.00000000 |
| SOURCE_P1 | 149–296 | 0.36449791 | 0.36449791 | 0.28508389 | 0.06172662 | 0.00176874 | 0.00000000 | 0.00000000 |
| SOURCE_P1 | 297–444 | 0.35380955 | 0.35380955 | 0.27378438 | 0.05977965 | 0.00202455 | 0.00000000 | 0.00000000 |
| SOURCE_PAIR | 1–148 | 0.36873103 | 0.36873103 | 0.28712504 | 0.06334670 | 0.00182593 | 0.00000000 | 0.00000000 |
| SOURCE_PAIR | 149–296 | 0.36442845 | 0.36442845 | 0.28504389 | 0.06170665 | 0.00176779 | 0.00000000 | 0.00000000 |
| SOURCE_PAIR | 297–444 | 0.35377427 | 0.35377427 | 0.27370816 | 0.05979017 | 0.00202759 | 0.00000000 | 0.00000000 |
| TARGET_PAIR | 1–148 | 0.92863593 | 0.38551261 | 0.28896593 | 0.06669990 | 0.00298468 | 2.70326533 | 2.72796771 |
| TARGET_PAIR | 149–296 | 0.83693964 | 0.38351107 | 0.28821963 | 0.06465177 | 0.00306397 | 2.26425992 | 2.27002571 |
| TARGET_PAIR | 297–444 | 0.79670605 | 0.37353085 | 0.27786071 | 0.06243931 | 0.00332308 | 2.11817709 | 2.11357486 |

源项重建最大误差三臂依次为5.75e−8/5.96e−8/3.77e−8；TARGET总目标重建误差≤8.35e−8，源臂为0。全程总loss范围为SOURCE_P1 0.203821至0.593978、SOURCE_PAIR 0.203997至0.594078、TARGET 0.636065至1.171822。TARGET global/local三段均值下降，说明训练目标被降低，不等于目标设备识别或拒识提高。

| 臂 | 梯度范数全程范围 | 超过5的步数/裁剪调用 | 已记录最大delta L2 | 末步最大delta L2 |
|---|---|---:|---:|---:|
| SOURCE_P1 | 2.499472至10.836473 | 165/444 | 0.000801635 | 0.000777859 |
| SOURCE_PAIR | 2.505825至10.813035 | 167/444 | 0.003640871 | 0.003152985 |
| TARGET_PAIR | 2.575423至12.658208 | 199/444 | 0.004867048 | 0.004434829 |

两个源臂首步总loss相同，第2步起标量轨迹分开；TARGET从第1步加入U项。三臂全部adapter梯度非零，delta首步0、随后443步非零；各444份模块梯度记录中Sinc为0，浅/中层、时频、PA、reference、signature、projection均有非零梯度。目标原型梯度只在TARGET存在；原F0/default头、DURA、r03和旧PCD参数/缓冲保持不变。

新body/projection/adapter均实际更新，Sinc/native/default-head更新0，physical_clip_calls=0。残差全程最大仅0.000802/0.003641/0.004867，均远低于0.1数学上界；ordinary梯度裁剪与残差界是两件事。本轮没有发生“物理残差被裁到上界”的证据，也不能据小残差认定上界阻碍收益或据Q增加界。

## 完整十读出 × 三条件

以下为逐真实TX等权的百分比；AK为旧类正确且接收，CK为旧类接收；源RU使用三类NV，目标RU使用全部四未注册类，H为相应AK/RU调和均值。主方法固定，不按表中最大项替换。SOURCE、OLD_PCD是精确复用历史控制；新三臂的统计和源阈值各自独立合法拟合。

| 方法 | 条件 | AK | CK | RU | H |
|---|---|---:|---:|---:|---:|
| SOURCE_pooled_ratio | 源DURA | 89.777778 | 97.088889 | 60.063492 | 71.974388 |
| SOURCE_pooled_ratio | clean＋DURA | 31.475324 | 44.146095 | 62.409528 | 41.846157 |
| SOURCE_pooled_ratio | 星地＋DURA | 27.422108 | 48.651740 | 51.188933 | 35.712755 |
| SOURCE_predicted_ratio | 源DURA | 89.740741 | 96.744444 | 61.174603 | 72.754089 |
| SOURCE_predicted_ratio | clean＋DURA | 31.078517 | 43.181373 | 63.911653 | 41.820736 |
| SOURCE_predicted_ratio | 星地＋DURA | 26.751337 | 47.266339 | 52.269979 | 35.390244 |
| OLD_PCD_pooled_ratio | 源DURA | 89.770370 | 97.081481 | 63.619048 | 74.465443 |
| OLD_PCD_pooled_ratio | clean＋DURA | 32.123647 | 44.791196 | 60.790839 | 42.034854 |
| OLD_PCD_pooled_ratio | 星地＋DURA | 29.175335 | 50.230381 | 48.454372 | 36.420916 |
| OLD_PCD_predicted_ratio | 源DURA | 89.733333 | 96.748148 | 64.285714 | 74.907247 |
| OLD_PCD_predicted_ratio | clean＋DURA | 31.661804 | 43.761391 | 61.979647 | 41.912794 |
| OLD_PCD_predicted_ratio | 星地＋DURA | 27.951722 | 48.582412 | 49.307698 | 35.678111 |
| SOURCE_P1_pooled_ratio | 源DURA | 89.803704 | 97.181481 | 59.206349 | 71.363634 |
| SOURCE_P1_pooled_ratio | clean＋DURA | 30.956037 | 43.749539 | 61.421038 | 41.165017 |
| SOURCE_P1_pooled_ratio | 星地＋DURA | 27.689342 | 49.358446 | 49.361128 | 35.477451 |
| SOURCE_P1_predicted_ratio | 源DURA | 89.755556 | 96.770370 | 61.619048 | 73.072388 |
| SOURCE_P1_predicted_ratio | clean＋DURA | 30.470618 | 42.545321 | 64.061201 | 41.297934 |
| SOURCE_P1_predicted_ratio | 星地＋DURA | 26.051146 | 46.780614 | 51.285066 | 34.551336 |
| SOURCE_PAIR_pooled_ratio | 源DURA | 89.792593 | 97.155556 | 59.523810 | 71.590222 |
| SOURCE_PAIR_pooled_ratio | clean＋DURA | 30.970138 | 43.774818 | 61.665269 | 41.232223 |
| SOURCE_PAIR_pooled_ratio | 星地＋DURA | 27.623012 | 49.227459 | 49.553858 | 35.472462 |
| SOURCE_PAIR_predicted_ratio | 源DURA | 89.740741 | 96.725926 | 61.809524 | 73.201224 |
| SOURCE_PAIR_predicted_ratio | clean＋DURA | 30.291407 | 42.366109 | 64.060942 | 41.132968 |
| SOURCE_PAIR_predicted_ratio | 星地＋DURA | 26.039920 | 46.641302 | 51.389840 | 34.565194 |
| TARGET_PAIR_pooled_ratio | 源DURA | 89.774074 | 97.085185 | 63.460317 | 74.357867 |
| TARGET_PAIR_pooled_ratio | clean＋DURA | 32.179858 | 44.888458 | 60.491402 | 42.010969 |
| TARGET_PAIR_pooled_ratio | 星地＋DURA | 29.189388 | 50.258581 | 48.244726 | 36.372445 |
| TARGET_PAIR_predicted_ratio | 源DURA | 89.711111 | 96.692593 | 64.888889 | 75.307300 |
| TARGET_PAIR_predicted_ratio | clean＋DURA | 31.689910 | 43.802347 | 62.134576 | 41.972819 |
| TARGET_PAIR_predicted_ratio | 星地＋DURA | 27.910624 | 48.533058 | 49.552823 | 35.708460 |

原F0同DURA闭集三条件为90.981481/48.567015/41.721059%，所有读出完全相同。主AK损伤分别1.270370/16.877106/13.810435 pp。clean原闭集正确2142条、正确接收1649（正确误拒493）；星地闭集正确1944、正确接收1563（正确误拒381），已知错误接收1068。输入配对没有改动F0身份，不能把拒识损失与身份误分混合解释。

主AUROC/OSCR为源94.727667/87.933025%、clean54.252508/33.617506%、星地50.860843/26.890247%。相对旧PCD星地50.997190/27.001261%并无曲线改善；这是固定分数的描述，不导出新运行点或扫描τ。

## 分开解释输入效应与目标适应

主TARGET predicted相对两项预定predicted控制的同ID事件如下；净条数不是TX宏变化，尤其本Q每类容量不等。

| 控制 | 条件 | 正确接收增/失 | 接收净增 | 未知拒对增/失 | ΔAK / ΔRU / ΔH（百分点） |
|---|---|---:|---:|---:|---|
| SOURCE_PAIR | 源DURA | 20/28 | -9 | 167/70 | -0.029630 / +3.079365 / +2.106076 |
| SOURCE_PAIR | clean＋DURA | 19/17 | -2 | 64/173 | +1.398503 / -1.926366 / +0.839852 |
| SOURCE_PAIR | 星地＋DURA | 26/16 | +13 | 48/152 | +1.870704 / -1.837017 / +1.143267 |
| OLD_PCD | 源DURA | 2/8 | -15 | 22/3 | -0.022222 / +0.603175 / +0.400053 |
| OLD_PCD | clean＋DURA | 2/0 | +1 | 20/11 | +0.028106 / +0.154928 / +0.060025 |
| OLD_PCD | 星地＋DURA | 1/2 | -2 | 18/4 | -0.041098 / +0.245125 / +0.030349 |

匹配的 TARGET−SOURCE_PAIR 星地正确接收净+10、未知拒对净−104，是增加已知接受与牺牲拒识的权衡；虽AK/H提高，RU没有保持。相对OLD_PCD，星地正确接收净−1、未知拒对净+14，H仅增0.030349 pp且AK更低，不满足预登记严格AK收益。不能用SOURCE_PAIR这个较弱控制的提升替代旧PCD门槛。

| SOURCE_PAIR−SOURCE_P1（predicted输入效应） | ΔAK（pp） | ΔRU（pp） | ΔH（pp） |
|---|---:|---:|---:|
| 源DURA | -0.014815 | +0.190476 | +0.128836 |
| clean＋DURA | -0.179211 | -0.000259 | -0.164966 |
| 星地＋DURA | -0.011226 | +0.104774 | +0.013857 |

这项同结构源端比较的星地差异很小，clean的AK/H还下降；不支持本次固定配对设计有稳定联合收益。SOURCE_PAIR−SOURCE_P1反映源学习输入差异，TARGET−SOURCE_PAIR反映加入原PCD无标签目标后的差异；与旧PCD相比还同时改变参数量/融合路径，不能归为单一损失效应。本结果不证明P0没有身份信息，也不证明DURA必然删除身份。

## 场景、TX、RX/day与覆盖

| 主方法星地场景 | AK | CK | RU | H | 原F0闭集−AK（pp） |
|---|---:|---:|---:|---:|---:|
| high | 28.720515 | 43.158130 | 57.887577 | 38.392741 | 13.138692 |
| mid | 26.258401 | 40.408688 | 56.272756 | 35.807873 | 22.718010 |
| low_suburban | 31.216632 | 47.492998 | 51.509423 | 38.874105 | 15.769672 |
| high_urban | 37.468029 | 53.651000 | 54.578756 | 44.433022 | 15.785641 |
| mid_urban | 32.929816 | 47.393603 | 49.649436 | 39.597035 | 16.869242 |
| low_urban | 22.764154 | 50.696916 | 44.806738 | 30.190145 | 10.168036 |

未来整体与六主景所需名单覆盖完整，但所有主景的CK/RU/2pp保护均未通过。03-23缺旧20-19/8-20，完整宏AK/CK/H为N/A，不删类补认证；该日clean/星地RU可报62.315032/49.483986%。03-15主AK为clean31.814827%、星地27.919237%；目标只有RX34，不能把跨run不同未知集合/日期/容量的差额单因果归于RX。

星地六旧TX的AK/CK依次为：14-10 45.070423/64.118042%，14-7 11.759729/67.512690%，20-15 2.666667/28.000000%，20-19 1.970443/8.620690%，6-15 58.684654/62.731872%，8-20 47.311828/60.215054%。例如20-15原闭集正确62条仅正确接收4条；20-19正确21条仅接收正确8条，拒识损伤与原身份困难并存。

未知TX RU（clean/星地）为：5-5 35.213205/32.874828%，10-7 76.233495/60.180681%，2-19 70.637701/48.843728%，7-11 66.453901/56.312057%。源NV三个不同TX的RU为19-15 64.000000%、19-16 60.666667%、7-1 70.000000%，不与未来四类混同。

源known12（整体＋五RX＋六景）通过，known42不全过；五交叉损伤超2pp：RX3/high3.143369、RX3/mid_urban3.624502、RX3/low_urban2.063522、RX4/high_urban2.512911、RX8/low_urban2.719119 pp。RX3整体恰为2pp；已存next-q由RX3约束拒绝，本报告不再校准。源RU64.888889%仅开发诊断，不能代替跨RX目标。

源五RX主AK为83.148148%至92.481481%，RU为52.222222%至76.507937%；源三日AK为89.811111/89.633333/89.688889%。完整TX/RX/day/scene、交叉组、逐类Wilson区间及全部配对见[独立评分](../local_artifacts/pamf_phase3_paired_signal_20261031_r01/score/results.json)；单RX/seed区间不代表总体跨接收机置信保证。

## 真实资源、成本与证据边界

每学生386081参数（共享base334721＋adapter51360），训练386033，TARGET另有训练专用16×160原型2560。单主原F0232993＋学生386081＋G266951=886025参数，较旧834665增加51360；两次共享base前向不把参数也算两份。原型不部署，source/U行bank不部署；原F0/default head的闭集路径没有增加P0。

训练真实计数：SOURCE_P1与SOURCE_PAIR各113664复合调用/227328 base行；TARGET426240复合调用/852480 base行。合计653568复合调用、1307136 base行，另原r03源P1锚点缓存12434行。源P0缓存消费12434、目标P0消费576；U P1缓存复用，新G恢复0、新物理观测0。真实训练F0/G前向0，新增学生优化1332，其中目标UDA444。

评估三学生复合调用189180（源127752＋Q61428），共享base实际378360行；F0/G各20476，三学生共用每个Q的一次恢复。训练＋评估base1685496行，另原r0312434行，不将复合调用数与其两倍base前向相加误算。缓存特征/多路处理不增加独立数据量。

合成首步actual checkpoint smoke PASS，梯度probe无优化且冻结状态保持。训练合成F0/G各2、父signal2、学生复合48/base96；评估F0/G各2、学生复合6/base12。3套新密度/normalizer与6组新τ是实际源统计学习，不称零fit；旧24τ精确复用，新36τ/288B，共60τ/480B，部署单主6τ/48B。

SOURCE_P1与SOURCE_PAIR各纯源U0，TARGET含合法无标签U祖先；SOURCE_PAIR与TARGET名义学生结构/步数匹配，但TARGET有额外全U弱前向及两强视图成本，不能称全部算力预算相同。旧SOURCE444、OLD_PCD444/U576、F0E50/11100、r032220、G4940与历史缓存成本保持继承，不重复训练旧控制。

实际train230.653230 s、predict98.830313 s、CPUscore28.027663 s，pipeline358.098473 s。阶段耗时不是B1延迟或受控加速实验；同一学生的双输入有两次base前向，不能仅凭共享参数宣称免费融合。三阶段全部return0，科学低分未取消任何预测条件。

四份stdout共27行（train23/predict2/score1/driver1），全文warning/traceback/error均0。10次观测捕获driver1860550、train1860575、predict1862535的CWD/fullargv与CUDA5；GPU UUID ef5f59f2，训练406/1006/1290 MiB、预测450/582 MiB均离散样本、非峰值。CPUscore1863713只有pipeline PID/argv/gpuNone、CUDA空实现及终态产物，未独立live捕获，不能说全部阶段live已核。

训练日志从启动到1745B、预测531B、score43B、driver21B；终态owned live/GPU为空，foreign operations=0。独立真值连接依据全预测固定与CPU评分产物，不以日志增长或进程分配单独证明评分内容。

本地现有证据为11 unique tests/10 subtests通过，8AST/4CLI/actual runtime PASS，一次P0/P1审查PASS。训练首次完整5项中3过2失败，原因是旧合成fixture缺received P0；只补同ID合成P0后受影响2项12.38 s通过，其余3项未重复。eval/launch共6项10sub，受影响2项12.82 s通过；不把定向重复计为新unique或声称初轮全绿。

单包823文件/data0，SHA256 946d3be85fd4ce41ff9e5d0cc93ad514cba741c568875473d5e7f23cfdd6179d；archive比较/remote compile/runtime均VERIFIED。执行成功不改变科学判定；没有真实运行技术失败、选择性补测或临时更换模型。

固定成功条件仍是sourceknown12与cleanRU≥95%、星地RU≥90%、CK≥90%、同P1原F0闭集−AK≤2pp及主景覆盖，并要求主星地AK高于SOURCE_PAIR/OLD_PCD、RU不低两者。本轮source保护通过，但相对与绝对目标均失败，adaptation_success=False、NO_PROMOTION。AN/HOW=N/A，无发现/匿名注册，完整目标ACTIVE。

本轮只否定当前具体配对模型、训练和读出组合，不能推广为所有P0信息或双输入方法无效。源V非独立测试，NV/Q已曝光、单RX/seed、源/目标日期重合、global physical history=UNKNOWN及正式RX24-18未读均保持；早日U可含未注册设备，不能称所有未来unknown严格从未接触。不依据当前Q加大残差、调温度/阈值或换主。

证据：[完整分析](../local_artifacts/pamf_phase3_paired_signal_20261031_r01/analysis.json)、[训练终态](../local_artifacts/pamf_phase3_paired_signal_20261031_r01/train/complete.json)、[预测终态](../local_artifacts/pamf_phase3_paired_signal_20261031_r01/predict/complete.json)、[独立评分](../local_artifacts/pamf_phase3_paired_signal_20261031_r01/score/results.json)、[pipeline](../local_artifacts/pamf_phase3_paired_signal_20261031_r01/pipeline_state.json)、[release验证](../local_artifacts/pamf_phase3_paired_signal_20261031_r01/release_verification.json)。
