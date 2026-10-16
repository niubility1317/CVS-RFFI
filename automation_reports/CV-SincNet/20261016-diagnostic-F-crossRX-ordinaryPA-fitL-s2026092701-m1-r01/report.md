# F＋跨 RX 同 TX：普通 PA 增强的同抽样身份反事实

状态：ANALYZED_SOURCE_ONLY／VERIFIED。全部固定视图已完成并独立复算。此项审计现有普通增强，不引入新的 PA 损伤，不是新 A_g 效果实验。固定谱滤波候选已因有限 IQ 的 PSD 保真失败关闭，不调参救跑。

唯一假设：已启用的 ordinary PA-normal 在真实触发包上造成可避免的身份错误。普通管线实际为 DC/taper/循环移位/gain/phase/CFO/Wiener/multipath/AWGN/DAC/PA；PA 位于末尾、DAC p=0、class_signature=false。概率按实际日程为0.014至0.049；Rapp的sat0.6至1.1及p2至5未随强度缩减。该继承配置没有获得身份保持证明。

## 固定最小源矩阵

只用自有冻结 F＋CROSS_RX B00/seed2026092701/fixedfinal20，既有三路SIG_TIME、统一Sinc、joint160、NoExplicitCFO及必要补偿、原默认cosine头保持。checkpoint来自20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01的B00，scratchE20/4440更新，无祖先、目标接触或目标选模；实际L5040/U45360/V21600契约与当前源折一致，不加载E50/E200或其他来源。它仅作冻结诊断，不初始化新训练。原父成本E20/4440/4686678NN另列。

复用既有source_contract与SOURCE_GEOMETRY_AUDIT_COMPLETE缓存、EXACT_MATCH角色结论。在任何IQ统计、变换或推理前排除heldRX1，fit ID严格等于冻结模型L5040，RX3/4/6/8×day1/2/3×全部6TX×70。未用U、V、heldRX1或target；缓存可反序列化其他源角色和旧特征但不计算使用。该fit分析不能代替未见RX验证，不能冒充完整L6300或L5580/Ldev720。

raw只推理一次。对全部20轮原生增强日程，使用实际checkpoint args调用native build_aug_base_cfg/make_augmentor/configure_augmentor_for_epoch。每轮Python/Torch seed=20261016241+epoch，固定fit缓存顺序及128/末48批次。这是新诊断抽样，不伪称重放旧训练随机轨迹。每批只调用一次原生augmentor，从post-general-channel输出捕获完整prefix，再取得PA后的full；prefix即只旁路PA的反事实。未重抽通用噪声/多径、未强制PA触发、未更改操作范围或追加RMS归一化。记录真实strength、selected输入、subchain调用、变动包、全包及80:160参考窗MSE、功率比。未触发包full/prefix须逐位相同，NN eval不消费后续增强RNG。

raw5040＋20×5040×2=206640源默认头预测，加2合成CP smoke=206642实际NN包前向；100800普通增强包，实际PA触发与变动数读回。0训练/头拟合/teacher/As/额外分类器。局部验证1344合成增强包，0真实IQ/CP/NN，11负例PASS；全部20日程canonical native输出逐位一致、Python/Torch RNG终态相同、同起始随机状态noPA输入等于捕获prefix。

## 预登记判断

在所有20轮汇总的真实PA变动包上，至少1000个观测，旁路PA相对full的准确率恢复≥2个百分点；至少3个RX恢复为正，任一RX不低于−1个百分点，每个day不低于0。记录repair与new break、各TX负结果、原始及full/bypass的accuracy/MacroF1/CM/CE；全部20轮均报告，不挑强度、seed、RX或子链。每物理包重复20次抽样相关，不用朴素包级p值宣称独立证据。结果不满足即关闭这项PA伤害训练依据，不改阈值或救跑；失败也不能证明PA身份保持。

通过仅支持另行登记同预算scratchE20普通配方消融，仍须留出RX的真实clean正收益和practical保护，不能把源拟合恢复直接称A_g泛化有效。当前不自动训练或启动target。主联合目标lowurban>60%及suburban接近clean仍未达成；原始B00/B10/B01/B11要求保留。

## 发布与产物

唯一owner codex/root/F-CROSS_RX-ordinaryPA-20261016；普通N607账号，CPU2线程、CUDA_VISIBLE_DEVICES空，不占GPU槽或干预健康任务。CWD为PROJECT/releases/cvs_F_crossRX_ordinaryPA_fitL_20261016_r01，命令CVS-RFFI/bin/python -u experiments/cvs_pa_counterfactual16/runner.py。精确路径、数据与六类seed及逐行矩阵见experiment.json。代码身份以实际Git commit/release_commit/resolved为准。一次N607 preflight、一个归档单次SHA对比、一次远端compile；launcher第一步真实CP合成无query smoke，PASS立即继续；独立PID/CWD/argv/CPU环境/log增长核验。

角色/CP/非有限输入、实际抽样或成本/状态不匹配、输出冲突为技术失败，保留产物无自动重试；科学负结果不停止或热改，全部20日程完成。保存resolved/smoke/fitID/raw与20轮全部logits预测及真实触发/变化记录、epoch充分统计、result/completion或failure、完整stdout。预测完成后独立重算全部身份指标、触发计数、RX/TX/day配对和固定判断。IQ/CP/逐包预测及日志不入Git，仅汇总报告与代码/配置/登记入Git。


实际执行commit `cf1586790344aff65d5d7034e8d92e72f91b27d5`，archive/compile/realCP smoke及PID-CWD-argv-CPU/log增长均VERIFIED，无目标访问；全部固定源视图继续，未重启或热改。


## 实际完整结果

预登记判定 **PASS**，条件：{"sufficient_changed": true, "changed_identity_recovery": true, "RX_coverage": true, "RX_positive": true, "RX_protection": true, "day_protection": true}。全部206640默认头预测及logits/CE/CM/F1、20轮真实触发/变动、1440源单元和RX/TX/day配对已独立复算。

原raw准确率99.841270%；实际PA触发3060、IQ变动3054观测，涉及2343个唯一物理包。变动包修复640、新增错27，净恢复+20.072037pp；全100800增强观测净恢复+0.608135pp。只旁路PA，其余同抽样；不把重复物理观测当独立样本显著性。

| 轮次 | PA概率 | 触发/变动 | full准确率 | bypass准确率 | 变动包恢复 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.014000 | 79/79 | 99.603175% | 99.841270% | +15.189873pp |
| 2 | 0.014000 | 72/72 | 97.896825% | 98.134921% | +16.666667pp |
| 3 | 0.014000 | 57/57 | 97.341270% | 97.440476% | +8.771930pp |
| 4 | 0.015186 | 78/78 | 98.730159% | 98.928571% | +12.820513pp |
| 5 | 0.016820 | 97/97 | 97.678571% | 97.916667% | +12.371134pp |
| 6 | 0.018681 | 87/87 | 95.972222% | 96.170635% | +11.494253pp |
| 7 | 0.020707 | 100/100 | 98.511905% | 98.769841% | +13.000000pp |
| 8 | 0.022865 | 115/115 | 99.325397% | 99.781746% | +20.000000pp |
| 9 | 0.025134 | 129/123 | 94.265873% | 94.742063% | +19.512195pp |
| 10 | 0.027500 | 165/165 | 94.603175% | 95.119048% | +15.757576pp |
| 11 | 0.029952 | 161/161 | 97.916667% | 98.472222% | +17.391304pp |
| 12 | 0.032482 | 158/158 | 91.706349% | 92.559524% | +27.215190pp |
| 13 | 0.035084 | 183/183 | 92.142857% | 92.718254% | +15.846995pp |
| 14 | 0.037752 | 180/180 | 92.976190% | 93.730159% | +21.111111pp |
| 15 | 0.040481 | 192/192 | 88.095238% | 89.265873% | +30.729167pp |
| 16 | 0.043267 | 205/205 | 87.896825% | 88.571429% | +16.585366pp |
| 17 | 0.046108 | 242/242 | 87.678571% | 88.849206% | +24.380165pp |
| 18 | 0.049000 | 272/272 | 90.218254% | 91.646825% | +26.470588pp |
| 19 | 0.049000 | 253/253 | 90.357143% | 91.388889% | +20.553360pp |
| 20 | 0.049000 | 235/235 | 91.170635% | 92.202381% | +22.127660pp |

| 分层 | 变动包数 | 修复 | 新错 | 恢复 |
|---|---:|---:|---:|---:|
| by_RX=3 | 766 | 200 | 8 | +25.065274pp |
| by_RX=4 | 784 | 138 | 5 | +16.964286pp |
| by_RX=6 | 737 | 116 | 9 | +14.518318pp |
| by_RX=8 | 767 | 186 | 5 | +23.598435pp |
| by_day=1 | 1027 | 218 | 10 | +20.253165pp |
| by_day=2 | 970 | 188 | 6 | +18.762887pp |
| by_day=3 | 1057 | 234 | 11 | +21.097446pp |
| by_TX=0 | 523 | 96 | 5 | +17.399618pp |
| by_TX=1 | 512 | 162 | 4 | +30.859375pp |
| by_TX=2 | 525 | 109 | 2 | +20.380952pp |
| by_TX=3 | 533 | 127 | 11 | +21.763602pp |
| by_TX=4 | 470 | 57 | 3 | +11.489362pp |
| by_TX=5 | 491 | 89 | 2 | +17.718941pp |

PA子链实际调用与涉及包数：{"_pa_memory_polynomial": {"calls": 684, "selected_packets": 2747}, "_pa_apply_fir_memory": {"calls": 318, "selected_packets": 1298}, "_pa_rapp": {"calls": 667, "selected_packets": 2734}, "_pa_am_pm": {"calls": 449, "selected_packets": 1865}, "_iq_imbalance": {"calls": 37, "selected_packets": 174}}。触发不必改变IQ，二者独立保留。

实际执行commit `cf1586790344aff65d5d7034e8d92e72f91b27d5`；单归档传输/compile/CP smoke/PID-CWD-argv-CPU/log增长VERIFIED，进程自然退出且无failure；无重启或热改。新增206642NN包前向、100800普通增强观测、0训练/头拟合/teacher/As，参数与buffer不变，墙钟121.896775秒，完整stdout23行。父模型与本地1344合成增强包成本另列；未用heldRX1/U/V/target。

已证实：当前冻结默认头和实际日程下，旧PA分量带来可避免的fit源身份错误，且原定RX/day条件通过。未证实：移除PA能改善训练后的未见RX clean或星地适应。下一唯一训练假设是仅旁路ordinary PA、保留相同原生抽样及原L/U/F+CROSS_RX，可改善未见RX clean；最小为固定RX1源折同seed各从零E20的匹配B00/PA-bypass两组，保持六L前向/原损失尺度、同物理序列/默认头/U，使用同次完整原生call捕获prefix以保留native随机流。必须先单独登记数值门槛、成本与自动源七视图；若clean不增或practical保护失败，不改seed/RX/范围救跑。此处仅提出，未实施/登记/启动训练。
