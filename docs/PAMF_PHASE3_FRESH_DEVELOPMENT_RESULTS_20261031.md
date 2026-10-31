# Phase3 新开发接收机 RX3-2 固定PCD验证结果（2026-10-31）

**主方法 `PCD_predicted_ratio` 不晋级（NO_PROMOTION）。** 新RX上的clean/star宏A_K为31.661804%/27.951722%，宏R_U为61.979647%/49.307698%，联合门槛未过。星地A_K相对两个控制略增，却伴随R_U下降，不能凭H微增确认普遍收益。PCD仍是研究起点，原4af参考及旧RX24-1困难结果保持，完整目标ACTIVE。

Run：`20261031-phase3-fresh-development-rx3-2-s2026092701-r01`；实际代码 `f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7`。固定方法、角色与信道见[本轮计划](PAMF_PHASE3_FRESH_DEVELOPMENT_PLAN_20261031.md)。

## 新池的实际角色与证据边界

RX3-2/index34按既有VALIDATED_ONCE的Full eq1元数据，只解码三个对应PKL成员；builder实际完成U576及future-Q10238，U/Q canonical互斥、clean/star未来ID和顺序一致，独立canonical共10814。角色排除覆盖37个既有RX；但全项目历史packet未用状态仍UNKNOWN，不称正式盲测或全历史未见。原source5RX、旧开发24-1、正式24-18等角色均保持，不扩大当前证据。

U为03-01：六旧类和三可发现设备5-5/10-7/2-19各64，总576；7-11虽早日有记录，全部从U扣除，只作为future-only未知。future取03-15/23十TX全部可用记录5525/4713，共10238：旧六4508、三可发现4320、late7-11共1410。真实名单/K/配额仅builder/scorer使用，不输入学生或原型分配；本轮不做注册，四非旧类均是拒识对象。

03-23实际缺旧20-19、8-20，完整六旧宏A_K/C_K/H和F0闭集损失必须N/A；四未知齐全时R_U仍可定义。合并两日与六场景覆盖另报，不缩真实roster、不按预测挑记录或重抽场景。新未知四类与原source NV的19-15/19-16/7-1三类不同，不能混作同一组泛化成绩。

每record单份G10 MIX01 received，新episode scene2026103133/channel2026103134，selection2026103131/shuffle2026103135；固定600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect。channel输入一次unit RMS，输出不再RMS/EQ；future clean为同record一次unit RMS控制，无clean U。builder实际10814次channel/10238份同物理clean，不能加作21052个独立样本。

## 完整444步训练与SOURCE匹配

新PCD从原source-only r03独立初始化，复用旧body a336/r01的SOURCE444控制及4273/r02源读出，不继承旧TARGET。冻结Sinc48/buffers、原F0/defaulthead/G/r03，更新body231985+projection102688；source=crossset+OE_KL+10×父z锚点，target再加.1(global+local)，无CCD/PL/EMA。16×160辅助原型仅训练容量，不是真实设备数或部署身份头。

全444步连续有限，源L/OE逐步索引与旧SOURCE完全一致；batch192L/64OE/64U，unique L6300/OE5913/U576。源曝光85248/28416，弱全U255744、两strong56832，SOURCE无U。总loss/grad从step1、源分量从step2分歧；两个目标不同，不能以总loss绝对值判胜。每step方程、完整标量曲线和极值步均存analysis。

下表为完整连续三段及全程均值；anchor未乘10，G/L为目标global/local。

| 模型 | 步段 | total | source | crossset | OE_KL | anchor | G / L |
|---|---|---:|---:|---:|---:|---:|---:|
| SOURCE | 1–148 | 0.368763 | 0.368763 | 0.287152 | 0.063361 | 0.001825 | 0.000000 / 0.000000 |
| SOURCE | 149–296 | 0.364528 | 0.364528 | 0.285097 | 0.061748 | 0.001768 | 0.000000 / 0.000000 |
| SOURCE | 297–444 | 0.353902 | 0.353902 | 0.273840 | 0.059812 | 0.002025 | 0.000000 / 0.000000 |
| SOURCE | 1–444 | 0.362398 | 0.362398 | 0.282029 | 0.061640 | 0.001873 | 0.000000 / 0.000000 |
| FRESH_PCD | 1–148 | 0.928747 | 0.385545 | 0.288989 | 0.066710 | 0.002985 | 2.703647 / 2.728372 |
| FRESH_PCD | 149–296 | 0.837144 | 0.383604 | 0.288296 | 0.064668 | 0.003064 | 2.264803 / 2.270592 |
| FRESH_PCD | 297–444 | 0.796956 | 0.373687 | 0.277963 | 0.062477 | 0.003325 | 2.118620 / 2.114073 |
| FRESH_PCD | 1–444 | 0.854282 | 0.380945 | 0.285083 | 0.064618 | 0.003124 | 2.362357 / 2.371012 |

新PCD总loss三段下降.928747→.837144→.796956，global/local末段2.118620/2.114073；末段source.373687、anchor.003325仍高于SOURCE.353902/.002025。普通AdamW lr1e−4/wd1e−4/eps1e−8、norm clip5调用444次，197步裁剪前>5、最大12.657046；Sinc优化/physicalclip0。首2步浅中层/四分支/body/projection/prototype梯度非零、Sinc0；全程模块梯度未记录，不补推逐步证据。

完成标记确认body/projection变化、Sinc/buffers和原模型保持；复制feature累计13764、投影2664，原F0仍11100。原型初始化25轮分配13至73，容量16非真实K；本轮steps未记录每步原型质量，不能据初始化推纯度或发现成功。

## 18个固定运行点

数值为百分比；损失为同DURA原F0闭集−A_K（百分点）。所有宏平均对必需真实TX等权，非按样本数；H为宏A_K/R_U调和平均，C_K为宏已知接收率。

| 条件 | 方法 | A_K | C_K | R_U | H | F0损失 | AUROC / OSCR |
|---|---|---:|---:|---:|---:|---:|---:|
| source_DURA | E0_pooled_ratio | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 / 87.944738 |
| source_DURA | E0_predicted_ratio | 89.744444 | 96.759259 | 61.936508 | 73.291437 | 1.237037 | 94.734250 / 87.937558 |
| source_DURA | SOURCE_pooled_ratio | 89.777778 | 97.088889 | 60.063492 | 71.974388 | 1.203704 | 94.963471 / 87.965076 |
| source_DURA | SOURCE_predicted_ratio | 89.740741 | 96.744444 | 61.174603 | 72.754089 | 1.240741 | 94.685262 / 87.940159 |
| source_DURA | PCD_pooled_ratio | 89.770370 | 97.081481 | 63.619048 | 74.465443 | 1.211111 | 94.961329 / 87.937604 |
| source_DURA | PCD_predicted_ratio | 89.733333 | 96.748148 | 64.285714 | 74.907247 | 1.248148 | 94.730673 / 87.940433 |
| clean_DURA | E0_pooled_ratio | 31.187288 | 43.537692 | 62.980616 | 41.716859 | 17.379728 | 53.846068 / 32.740387 |
| clean_DURA | E0_predicted_ratio | 30.999690 | 42.654045 | 64.187968 | 41.808091 | 17.567325 | 53.862200 / 33.089825 |
| clean_DURA | SOURCE_pooled_ratio | 31.475324 | 44.146095 | 62.409528 | 41.846157 | 17.091691 | 54.399171 / 33.040709 |
| clean_DURA | SOURCE_predicted_ratio | 31.078517 | 43.181373 | 63.911653 | 41.820736 | 17.488498 | 54.219966 / 33.219089 |
| clean_DURA | PCD_pooled_ratio | 32.123647 | 44.791196 | 60.790839 | 42.034854 | 16.443369 | 54.164696 / 33.247293 |
| clean_DURA | PCD_predicted_ratio | 31.661804 | 43.761391 | 61.979647 | 41.912794 | 16.905211 | 54.364758 / 33.693180 |
| star_DURA | E0_pooled_ratio | 26.867357 | 47.141720 | 51.837083 | 35.391279 | 14.853702 | 51.396572 / 26.658213 |
| star_DURA | E0_predicted_ratio | 26.440597 | 46.197372 | 52.658525 | 35.204508 | 15.280462 | 51.242710 / 26.543484 |
| star_DURA | SOURCE_pooled_ratio | 27.422108 | 48.651740 | 51.188933 | 35.712755 | 14.298951 | 51.487998 / 26.781123 |
| star_DURA | SOURCE_predicted_ratio | 26.751337 | 47.266339 | 52.269979 | 35.390244 | 14.969721 | 51.144838 / 26.486365 |
| star_DURA | PCD_pooled_ratio | 29.175335 | 50.230381 | 48.454372 | 36.420916 | 12.545723 | 51.015031 / 26.992789 |
| star_DURA | PCD_predicted_ratio | 27.951722 | 48.582412 | 49.307698 | 35.678111 | 13.769336 | 50.997190 / 27.001261 |

同DURA原F0闭集source/clean/star为90.981481%/48.567015%/41.721059%，默认头不变。新RX闭集本身弱，开放拒识还使正确样本损失16.905211/13.769336pp；这是两个可量化限制，不能把退化唯一归因于原F0，未知拒识本身也未达标。主clean接收2470/4508，正确1647/错误823；star接收2633，正确1564/错误1069，接受不等于正确身份。

主对E0 predicted：source正确35增/38失（−3）、未知155/81（+74）；clean正确13/22（−9）、未知50/175（−125）；star正确27/21（+6）、接收+21、未知47/237（−190）。星地宏ΔA_K+1.511126/ΔR_U−3.350826/ΔH+.473603pp，RU保持条件失败。

主对匹配SOURCE predicted：source正确19/21（−2）、未知166/68（+98）；clean正确14/23（−9）、未知67/176（−109）；star正确24/16（+8）、接收+12、未知37/205（−168）。星地宏ΔA_K+1.200385/ΔR_U−2.962281/ΔH+.287867pp。各类容量不同，净条数不与宏指标同比：clean正确净少9但宏A_K略增，不能据总数反算宏准确率。

主source known12通过、known42未全过（RX3 high/mid urban/low urban、RX4 high urban、RX8 low urban五交叉组超2pp）。原source V27000是额外known校准监督，源NV3150只是原三未知历史诊断；本轮用同V拟合两新六τ，不是目标label或独立known盲测。

## 日期、TX与场景覆盖

| 条件 | 日期 | known / unknown | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|---:|
| clean_DURA | 2021_03_15 | 2679 / 2846 | 31.814827 | 42.134711 | 61.753558 | 41.994500 |
| clean_DURA | 2021_03_23 | 1829 / 2884 | N/A | N/A | 62.211551 | N/A |
| star_DURA | 2021_03_15 | 2679 / 2846 | 27.957368 | 47.745760 | 49.505315 | 35.734324 |
| star_DURA | 2021_03_23 | 1829 / 2884 | N/A | N/A | 49.136802 | N/A |

目标未知5-5/10-7/2-19/7-11各1454/1439/1427/1410条；主clean拒对510/1087/1008/940，star479/857/695/790。旧20-15和20-19主clean正确仅3/150与2/406，star4/150与9/406；完整十TX、RX34、两日、源五RX三日及CI保留于结果文件。future-only7-11是设计性从U扣除，不是自然早期无记录；当前未注册状态不等于全祖先严格未见。

合并两日的star六景类别覆盖完整；主A_K/R_U依次high28.720515/57.887577、mid26.258401/55.771659、low suburban31.321362/51.231701、high urban37.468029/54.578756、mid urban32.929816/49.168666、low urban22.764154/44.635122（%）。03-23缺类只保留N/A，不补零/删类；它与旧RX24-1 high缺19-15的历史缺项不是同一数据问题。

合并主clean CK43.761391/RU61.979647/损失16.905211pp，star CK48.582412/RU49.307698/损失13.769336pp；远未达cleanRU95/starRU90/CK90/原F0损失2pp。即使star宏A_K高于两个control，也未保持R_U，故adaptation_success=False、NO_PROMOTION，不以pooled的局部AK替换固定主。

## 冻结顺序、成本与执行

训练神经先冻，再以原L6300/OE6134新拟合一份known90/OE32/normalizer，V27000只校准PCD两组六τ12值；E0/SOURCE源四列聚合与24旧τ精确复用。全36τ288B冻后NV3150和两个future10238完整预测，独立CPU最后连接三truth各一次，Utruth0。SOURCE权重无U，新PCD及其源GMM有U576祖先；目标真实四类/K/比例未用于拟合。

新训练父signal13010+学生426240/G576/F00，新body/projection各444，旧SOURCE444仅继承；新评估signal104012（E0/SOURCE各20476、PCD63060）、F0/G各20476。新增链signal543262/G21052/F020476；builder观测生成10814与同物理clean10238另列，不称更多独立样本。训练parent/F0/G各2、student probe48，评估三signal共6/F0G各2合成行独列。

部署主NN834665、GMM364340B、单主六τ48B不变，不部署16辅助原型或source/U行bank；本轮1源GMM/normalizer及12τ是真实新统计拟合。祖先F011100/r03原2220/G4940、SOURCE训练及缓存成本继承，不将新RX改变冒充同成本同未知集合的消融。

三阶段COMPLETE/return0，train83.716393s、predict35.859623s、CPU score18.341892s、流程138.107009s，非B1。6次观测核driver1601671、train1601716和predict1604856 CWD/fullargv/CUDA3；GPU UUID7b4bacb3-4901-6845-a414-0dfdfeeade97训练离散10/406/1382/1382MiB、预测580MiB，非峰值。CPU1605578仅pipeline gpuNone/代码CUDAempty/产物，未独立live；终态owned空、foreign0。

builder/train4项+7子项（10.41s）、producer/score6+12（14.60s）、launcher13项（3.11s）PASS；同case定点handoff1+3/7.81s和1+5/7.70s不额外增加unique项。最终整套测试旧np.trapz警告1124、定点1056另记，RED历史保留；实际全stdout无warning/error/nonfinite，一次P0/P1及真实CP smoke PASS。发布包740文件/11个本轮数据文件，SHA `2ebd6d3be0b6cd6de2050a7f01900ceb5d662de3c50c736a3052275824ab1032`比较/compile/runtime VERIFIED。

本地start返回曾因monitor缺psutil出错，builder PID26824健康继续，改用Win32_Process/进程头CWD核实后完成，未重启。初私有runtime缺view_family在验证层被拒，模型启动前补生成字段；tracked方法/预算未变，非真实三阶段技术失败。

新RX与旧RX24-1同时改变未知TX集合、日期、每类容量及episode，跨run差额不能解释为纯RX或纯方法因果。global history UNKNOWN、单RX/seed、source日期重合、旧NV/Q历史曝光仍保留，正式RX24-18未访问。本轮未确认普遍收益，不替代旧困难结果；无匿名REG，A_N/H_OW=N/A，完整目标ACTIVE。

后续研究需同时检验合法身份适应与拒识表征：当前open-only适配保持原F0身份输出，只改变接收/拒绝，无法把错误TX预测纠正。该判断不是从本次Q选择参数、阈值或新方案有效性，也不证明纯RX因果；任何新增试验另行固定设计。

复核产物：`local_artifacts/pamf_phase3_fresh_development_20261031_r01/pool/{builder_report,complete}.json`（实际池/角色元数据，未重读truth）、`train/steps.jsonl`（完整444步）、`score/results.json`（18点/配对/全部TX-RX-day-scene与CI）、`predict/models/complete.json`/`calibration/complete.json`、`analysis.json`（全曲线/匹配/预算/日志）、`observations.jsonl`/`pipeline_state.json`。
