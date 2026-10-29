# Phase3 源端星地编码器联合增量结果

**主候选 practical_increment::mlp16_oe 为 NO_PROMOTION。** P1 未知拒识由父 E51 的38.8889%升至50.0000%，但开放识别由68.5764%降至62.8472%，总损伤9.8958 pp；C0 未知98.6111%单项通过，闭集却降至67.5347%、开放57.8125%，总损伤21.8750 pp。两arm／四条开发行的完整目标均失败，不以未知拒识单项提升换主arm或解释为成功。

run：20261029-phase3-source-practical-encoder-s2026092701-r01；执行代码b578c074ba68b8ff9aa539209b3410f1f103ba3d，预登记提交a5ce1ee04b68288e382184b33530f2570233c71e。15jobs均COMPLETE，实际终态VERIFIED／ANALYZED_DEVELOPMENT_ONLY；本报告只分析取回元数据／日志，没有读真实IQ、weights、truth或重跑，总Phase3尚未完成。下表C／P分别指clean_increment控制／practical_increment主候选，百分数与百分点严格区分。

## 数据、匹配预算与权限

原F E50的source L6300／U56700对应10%源训练标签比例，U的TX真值隐藏；V27000另用于验证／校准，不能算入训练标签分母。source五RX为1-19、18-2、19-2、2-19、3-19（index1/3/4/6/8），三日为2021-03-08、03-15、03-23。旧六类顺序14-10、14-7、20-15、20-19、6-15、8-20。两arm独立从uniform-OE E51出发：11100＋222＋本次各222＝各11544步，不串接两个arm。

新增编码器训练只用L6300与source OE21／6134，OE为成员监督、真实TX隐藏；原U、V、novel、target不进优化。共享缓存12434物理／49736计算view，clean＋low suburban／mid urban／low urban residual后接冻结DURA R69 E20/4940（FP32、TF32off、reuse/graph true、native false），不二次RMS/EQ；channel seed2026102931、receiver2026102912。缓存与两arm复用不增加独立物理样本。

| 每arm实际训练曝光 | 原记录抽取 | packet views |
| --- | --- | --- |
| L | 28416 | native170496＋额外28416＝198912 |
| OE21 | 14208 | clean＋第二view＝28416 |

两组都保留native L128六路、ordinary epoch13、As、cross-RX0.1、LS0.01；As独立KL权重实际为0，As输入／CE机制仍在。每步同一L128附加0.68 CE，C额外clean，P逐样本iid三practical/G；OE64的clean＋对应第二view合成128，uniform6 KL均值权重0.1。control不能称整体无As／无增强。两组fresh AdamW CENTER_UNITS、lr1e-4常数、实际wd0.0001、physical clip5，一次反传／更新，Sinc及原默认头均实际联合更新，不替换头。

L/OE/scene独立固定种子2026092701／2026102907／2026102932。两组native As计数均64→14208，每步768 native packets、128额外packets；PL seen/accepted与旧hook的T1_sat_seen均为0。后者不统计本候选额外practical，不能用它否认P的额外星地曝光。两个实际known/OE gradient smoke均恢复state／RNG、优化0，后续fresh优化器／Sinc steps／clip及native计数完整222的运行断言通过。

| P实际额外scene histogram | low suburban | mid urban | low urban | 合计 |
| --- | --- | --- | --- | --- |
| L | 9568 | 9582 | 9266 | 28416 |
| OE | 4839 | 4663 | 4706 | 14208 |

C额外L/OE全部clean，各28416／14208；P直方图来自类独立iid流，不强制等配额。source评估每armL6300／OE4 4200／V27000／novel3 3150，共40650物理、162600计算view；OE4为11-14、14-19、17-20、18-19，novel为19-15、19-16、7-1各1050。每个新F用自身clean L拟合8分支聚合几何，L/OE4四view训练linear101／MLP1633各200CPU步；V不拟合／早停，novel在自身τ保存后才打开。

开发每arm C0/P1各Q864＝known576＋unknown288，RX24-1、2021-03-15。两arm／条件仍共享864原Q；C0独立clean无G，P1同一received经固定G、无clean参考。source与开发跨RX但日期非严格未见，novel3／开发池已暴露；仅一seed、一开发RX、三star主景。targetU0／formal RX24-18 U/Q0，无DA／发现／匿名注册；历史planned later仅诊断，本run未见过U。

## 完整训练曲线与读出拟合

| 222步总loss | 首→末 | 全均值 | 首37均值→末37均值 | 上升步/221 |
| --- | --- | --- | --- | --- |
| clean_increment | 0.604306 → 1.211241 | 0.632637 | 0.625237 → 0.672813 | 114 |
| practical_increment | 0.934773 → 1.328920 | 0.806549 | 0.803557 → 0.838584 | 110 |

| 完整trace分项：首37→末37均值 | C | P |
| --- | --- | --- |
| OE均匀KL | 0.468342 → 0.392204 | 0.722533 → 0.577146 |
| 额外CE | 0.090009 → 0.088556 | 0.308275 → 0.287075 |
| ordinary KL | 0.023656 → 0.030175 | 0.024364 → 0.029670 |
| As KL（权重0） | 0.067232 → 0.075440 | 0.067466 → 0.074870 |
| 原contrast | 1.141220 → 1.363998 | 1.142440 → 1.364736 |

444步及全部分项／梯度均有限。总loss非单调，末37步均值均高于首37；OE目标与额外CE有局部下降，native ordinary／As诊断及contrast波动并后段升高，不能称编码器收敛。实际gradient smoke的Sinc约0.04849／0.03069、head约11.35870／9.02152（known／OE）均非零；训练终标记确认Sinc/head更新。固定222步终点，无V挑步。

| 读出：各200点 | 模型 | 首→末loss | 全均值 | 首25→末25均值 | 上升步 |
| --- | --- | --- | --- | --- | --- |
| clean_increment | linear_oe | 0.693147 → 0.218277 | 0.314732 | 0.546623 → 0.222573 | 0 |
| clean_increment | mlp16_oe | 0.850289 → 0.145498 | 0.250301 | 0.558009 → 0.149094 | 0 |
| practical_increment | linear_oe | 0.693147 → 0.199795 | 0.294954 | 0.537445 → 0.203734 | 0 |
| practical_increment | mlp16_oe | 0.872119 → 0.136717 | 0.236973 | 0.551741 → 0.139570 | 0 |

800个读出loss点均有限且逐步下降，末段仍下降；这证明固定成员监督拟合完成，不证明source novel或跨RX拒识有效。读出mean/std只来自本arm L/OE4，自己的几何／读出／τ在Q期间冻结，无source逐样本bank部署。

## Source：局部星地收益没有转成完整通过

14套方法校准均no_rejection_feasible=true、constraints_passed=true，下一候选τ均失败，没有不可行全接收兜底。只用known V四view，按六类macro保护总体／5RX／4view／20交叉共30组FRR≤5%、原F0同输入closed−open≤2pp，Fraction预算及整组ties；旧encoder0.5pp仅诊断（false），未变成新门槛。

| MLP校准 | global τ | 30组最大FRR % | 30组最大总损伤 pp |
| --- | --- | --- | --- |
| clean_increment | -2.25931883 | 2.7963 | 2.0000 |
| practical_increment | -1.66401005 | 4.6667 | 2.0000 |

| source MLP：C／P | 未知拒识 % | 闭集 % | 开放 % | 原F0总损伤 pp |
| --- | --- | --- | --- | --- |
| clean | 73.3651／77.1111 | 98.5519／98.3481 | 98.3185／97.8704 | 0.1481／0.5963 |
| practical_low_suburban | 49.1746／54.9524 | 95.8556／95.8111 | 95.5519／95.1481 | 0.1852／0.5889 |
| practical_mid_urban | 39.8730／46.6349 | 91.3519／91.7037 | 90.8667／90.6185 | 0.5630／0.8111 |
| practical_low_urban | 31.5238／38.6032 | 84.9519／85.6704 | 83.9407／83.7593 | 1.1593／1.3407 |

P相对C的source未知点拒识提高3.7460／5.7778／6.7619／7.0794 pp，mid/low urban闭集分别提高0.3519／0.7185 pp；但四view开放率与AUROC均更低，不能写成source整体改善。P源clean闭集仍98.3481%（C98.5519%）、low urban85.6704%（C84.9519%），与新RX大幅下降形成明显边界；证据支持跨RX泛化损伤，不足以断言全局灾难性遗忘。

| source七方法未知 %：C／P | clean | low suburban | mid urban | low urban |
| --- | --- | --- | --- | --- |
| msp | 47.7143／40.5397 | 26.2222／23.8095 | 22.5714／20.3175 | 17.7778／16.3492 |
| maxlogit | 64.9206／55.7778 | 30.3492／28.8571 | 27.2063／25.7143 | 20.0317／21.3968 |
| margin | 28.0317／18.5714 | 15.8730／12.0317 | 14.5079／10.3175 | 12.1270／9.3333 |
| energy | 68.0000／61.6508 | 29.5873／29.1746 | 25.6825／26.5079 | 19.0476／21.1429 |
| cosine | 10.1270／7.9365 | 11.6190／6.2540 | 9.7460／4.9524 | 9.4286／5.4921 |
| linear_oe | 66.2857／66.5397 | 51.0159／50.1270 | 41.6508／42.1905 | 36.4444／35.7460 |
| mlp16_oe | 73.3651／77.1111 | 49.1746／54.9524 | 39.8730／46.6349 | 31.5238／38.6032 |

两arm主source goal均false；MLP逐RX×view未知目标通过数为C 0/20、P 0/20。source known保护只是在同批V校准上的经验结果，novel也重复开发；完整七方法／RX／TX／view及交叉见末尾索引，跨view pooled不作独立CI。

## 开发主表：父E51、增量与原F0分开

parent_E51精确复用上一run的augmented_readout state，非更早original D0；其全部七方法总体／scene／TX／planned组与原评分逐项一致。原F0基准始终C0=459/576（79.6875%）、P1=419/576（72.7431%），不是父E51自身closed。

| 条件 | state/MLP | closed正确 | 未知拒识 | known接收 | FRR % | 开放正确 | 总损伤 pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | parent_E51 | 451/576（78.2986%） | 279/288（96.8750%） | 521/576 | 9.5486 | 412/576（71.5278%） | 8.1597 |
| C0 | C | 436/576（75.6944%） | 281/288（97.5694%） | 520/576 | 9.7222 | 404/576（70.1389%） | 9.5486 |
| C0 | P主候选 | 389/576（67.5347%） | 284/288（98.6111%） | 494/576 | 14.2361 | 333/576（57.8125%） | 21.8750 |
| P1 | parent_E51 | 414/576（71.8750%） | 112/288（38.8889%） | 536/576 | 6.9444 | 395/576（68.5764%） | 4.1667 |
| P1 | C | 414/576（71.8750%） | 107/288（37.1528%） | 543/576 | 5.7292 | 398/576（69.0972%） | 3.6458 |
| P1 | P主候选 | 403/576（69.9653%） | 144/288（50.0000%） | 513/576 | 10.9375 | 362/576（62.8472%） | 9.8958 |

相对父E51，P在C0少62条closed正确、少79条开放正确，仅多拒未知5条；P1少11条closed正确、少33条开放正确，多拒未知32条。相对同预算C，P的C0 closed少47条＝−8.1597 pp；P1未知多拒37条＝+12.8472 pp，开放却少36条＝−6.2500 pp。接收不是分类正确，拒识收益不能掩盖身份损伤。

| 新encoder／原联合头判决漂移 | 条件 | 相对parent改变的label数/864 |
| --- | --- | --- |
| C | C0 | 75 |
| C | P1 | 15 |
| P | C0 | 141 |
| P | P1 | 45 |

上述label改变数包含unknown，不等于已知正确数净变化；它来自新F／原头联合更新后的判决，不能称历史数值复现误差。每arm自有新几何／读出／τ都在source阶段冻结，同输入原F0 baseline复用且未另跑。

| MLP主景 | scene | 未知 % | 接收 % | closed % | open % | 总损伤 pp | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C | clean | 97.5694 | 90.2778 | 75.6944 | 70.1389 | 9.5486 | 0.975272 |
| C | practical_low_suburban | 46.8750 | 95.8333 | 78.1250 | 75.0000 | 2.0833 | 0.918403 |
| C | practical_low_urban | 31.2500 | 94.2708 | 67.7083 | 65.1042 | 5.7292 | 0.837674 |
| C | practical_mid_urban | 33.3333 | 92.7083 | 69.7917 | 67.1875 | 3.1250 | 0.856771 |
| P | clean | 98.6111 | 85.7639 | 67.5347 | 57.8125 | 21.8750 | 0.971861 |
| P | practical_low_suburban | 60.4167 | 90.1042 | 74.4792 | 66.1458 | 10.9375 | 0.898817 |
| P | practical_low_urban | 42.7083 | 87.5000 | 67.1875 | 59.3750 | 11.4583 | 0.809950 |
| P | practical_mid_urban | 46.8750 | 89.5833 | 68.2292 | 63.0208 | 7.2917 | 0.843479 |

全部开发行／主景完整目标均失败：clean未知单项通过，known保护失败；三个star景未知均远低90%。P1每景known192／unknown96、各TX等量。各模型分数与归一化不同，不能直接比较τ大小解释跨arm作用，也没有用target扫τ或选步。

| 开发七方法：C／P | C0未知 % | P1未知 % | C0开放 % | P1开放 % |
| --- | --- | --- | --- | --- |
| msp | 83.3333／69.4444 | 22.5694／21.8750 | 72.9167／65.9722 | 69.6181／68.0556 |
| maxlogit | 95.4861／92.7083 | 25.3472／28.8194 | 70.8333／62.6736 | 69.4444／67.0139 |
| margin | 51.7361／28.4722 | 12.8472／12.1528 | 71.7014／65.9722 | 68.9236／67.7083 |
| energy | 93.7500／95.1389 | 26.0417／30.2083 | 69.9653／60.2431 | 69.6181／65.7986 |
| cosine | 10.4167／8.6806 | 7.2917／5.5556 | 74.8264／67.5347 | 70.8333／69.4444 |
| linear_oe | 94.4444／94.4444 | 41.3194／43.0556 | 71.5278／61.2847 | 68.2292／64.4097 |
| mlp16_oe | 97.5694／98.6111 | 37.1528／50.0000 | 70.1389／57.8125 | 69.0972／62.8472 |

## 逐类损伤与无需阈值搜索的边界

P的C0旧类14-7／20-19仅6／28条closed正确（父为21／67、C为18／63），两类仍分别全接收／接收94条，主要问题包含身份混淆；6-15 closed73条，却误拒71条，open仅22条。P1的6-15／8-20分别误拒40／20，open50／68条。完整六类及七方法的逐TXclosed／接收／open细节均在评分索引，不按TX制定规则。

| unknown TX（各96）：C／P | C0拒识 % | P1拒识 % |
| --- | --- | --- |
| 19-15 | 97.9167／100.0000 | 56.2500／66.6667 |
| 19-16 | 98.9583／97.9167 | 18.7500／15.6250 |
| 7-1 | 95.8333／97.9167 | 36.4583／67.7083 |

planned later 7-1诊断：C0为C92/96、P94/96；P1为C35/96、P65/96，星地仍低90%。该角色本run只诊断、不改变primary判定；U读取0，不能写成已无标签见过、适应或注册。

直接身份上界：open≤closed，而P的C0/P1 closed为67.5347%／69.9653%，已低于原F0closed−2pp＝77.6875%／70.7431%。因此即使全部接收，这个冻结新F也无法通过两条件身份保护；不是只改τ能解决。P的P1 AUROC=0.852515＜0.95×0.90＝0.855（等量池达到95%接收及90%拒识的必要排序下界），也排除了仅平移当前global分数阈值同时满足两目标。C0高AUROC不推翻closed上界。均为本项目数学推断，未计算oracleROC／挑target阈值，不推广为所有未来模型不可能。

## 运行证据、时长与完整索引

一次独立P0/P1发现真实父fit路径漏/source的问题，修复后一次定向核对PASS；覆盖32 unique tests／68 subtests。dev原7tests＋24sub与realpath修复及邻近定向3tests＋10sub分别记录，没有重复整套审查／测试。一次preflight、单归档本地／远端比较、远端compile通过；不新增报告门槛。

实际cache checkpoint/G smoke先于真实cache角色；每arm native known/OE梯度smoke先于训练角色，均Sinc/head非零且state/RNG恢复；source实际新checkpoint2合成IQ／8view先于新L，C0/P1各合成smoke先于Q。完成标记与完整log保存PASS，不能把合成smoke当源优化／目标曝光。

| 整子进程墙钟 s | C（或共享） | P |
| --- | --- | --- |
| 共享cache | 126.8942 | — |
| train | 107.1370 | 91.8815 |
| source | 455.1557 | 459.7622 |
| source_score | 4.2658 | 4.4647 |
| C0_predict | 4.7982 | 4.3557 |
| C0_score | 2.3082 | 2.1852 |
| P1_predict | 7.5632 | 7.3192 |
| P1_score | 2.3070 | 2.5493 |
| 全pipeline（两row并行） | 712.9620 | — |

训练内部实际优化计时C101.4671s／P86.8424s，与表中含构建／smoke／保存的train进程分开；不同GPU与并发负载下的差异不作因果加速或逐包延迟。两source正常各162600个新F forward、120600原F0 forward、121950个G/channel计算，另计合成smoke；开发父预测／F0 baseline复用，没有重复父Q前向。

独立live核实总driver2673776、row drivers2675837／2675840、训练PID2675845／2675846及CWD／命令；训练C GPU7／P GPU4各采样1540MB，cache采样800MB，source阶段采样1286／1356MB，均非峰值。完整log增长、15jobs完成及最终owned已退出可核查，foreignstop0；不将其他owner占用或整卡列表归成本run。

已完整读取85份元数据／日志、444步trace和800个loss点；全部六份评分共1834指标分组的计数、macro、RX／TX／view加总、父exact reuse及delta均核对通过。明细索引覆盖全部七方法，以下<arm>为clean_increment或practical_increment：

- [C source完整评分](E:/type10-7/local_artifacts/pamf_phase3_source_practical_encoder_20261029_r01/rows/clean_increment/source/score/scores.json)、[P source完整评分](E:/type10-7/local_artifacts/pamf_phase3_source_practical_encoder_20261029_r01/rows/practical_increment/source/score/scores.json)：methods.<method>.{overall,by_view,by_receiver,by_transmitter,by_receiver_view,by_transmitter_view,goals}，含五RX／四view／TX及交叉。
- rows/<arm>/development/{C0,P1}/score/scores.json：states.{parent_E51,increment}.methods.<method>.{overall,by_scenario,by_transmitter,planned_discovery_unknown,planned_later_unknown}和deltas；四条row均完整评分。
- rows/<arm>/train/{steps.jsonl,complete.json}、source/fit/training.json、source/calibration/complete.json及source／development完成标记，保留完整曲线、histogram、冻结／truth-last与校准可行性。
- [最终post-state](E:/type10-7/local_artifacts/pamf_phase3_source_practical_encoder_20261029_r01/post_state.json)、[完整观察](E:/type10-7/local_artifacts/pamf_phase3_source_practical_encoder_20261029_r01/observations.jsonl)、两row pipeline_state.json与完整logs，保留实际PID/GPU／日志增长／退出证据。

该固定预算源端星地联合增量没有通过新RX的身份与开放集联合保护；保持NO_PROMOTION，不换主arm、不调target阈值、不按单项扩seed。下一候选尚未启动，正式U/Q及未来无标签发现／匿名注册仍需独立证据，总Phase3保持未完成。
