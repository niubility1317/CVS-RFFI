# Phase3 原源域 Us 星地重放与伪标签结果（2026-10-31）

**NO_PROMOTION，原 PCD 研究基线保留，完整目标 ACTIVE。** 新主 SOURCE_US_predicted_ratio 的星地 A_K 为27.490040%、R_U为49.835372%：较纯源SOURCE的A_K增加0.738703个百分点，却损失2.434607个百分点R_U；对OLD_PCD的A_K下降0.461682个百分点。额外源Us重放与PL未同时满足相对控制和绝对目标，不能由源拒识改善或教师接收率宣布有效。

Run `20261031-phase3-source-us-replay-s2026092701-r01`；actual code `3988af354d00a3453384d7f61a799a4a237084fc`；cache/train/predict/score均COMPLETE/0，pipeline `ANALYZED_DEVELOPMENT`。方法及权限见[最终计划](PAMF_PHASE3_SOURCE_US_REPLAY_PLAN_20261031.md)。

## 方法、数据与祖先

SOURCE_US从source-only r03独立初始化，源L/OE目标沿SOURCE444：crossset metric＋OE结构KL＋10×父z锚点，再增加权重1的Us置信度PL。固定F0对弱P1的最大概率≥0.95才接收其argmax；学生同观测相位/0.95至1.05增益强view与r03源L六单位均值计算cosine/0.1训练分数，按完整128批平均。没有EMA、新身份头或Us真实标签。训练锚点不部署，身份及密度/τ路由始终是原F0/default6。

新副本body231985＋projection102688共同更新444步，Sinc48/buffers冻结；原F0、G、r03和旧学生不更新。feature祖先11100＋r032220＋本轮444累计13764，projection2220→2664。SOURCE控制仅源数据；OLD_PCD继承目标U576适应。新训练未访问target U，不能将两种权限合称预算匹配或新增目标UDA。

身份模型为原 Tri-SIG-TIME SSL F0 E50/11100 及其联合训练的六类默认头。恢复器固定为 DURA v1.0 R69，266951 参数、E20/4940，沿用 `reuse=True, graph=True, native=False`、FP32/TF32off；本轮不训练或更换恢复器。

原source合同392005沿用L6300/Us56700/V27000、六旧TX、五RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、03-08/15/23、eq1，OE候选池6134。Us是原角色，不是新增独立物理记录；原F0已有clean Us半监督祖先，新使用限本轮Phase3星地P1增量。

可信离线builder枚举90000源坐标并按原role_ids选Us56700。ManySig whole PKL实际反序列化且含其他角色载荷；builder只索引Us记录值，不把L/V值送入本轮Us观测构建，不导出TX/y、eq/packet坐标或定位表。Us真实标签读取0，公共metadata恰为RX/day/domain。各记录只生成一次固定G10 received并经一次DURA，input RMS56700次、output RMS0，不二次归一化或EQ。

公共Us句柄为`pamf-source-Us392005-G10-MIX01-20261031-r01`/`392005-Us-source5rx-3day-eq1`，VALIDATED_ONCE；MIX01 scene2026103001/channel2026104001/receiver2027、600 km/25 MHz/2.45 GHz、residual/post_sync/noEQ/refSNR30 dB/reflect保持。公共仅manifest/ids/received/restored/metadata五文件，无Us私有真值。

评估源L6300/OE6134拟合一套新known90/OE32/normalizer，source V27000只校准两新六τ；NV3150及target RX3-2/index34 Q10238（旧4508、未知四类5730）均不拟合。SOURCE/OLD_PCD四列密度、τ、特征及预测精确复用fresh；source V非独立已知测试。两个Q同canonical、均经DURA，clean仅冻结测试；03-15/23分别5525/4713条。

旧 PCD 的 U576 来自 RX3-2 的 2021-03-01：六旧类与三类可发现设备 5-5、10-7、2-19 各64条。7-11 从这个早日 U 整类排除，作为相对于该 U 的未来专有未知；本轮只复用旧 PCD 预测，不新读取 U。未来 Q 六旧类按原顺序分别为1491/1182/150/406/1186/93条，三类可发现设备共4320条，7-11为1410条；旧 U 与未来 Q 物理记录互斥。当前“未知”指未注册设备，不证明所有训练祖先均未接触其 TX。

## 完整444步训练与教师PL

444步1..444连续，全部数值有限，L/OE索引每步与SOURCE记录严格匹配。实际曝光L85248/OE28416/Us56832；唯一L6300、OE5913、Us56700。OE池6134不等于全部被训练采样，评估背景拟合才遍历全部6134。Us全池一次＋132条重复，接受曝光47781/56832=84.074113%。

每步`source_loss=crossset_metric+OE_KL+10×source_anchor`、`loss_total=source_loss+Us_PL_loss`；全444步最大绝对残差分别5.40e−8/5.96e−8。PL从第1步启用；不同总目标不能与无PL的SOURCE按loss大小排序。下表是完整三个148步段的均值，并非首尾抽样。

| 步段 | 总loss | crossset | OE KL | anchor（未乘10） | source_loss | Us PL | clip前范数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–148 | 1.077490 | 0.300145 | 0.069397 | 0.006184 | 0.431379 | 0.646111 | 7.053371 |
| 149–296 | 0.944596 | 0.301034 | 0.072737 | 0.006847 | 0.442238 | 0.502358 | 6.606174 |
| 297–444 | 0.874681 | 0.290995 | 0.072312 | 0.007671 | 0.440020 | 0.434662 | 7.201577 |

总loss首1.503126→末0.790456，全程最小0.687480（step428）、最大1.681729（step4）。Us PL首1.146314→末0.428239；source_loss首0.356812→末0.362216，anchor末0.008213。总目标下降主要伴PL项下降，不能据此推断未知分离或目标泛化改善。

AdamW固定lr1e−4/wd1e−4/eps1e−8，ordinary clip5共444次，其中clip前norm>5为394步；全程范数均值6.953707、范围3.678754至16.005602（最大step98）。Sinc optimizer及physical clip均0。模块梯度仅记录前2步：body6.591821/9.062161、MLP hidden1.111835/0.979625、last2.516500/1.623351，Sinc与固定原型均0；不能外推444步逐模块梯度。冻结原模型、学生Sinc/buffers均保持，学生body/projection实际改变。

F0教师在完整Us56700中接收47668（84.070547%）。这是固定教师置信mask接收率，**不是伪标签准确率**；本轮没有Us真实TX，无法评价伪标签精度。下表均按教师预测类计数，不是真实类配额。

| 教师预测类（原六类顺序） | 全池预测数 | conf≥0.95接收数 |
|---|---:|---:|
| 14-10 | 8911 | 7908 |
| 14-7 | 8972 | 6381 |
| 20-15 | 9435 | 8529 |
| 20-19 | 10277 | 6428 |
| 6-15 | 9522 | 9162 |
| 8-20 | 9583 | 9260 |

Us抽样/增强分别使用独立2026103150/2026103151流，不扰动原L/OE索引。虽同为444步且L/OE采样匹配，新增Us输入、教师前向与每步计算使完整数据/计算预算不匹配SOURCE，亦不匹配带target U576祖先的OLD_PCD。

## 六方法 × 三DURA条件

下表均为逐TX等权宏百分比。原F0同DURA闭集依source/clean/star为90.981481/48.567015/41.721059%，六方法相同；新主没有身份纠错机制。

| 方法 | 条件 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|
| SOURCE_pooled_ratio | source_DURA | 89.777778 | 97.088889 | 60.063492 | 71.974388 |
| SOURCE_pooled_ratio | clean_DURA | 31.475324 | 44.146095 | 62.409528 | 41.846157 |
| SOURCE_pooled_ratio | star_DURA | 27.422108 | 48.651740 | 51.188933 | 35.712755 |
| SOURCE_predicted_ratio | source_DURA | 89.740741 | 96.744444 | 61.174603 | 72.754089 |
| SOURCE_predicted_ratio | clean_DURA | 31.078517 | 43.181373 | 63.911653 | 41.820736 |
| SOURCE_predicted_ratio | star_DURA | 26.751337 | 47.266339 | 52.269979 | 35.390244 |
| OLD_PCD_pooled_ratio | source_DURA | 89.770370 | 97.081481 | 63.619048 | 74.465443 |
| OLD_PCD_pooled_ratio | clean_DURA | 32.123647 | 44.791196 | 60.790839 | 42.034854 |
| OLD_PCD_pooled_ratio | star_DURA | 29.175335 | 50.230381 | 48.454372 | 36.420916 |
| OLD_PCD_predicted_ratio | source_DURA | 89.733333 | 96.748148 | 64.285714 | 74.907247 |
| OLD_PCD_predicted_ratio | clean_DURA | 31.661804 | 43.761391 | 61.979647 | 41.912794 |
| OLD_PCD_predicted_ratio | star_DURA | 27.951722 | 48.582412 | 49.307698 | 35.678111 |
| SOURCE_US_pooled_ratio | source_DURA | 89.807407 | 97.200000 | 66.000000 | 76.084815 |
| SOURCE_US_pooled_ratio | clean_DURA | 31.492840 | 43.980937 | 62.244610 | 41.824469 |
| SOURCE_US_pooled_ratio | star_DURA | 27.833368 | 49.378186 | 48.965665 | 35.492097 |
| SOURCE_US_predicted_ratio | source_DURA | 89.770370 | 96.862963 | 66.952381 | 76.700287 |
| SOURCE_US_predicted_ratio | clean_DURA | 31.175965 | 42.939813 | 63.625483 | 41.847163 |
| SOURCE_US_predicted_ratio | star_DURA | 27.490040 | 47.992691 | 49.835372 | 35.434053 |

主对同DURA原F0的A_K损伤依source/clean/star为1.211111/17.391051/14.231018个百分点。source known12通过，但target cleanRU95/starRU90、C_K90、原F0闭集损伤≤2pp均未满足，星地A_K也未高于两控制且RU不降；无论pooled局部成绩如何都不换主。

| 主−控制 | 条件 | 正确接收gain/loss/net | 接受net | 未知拒识gain/loss/net | ΔA_K pp | ΔR_U pp |
|---|---|---|---:|---|---:|---:|
| SOURCE | source_DURA | 46/38/8 | 32 | 276/94/182 | 0.029630 | 5.777778 |
| SOURCE | clean_DURA | 22/29/-7 | -19 | 145/160/-15 | 0.097448 | -0.286170 |
| SOURCE | star_DURA | 28/23/5 | 7 | 85/223/-138 | 0.738703 | -2.434607 |
| OLD_PCD | source_DURA | 39/29/10 | 31 | 162/78/84 | 0.037037 | 2.666667 |
| OLD_PCD | clean_DURA | 24/22/2 | -5 | 173/79/94 | -0.485839 | 1.645835 |
| OLD_PCD | star_DURA | 17/20/-3 | -5 | 125/95/30 | -0.461682 | 0.527673 |

星地对SOURCE正确净＋5但未知拒对净−138；对OLD_PCD正确净−3、未知拒对净＋30，宏H仍下降0.244058pp。clean对SOURCE正确净−7而宏A_K略增0.097448pp，源于各类容量不等；不能用总正确条数替代六类宏指标。

## 分组失败与覆盖

| 主星地场景 | A_K% | C_K% | R_U% | 原F0损伤pp |
|---|---:|---:|---:|---:|
| practical_high | 28.674547 | 42.803520 | 58.625225 | 13.184660 |
| practical_mid | 26.258401 | 40.408688 | 56.500127 | 22.718010 |
| practical_low_suburban | 30.233611 | 46.065013 | 51.917426 | 16.752692 |
| practical_high_urban | 37.844153 | 54.027124 | 53.951606 | 15.409517 |
| practical_mid_urban | 32.779665 | 47.054059 | 49.409051 | 17.019392 |
| practical_low_urban | 22.836305 | 50.909559 | 45.068610 | 10.095886 |

六主景总体类覆盖完整，但联合门槛全失败。星地六旧TX依14-10/14-7/20-15/20-19/6-15/8-20的主A_K为44.936284/12.098139/3.333333/1.970443/58.516020/44.086022%；C_K中20-15仅28%、20-19仅8.620690%，不能靠高接受率掩盖身份与拒识责任。

星地四未知5-5/10-7/2-19/7-11各1454/1439/1427/1410条，主R_U分别33.081155/59.555247/49.754730/56.950355%；clean对应35.625860/76.372481/74.702172/67.801418%。source NV19-15/19-16/7-1各1050条，R_U64.952381/68.095238/67.809524%，仅为已曝光源诊断。

03-23缺旧20-19与8-20，完整A_K/C_K/H保留N/A，RU独立可报clean63.751389%/star49.737621%；03-15主A_K为clean31.529938%/star27.621779%。不缩小旧类名单认证。source五RX及六主景的known12通过，但30交叉诊断中RX3/high、RX3/mid_urban、RX3/low_urban、RX4/high_urban、RX8/low_urban原F0损伤超2pp，known42未全过。

source主AUROC95.231945%/OSCR88.251692%，star51.084315%/26.677712%；源曲线或源RU改善不能替代未来联合性能。全部TX/RX/day/场景、配对、覆盖与逐类Wilson95%区间见[既有score](../local_artifacts/pamf_phase3_source_us_replay_20261031_r01/score/results.json)和[完整分析](../local_artifacts/pamf_phase3_source_us_replay_20261031_r01/analysis.json)，未补读真值。

## 成本、执行与边界

新增真实signal245990=父r03源L/OE12434＋学生训练170496＋评估63060；真实F0/G各77176=Us56700＋Q20476（F0为训练教师、G为缓存恢复）。评估SOURCE_US仅新学生：源L/OE/V/NV42584＋两Q20476，Q同时F0/G各20476；旧SOURCE/PCD四列无新前向。

合成烟测按实际口径分列：cache F0=4（两原始合成＋两恢复后合成）、G=2、signal=0；train F0/G各2、父signal2＋学生梯度smoke8；predict F0/G/新signal各2。全链合成F0=8、G=6、signal=12，独立于真实前向及物理样本数；不把cache的4误标为含r03。

新source统计为1套known90/OE32/normalizer与2×6τ；旧4×6τ精确复用，共36值/288 B、单主6τ48 B。神经部署F0 232993＋新signal334721＋G266951=834665；原型仅训练、不部署Us/source行bank。F011100/r032220、SOURCE444及OLD_PCD444/U576、G4940和既有缓存成本继承，不能只报告新增444而抹去祖先。

| 阶段 | 实际walltime（s） | PID | 独立live证据 |
|---|---:|---:|---|
| cache | 183.016036 | 1749355 | CUDA5，404→470 MiB |
| train | 75.099340 | 1751664 | CUDA5，478→868 MiB |
| predict | 28.825354 | 1752350 | CUDA5，446 MiB |
| score | 18.458417 | 1752508 | CPU/CUDAempty，完整argv/CWD已核 |
Pipeline 305.686015 s；GPU UUID ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331均为离散采样，非峰值/B1。11次观测含driver1749339及各阶段独立PID/CWD/argv；日志cache498→1635 B、train498→1476 B、predict496→534 B，最终live/ownedGPU空、foreign0。

五份完整stdout合计59行，warning/error均0，原始444条训练记录全读。本轮执行无真实技术失败。验证17unique＋28sub、10 AST/5 CLI/实际config PASS；一次P0/P1曾拦截launcher的冻结字段名不符，修复后仅实际producer→terminal原P1定点复审PASS；旧green不重复计数。
包784files/data0，SHA256 `c2dec3fd4d6a94845804c03268048867a6add06d653970d1287cfa4fc1d2e294` 单次比对、remote compile/runtime VERIFIED。新增Us数据由本次cache在N607生成，不将发布data0误写为本轮没生成数据。

全程新Us真实标签0、sourceNV/两Q仅完整预测后独立真值连接各一次，Us/U truth0。source V非blind，NV/Q历史曝光、source日期重合、单RX/seed/global physical history UNKNOWN、正式RX24-18未读保持。无新targetUDA/发现/匿名REG，adaptation_success与A_N/H_OW均N/A；新Us PL未确认对旧PCD的联合收益，不换基线或从本Q挑参数。
