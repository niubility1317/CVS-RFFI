# G10 源端预测类别条件风险结果（2026-10-29）

**唯一主 `predicted_class`：NO_PROMOTION；宏 A_K/R_U/H-score＝90.0593/59.6508/71.7668%。** Known42保护通过；按实际评分，unknown必需12/12组未达90%。本轮target/DA/发现/注册访问0，完整Phase3目标保持ACTIVE。

run `20261029-phase3-star-class-risk-s2026092701-r01`，code `bb507e2b6f4d66bd808a485d70d64d25c3556da3`，预登记 `a23d46b3e54b35db25f5c90f8e8f5b3458b92929`。本轮只在冻结父 `paired_mlp16_step1000` 的同一rawscore上重校准阈值，未新训练或改变身份预测。

## 数据、权限与阈值机制

祖先为原Phase1 Tri-SIG-TIME F0 E50/11100（232993参数）与DURA v1（266951参数）；原source L6300/U56700/V27000，原训练标签比例L/(L+U)=10%。父读出曾用星地L6300和OE21设备6134条的额外known/outside-six成员监督，两条匹配MLP轨迹各1000步，共2000更新；这些训练、geometry和normalizer成本与标签来源保留，不因本轮cal-only而归零。

本轮只复用父已完成的knownV27000及novel3150标量分数/原P1闭集预测。V额外known标签用于经验风险校准；novel3只在全部阈值冻结后生成两列margin，完整预测后才独立truth-side评分。没有新L/OE训练输入、raw/clean/IQ、F/G前向、channel生成、模型/normalizer/geometry拟合或新物理记录；原U_s与正式RX24-18未访问。

source RX index1/3/4/6/8＝1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23；旧六类为14-10、14-7、20-15、20-19、6-15、8-20。novel为19-15/19-16/7-1各1050条，每TX×RX×day70条；它们与known同源RX/day且已多次源开发暴露，不是未见RX target或新TX盲测。底层缓存的P0/P1都是既有G10星地观测，不以两列margin增加独立样本数。

global_control重算一个global τ；主法用冻结原六类head的预测c选择六个τ_c。两列分别为s−τ和s−τ_c，均在margin≥0时接收、margin<0时拒绝；等于0的tie接收。路由只看模型c，不看真实TX、RX、day或scene；这些真值/分组只用于合法known校准审计和独立评分。

六阈值以公共尾率q控制各预测桶：每桶n_c只可拒绝不超过floor(q*n_c)的完整score-tie前缀。候选q是六桶完整前缀比例的并集，使用Fraction精确计数；在known42组CK≥90%和原F0同输入P1 closed−AK≤2pp内取最大合法q。它是同一V上的经验校准，不是独立conformal覆盖保证，也不对novel或target给出概率保证。

主报A_K为真实旧TX逐类“正确且接收”的宏平均，C_K只计接收；R_U为三个真实novel TX拒识宏平均，H=2A_KR_U/(A_K+R_U)，不设独立H硬门槛。预测桶统计不是这些真实TX宏指标。unknown验收仍为overall/5sourceRX/6scene各≥90%，RX×scene未知只诊断；唯一主不能由global_control替换，A_N/H_OW=N/A。

## 总体与固定父对照

| 方法 | A_K% | R_U% | H-score% | C_K% | F0损pp | margin宏AUROC% | margin宏OSCR% | sourcegate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_control | 90.3593 | 43.7143 | 58.9227 | 98.4778 | 0.6222 | 94.2525 | 86.6136 | False |
| predicted_class | 90.0593 | 59.6508 | 71.7668 | 97.8407 | 0.9222 | 96.0749 | 88.5425 | False |

共同原F0 P1 closed=90.9815%；主接收26417/27000、正确接收24316/27000、未知拒识1879/3150，仍有1271条未知被接收。被接收known中2101条身份错误，不能把CK=97.8407%写成识别准确率。

global_control的τ、known/novel接收集合及指标精确复现父paired1000；producer记录global_control_matches_parent=True，本轮global AK/RU/H=90.3593/43.7143/58.9227%。本报告不重新打开父旧score或truth；两列同rawscore、同opaqueID、同closed/baseline的关系已在本轮产物逐条核对。

等量总体记录的Wilson95%描述区间：AK 89.70–90.41%、CK 97.66–98.01%、RU 57.93–61.35%。这些区间不覆盖单seed、同源RX、历史重复物理和配对观测相关性，不是跨RX或target总体置信区间；不把不等量scene pooled区间当宏区间。

## 公共q、六阈值与42组保护

公共q=97/4473（2.168567%），候选数26984，选中索引583（0起）。六桶预测n、cutoff、floor预算及实际完整前缀拒绝数如下；实际被拒数同时从已完成margin重新计数。

| 预测head桶c | V n_c | rawscore cutoff τ_c | floor(q*n_c) | 实际拒绝数 | 桶拒绝率% | 其中原身份正确被拒 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 4252 | 1.20813429 | 92 | 92 | 2.1637 | 48 |
| 1 | 4241 | 0.349865913 | 91 | 91 | 2.1457 | 39 |
| 2 | 4473 | -0.319329709 | 97 | 97 | 2.1686 | 37 |
| 3 | 4958 | 1.12846076 | 107 | 107 | 2.1581 | 49 |
| 4 | 4506 | -3.10627317 | 97 | 97 | 2.1527 | 46 |
| 5 | 4570 | -1.51278985 | 99 | 99 | 2.1663 | 30 |

六桶合计拒known583/27000。上述n_c并非真实TX数量；例如预测桶可包含错分的其他旧TX。所有cutoff都由V的预测桶前缀得出，不让unknown真实类别或分数参与。

下一q=92/4241（2.169300%）已由原calibration记录为不合法，仅预测桶1的cutoff从0.349865913变到0.365173429。失败组为rxscene/6/practical_low_urban，原F0−open损为2.0316pp，超过2pp。这里仅核验已登记的下一候选与失败理由，没有重新搜索阈值或把它应用到novel。

| 方法 | 零拒绝可行 | 42组通过 | 最大known FRR% | 最大F0损pp | fallback |
| --- | --- | --- | --- | --- | --- |
| global_control | True | True | 4.1833 | 1.9767 | False |
| predicted_class | True | True | 5.3350 | 1.9844 | False |

实际known失败/不可认证0/42，全部42组缺必需TX 0组。完整两方法×42校准及score组已解析；下表覆盖overall、全部5RX、6scene及30交叉组，global/主单元格为AK/RU/H（%），未知交叉组只诊断。

| 组 | global AK/RU/H% | 主 AK/RU/H% | 主CK% | 主F0损pp | coverage K/U |
| --- | --- | --- | --- | --- | --- |
| overall | 90.3593/43.7143/58.9227 | 90.0593/59.6508/71.7668 | 97.8407 | 0.9222 | True/True |
| rx/1 | 92.6852/52.6984/67.1928 | 92.4815/77.4603/84.3070 | 98.9074 | 0.6111 | True/True |
| rx/3 | 84.3519/29.5238/43.7387 | 84.1667/41.2698/55.3833 | 96.0556 | 0.9815 | True/True |
| rx/4 | 91.2407/55.8730/69.3055 | 90.5926/64.1270/75.0962 | 97.8148 | 1.0370 | True/True |
| rx/6 | 91.7963/54.7619/68.5999 | 91.7222/77.7778/84.1764 | 98.3519 | 1.1111 | True/True |
| rx/8 | 91.7222/25.7143/40.1676 | 91.3333/37.6190/53.2890 | 98.0741 | 0.8704 | True/True |
| scene/practical_high | 97.4508/58.8235/73.3633 | 97.1982/72.1505/82.8220 | 98.3770 | 0.7147 | True/True |
| scene/practical_mid | 97.0974/52.6431/68.2716 | 96.6630/67.1233/79.2293 | 98.3459 | 0.7797 | True/True |
| scene/practical_low_suburban | 95.0673/54.1008/68.9587 | 94.7611/70.3265/80.7355 | 98.1061 | 0.8438 | True/True |
| scene/practical_high_urban | 96.6912/48.7043/64.7789 | 96.3560/67.3058/79.2527 | 98.1436 | 0.5964 | True/True |
| scene/practical_mid_urban | 91.3447/40.5235/56.1410 | 91.1536/58.0662/70.9416 | 98.4550 | 0.5603 | True/True |
| scene/practical_low_urban | 82.7776/30.1571/44.2084 | 82.4819/45.6084/58.7378 | 97.2873 | 1.1406 | True/True |
| rxscene/1/practical_high | 98.5692/74.6032/84.9278 | 98.5692/100.0000/99.2794 | 99.5495 | 0.0000 | True/True |
| rxscene/1/practical_mid | 98.5692/65.2661/78.5329 | 97.9265/88.5154/92.9834 | 99.3572 | 0.6428 | True/True |
| rxscene/1/practical_low_suburban | 96.9686/63.8971/77.0333 | 96.7677/90.1471/93.3401 | 98.9117 | 0.4914 | True/True |
| rxscene/1/practical_high_urban | 99.2908/65.7143/79.0863 | 99.2908/100.0000/99.6441 | 99.6454 | 0.0000 | True/True |
| rxscene/1/practical_mid_urban | 93.7008/52.5964/67.3742 | 93.7008/76.3521/84.1415 | 98.8417 | 0.2451 | True/True |
| rxscene/1/practical_low_urban | 85.8602/36.3337/51.0602 | 85.6554/58.7697/69.7101 | 98.7110 | 0.9088 | True/True |
| rxscene/3/practical_high | 92.6023/55.5556/69.4472 | 92.2336/69.4444/79.2329 | 96.0207 | 1.5693 | True/True |
| rxscene/3/practical_mid | 92.8937/41.1184/57.0044 | 92.3381/54.1667/68.2797 | 94.7308 | 1.8333 | True/True |
| rxscene/3/practical_low_suburban | 91.1623/35.2612/50.8528 | 90.8782/49.9048/64.4290 | 96.4826 | 1.0176 | True/True |
| rxscene/3/practical_high_urban | 91.5023/36.1111/51.7853 | 91.5023/61.5741/73.6125 | 94.6650 | 0.3268 | True/True |
| rxscene/3/practical_mid_urban | 87.0466/27.7778/42.1158 | 87.1220/37.5000/52.4318 | 96.9545 | 1.0137 | True/True |
| rxscene/3/practical_low_urban | 74.2242/20.0272/31.5434 | 74.1595/25.6573/38.1245 | 95.9501 | 0.7198 | True/True |
| rxscene/4/practical_high | 96.7445/73.1481/83.3077 | 96.3380/77.3148/85.7844 | 96.9790 | 1.5105 | True/True |
| rxscene/4/practical_mid | 98.1496/63.3333/76.9882 | 97.3387/69.2593/80.9326 | 98.6471 | 0.8109 | True/True |
| rxscene/4/practical_low_suburban | 95.4797/70.2842/80.9672 | 94.9888/73.9341/83.1493 | 97.9113 | 1.0658 | True/True |
| rxscene/4/practical_high_urban | 98.2807/53.8889/69.6097 | 96.7115/60.5556/74.4774 | 98.0339 | 1.5693 | True/True |
| rxscene/4/practical_mid_urban | 92.2751/53.7037/67.8936 | 91.9444/65.9259/76.7911 | 98.8757 | 0.5622 | True/True |
| rxscene/4/practical_low_urban | 84.2561/38.6760/53.0161 | 83.5242/52.2388/64.2768 | 97.4421 | 1.0217 | True/True |
| rxscene/6/practical_high | 100.0000/72.5926/84.1202 | 100.0000/89.6296/94.5312 | 100.0000 | 0.0000 | True/True |
| rxscene/6/practical_mid | 97.9515/72.6557/83.4283 | 97.6951/89.8718/93.6203 | 99.1513 | 0.8487 | True/True |
| rxscene/6/practical_low_suburban | 96.2669/62.3336/75.6702 | 96.1159/86.0208/90.7886 | 98.9380 | 0.6296 | True/True |
| rxscene/6/practical_high_urban | 98.3415/70.0926/81.8482 | 98.3415/83.7963/90.4881 | 99.1708 | 0.4505 | True/True |
| rxscene/6/practical_mid_urban | 92.6733/50.9171/65.7238 | 92.6733/78.1200/84.7766 | 99.4675 | 0.2874 | True/True |
| rxscene/6/practical_low_urban | 84.1157/40.5609/54.7305 | 84.1081/64.8896/73.2594 | 97.1333 | 1.9844 | True/True |
| rxscene/8/practical_high | 98.4131/35.3683/52.0357 | 98.0163/48.5348/64.9221 | 99.1646 | 0.3968 | True/True |
| rxscene/8/practical_mid | 97.5333/28.3272/43.9033 | 97.5333/43.5897/60.2517 | 99.4350 | 0.0000 | True/True |
| rxscene/8/practical_low_suburban | 95.3884/36.2731/52.5595 | 94.9568/49.1943/64.8116 | 98.2236 | 1.0122 | True/True |
| rxscene/8/practical_high_urban | 95.9117/17.6768/29.8518 | 95.9117/31.0101/46.8672 | 99.2578 | 0.3876 | True/True |
| rxscene/8/practical_mid_urban | 91.1615/22.4974/36.0887 | 90.4647/34.0764/49.5051 | 98.0209 | 0.6968 | True/True |
| rxscene/8/practical_low_urban | 85.8042/15.3713/26.0720 | 85.3628/25.4744/39.2389 | 97.4150 | 1.0270 | True/True |

按实际组值，未知必需失败组12/12：overall, rx/1, rx/3, rx/4, rx/6, rx/8, scene/practical_high, scene/practical_mid, scene/practical_low_suburban, scene/practical_high_urban, scene/practical_mid_urban, scene/practical_low_urban。known保护通过、AUROC更高或某交叉组RU较高均不能替代这些必需组。

## 真实TX宏细分与配对净变化

逐TX下表旧类为A_K/C_K，未知为R_U（%）；这是真实TX评分，不按预测桶重加权。

| 真实TX | N | 原F0 P1 closed% | global A/C或R% | 主 A/C或R% |
| --- | --- | --- | --- | --- |
| 14-10 | 4500 | 89.0444 | 88.6000/98.4222 | 87.9778/97.4667 |
| 14-7 | 4500 | 80.3333 | 80.0667/98.6889 | 79.4667/97.6667 |
| 20-15 | 4500 | 95.9111 | 95.5556/99.1111 | 95.0889/98.3556 |
| 20-19 | 4500 | 85.4667 | 85.3556/98.8889 | 84.3778/97.5333 |
| 6-15 | 4500 | 96.3333 | 94.1333/96.5556 | 95.3111/97.2000 |
| 8-20 | 4500 | 98.8000 | 98.4444/99.2000 | 98.1333/98.8222 |
| 19-15 | 1050 | — | 33.9048 | 52.8571 |
| 19-16 | 1050 | — | 31.9048 | 64.4762 |
| 7-1 | 1050 | — | 65.3333 | 61.6190 |

| 主−global | ΔAK pp | ΔRU pp | ΔH pp | ΔCK pp | ΔF0损pp | ΔAUROC pp | ΔOSCR pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 预测类阈值效应 | -0.3000 | 15.9365 | 12.8441 | -0.6370 | 0.3000 | 1.8225 | 1.9288 |

| 同ID配对事件 | 仅主 | 仅global | 净主−global |
| --- | --- | --- | --- |
| known接收 | 72 | 244 | -172 |
| known正确且接收 | 53 | 134 | -81 |
| unknown拒识 | 630 | 128 | 502 |

主净多拒502条未知、净少正确接收81条known，宏RU/H增加15.9365/12.8441pp，AK下降0.3000pp。预测桶4的τ低于global而其余五桶更高：较松的一桶恢复72条known（其中53条正确），同时放过128条原先会拒的novel；其余桶更严，额外拒244条known（134条原本正确）及630条novel。净效应是阈值风险分配改变，不是F0、G或身份判别能力提高。三个真实novel TX的拒识净变化分别为19-15 +199条、19-16 +342条、7-1 −39条；宏RU提升不代表每个新类都改善，也不是未来注册能力证据。

主margin AUROC/OSCR=96.0749/88.5425%，较global高1.8225/1.9288pp。逐预测类平移会改变不同桶样本之间的排序，因此必须用本轮margin重算曲线，不能沿用父rawscore AUC；这些曲线也不能证明已冻结0运行点达到90%拒识。总体RU仍59.6508%，5RX及6scene均未过线；没有oracle曲线选阈值或放宽known保护。

## 成本、完整日志与运行证据

本轮新增神经更新、F/G前向、IQ读取、信道生成和GPUworker均0；未重读L/OE IQ、拟合normalizer或新读出。实际父readout checkpoint的两条合成200D特征smoke覆盖四个3233参数快照，先于源预测行读取PASS；这是NumPy feature/readout smoke，不是新星地IQ、F0 checkpoint或G smoke。祖先实际F0/G smoke及父2000步训练保持历史身份。

主部署参数仍为共享F0 232993＋DURA266951＋paired MLP3233=503177，双输入需要两次共享F前向，不能算两个F参数。新增主部署状态仅六个FP64 τ=48字节；global对照另一个FP64 τ=8字节，本轮合7标量56字节，不是新增NN参数。原两path geometry、400个FP64 paired normalizer系数、读出权重仍需保留；临时CDF/前缀及V逐行状态不部署。

| stage | 实际PID | 设备 | 整进程秒 | 终态 |
| --- | --- | --- | --- | --- |
| source_predict | 2955662 | CPU | 3.129533 | COMPLETE |
| source_score | 2955731 | CPU | 0.465889 | COMPLETE |

全pipeline3.598202s，含父聚合载入、known校准、margin写入、独立评分和IO，不是逐包部署时延或相对GPU版加速证据。source/score完整stdout均带真实PID/CWD/argv；独立live捕获[2955649, 2955662]的精确CWD/argv，CUDA_VISIBLE_DEVICES为空；CPU_only=True、新GPUworker0。source日志live485B→终态607B，终态live为空、foreignstop0；没有GPU占用或峰值证据，也无需虚构GPU采样。

实际code/prereg见开头；archive `bf082e5107c090753196d26c3b9bb7cda3c8385971b5d22a909e3c711d25300a` 的一次local/remote比对、remote compile/runtime均VERIFIED。一次独立P0/P1审查通过，24项相关检查/45子项的历史记录保留。本地完整测试log包括预期RED及最终GREEN，不能把TDD的模块未实现失败写成实际run失败；本次报告没有重跑模型或实验。

全量读取14份JSON、2条JSONL、5份完整log；两列known27000/novel3150的全部margin、2×42校准和score组、真TX细分、42组配对差/计数、六桶前缀预算、26984个known-only候选尾率和已登记nextq失败组均核对。未打开raw/parent truth文件、F/G权重或旧run评分文件。

完整索引：[`score/results.json`](../local_artifacts/pamf_phase3_star_class_risk_20261029_r01/score/results.json) 的 `methods/paired_primary_vs_global`；[`prediction/calibration/complete.json`](../local_artifacts/pamf_phase3_star_class_risk_20261029_r01/prediction/calibration/complete.json) 的 `common_tail_rate/methods/next_failure_groups`；`prediction/{thresholds.npz,smoke.json,complete.json}`、`observations.jsonl`及完整`logs/`。

结论：NO_PROMOTION。预测类别条件阈值在固定源分数上改善未知拒识与H，但未达星地RU90目标，并付出81条正确known接收的净损失。该源RX、单seed、历史novel的经验收益不代表跨RX target、DA、匿名发现或未来注册；A_N/H_OW=N/A，正式RX24-18未访问，总目标ACTIVE。没有按novel改q、选阈值、换主或启动下一实验。
