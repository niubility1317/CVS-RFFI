# PAMF P0开放分支与P1身份分工：匹配训练结果

**执行 VERIFIED，科学结论 NO_PROMOTION。** 唯一主 P0_known12：宏 A_K 90.048148%、C_K 98.503704%、R_U 3.936508%、H-score 7.543258%；相同原F0/P1闭集损失 0.933333 个百分点。相较固定P1_known12，R_U变化 -42.952381 个百分点、A_K变化 0.100000 个百分点。hard12通过=True，原42组全部通过=False，source早筛通过=False。

Run `20261031-phase3-star-raw-open-s2026092701-r01`，actualcode `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`。本轮新训练完整信号级开放分支；已知身份保持原F0的P1默认六类head。P0是星地received，P1是同一received经过冻结DURA v1.0，两者都不是clean。实验检验开放路径的整体差异，不预设哪条更优。

## 数据、权限与评估边界

| 角色 | 数量与划分 | 本轮使用 |
| --- | --- | --- |
| source L | 6300；旧六TX各1050，每TX×RX×day70 | 合法真实旧标签，P0信号级训练和已知统计拟合 |
| source OE | 6134；来自21辅助设备 | 只见outside-six成员角色/公共RX/day；原F0 P0 teacher、新分支训练和background统计，不见真实TX/K |
| source V | 27000；旧六TX各4500，每TX×RX×day300 | NN/全部统计冻结后仅校准风险阈值，同时报告该V上的已知风险 |
| source novel/NV | 3150；19-15、19-16、7-1各1050，每TX×RX×day70 | 所有阈值固定后才预测，独立CPU scorer最后读取trueTX |
| 原Us | 56700祖先记录 | 本轮不读取 |
| target U/Q与正式RX24-18 | 本轮0 | 未适应、发现或未来注册 |

四角色物理记录互斥，共42584条既有观测，没有新物理记录。source RX索引1/3/4/6/8分别为1-19、18-2、19-2、2-19、3-19；三天为2021-03-08、03-15、03-23，接收机和日期均已见。旧六TX为14-10、14-7、20-15、20-19、6-15、8-20。OE设备真实身份及类数没有进入算法。

**该结果是源侧开发验证。** 已知风险在用于阈值校准的同一个V上衡量；NV未参与本候选训练/阈值，但已有历史开发曝光。因此既不是独立已知测试，也不是新接收机或未知设备盲测，更不能代替用户要求的跨未见接收机星地评估。

固定G10MIX01信道：600km、25MHz、2.45GHz、residual/post_sync/noEQ、refSNR30dB、reflect；suburban/urban各0.5及原六面积权重，receiver2027/scene2026103001/channel2026104001。输入一次unitRMS，输出不再RMS。新分支只读received.npy，不读restored实值/P1 teacher/clean，不重抽信道、不运行G。旧P1身份和控制来自相同ID顺序的已完成预测。

## 实际优化设计

原F0为Tri-SIG-TIME SSL E50/11100、232993参数，sourcecontract392005与原DURA合法来源保持。先严格重建可微原F0并执行实际checkpoint synthetic2无query smoke，再导出独立学生，然后冻结原teacher；学生synthetic8梯度probe在真实L/OE读取前通过。复用已验证HeadlessSignal，保留真实SequenceReference和四贡献信号数学，仅删除W并旁路被丢弃内部logits。

新分支232033信号参数＋102688零末层MLP（640→128GELU→160），共334721。z=unit(u_theta+g(c_theta))，原四贡献及fused特征均可微；NoExplicitCFO坐标27保持0，BN/Dropout eval、buffers固定。原F0对OE6134的P0重新计算单位化u作为soft-topology teacher，冻结且状态不变；无P1 teacher复用、更新学生充当teacher、OE硬身份或未知K。逐行teacher仅保留私有训练目录，不进入checkpoint/部署/Git。

与已完成P1 signal-open r03匹配：同原F0独立初始化、2220步、L192/OE64、RNG3017/3018，全部2220步索引逐值一致；T=.1的L跨RX且跨day crossset KL＋OE frozen-u软结构KL权1，L/OE候选各.5并按公共域均衡。无P0P1一致性、VICReg、KD、EMA或新augmentation。教师与学生的输入路径一同从P1变为P0，不能单独把差异归因于恢复器。

fresh AdamW lr1e−4/wd1e−4/eps1e−8；Sinc沿原CENTER_UNITS scale25000/multiplier1000/lr.1/eps1e−11/物理衰减与globalclip5。全部2220实际Sinc steps/clipcalls、索引、loss和全局norm有限。首两步Sinc梯度范数0.016991/0.008324，L/OE全部行VJP均非零；零末层使MLP hidden首步0，第二步0.001577，符合初始化。首两步另保存VJP/模块梯度；未声称逐层梯度在所有后续步均被记录，未补造特征漂移。固定终点，无V/NV早停。

## 四列、阈值与真值隔离

P1_known12精确复用macro12 r01 col3模型/τ/raw/margin；P1_ratio12取signal-open r03 col3固定模型/raw，仅新C12校准。P0_known12为唯一主，P0_ratio12为同新NN诊断。各路径known8与ratio11共享相同known统计：D160、known90/background32、25 Lloyd/seed3019、covshrink.1；先冻结所有NN与统计，才读V。

known12对overall、5RX、6scene施加真实旧TX宏C_K≥90%及原F0 P1 closed−open≤2pp。六阈值按原F0预测类路由，使用exactFraction complete-tie common-tail最大可行q。其余30RX×scene全部报告为诊断，不把12通过写成42通过。全部四τ固定后才打开novel received和任一父novel预测；scorer先重建四raw/margin与控制，再唯一读未知trueTX，不重新fit/校准。

| 方法 | A_K% | C_K% | R_U% | H% | 原P1闭集损失pp | known12 | known42 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1_known12 | 89.948148 | 97.522222 | 46.888889 | 61.643672 | 1.033333 | True | False |
| P1_ratio12 | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | True | False |
| P0_known12 | 90.048148 | 98.503704 | 3.936508 | 7.543258 | 0.933333 | True | False |
| P0_ratio12 | 89.940741 | 98.185185 | 4.761905 | 9.044927 | 1.040741 | True | False |

| 方法 | AUROC% | OSCR% |
| --- | --- | --- |
| P1_known12 | 94.493082 | 87.505278 |
| P1_ratio12 | 94.985255 | 87.944738 |
| P0_known12 | 78.865888 | 74.733706 |
| P0_ratio12 | 78.480956 | 74.519424 |

所有列身份预测及原P1闭集基线完全相同，原闭集90.981481%。A_K为含误拒计错的旧六TX宏正确接收，C_K为宏接收，R_U为未知三TX宏拒识，H=2A_KR_U/(A_K+R_U)。

| 方法 | q | q% | 六预测类拒绝前缀 | 下一q失败硬组 |
| --- | --- | --- | --- | --- |
| P1_known12 | 56/2253 | 2.485575 | [105, 105, 111, 123, 112, 113] | rx/3 |
| P1_ratio12 | 72/2479 | 2.904397 | [123, 123, 129, 144, 130, 132] | rx/3 |
| P0_known12 | 64/4241 | 1.509078 | [64, 64, 67, 74, 67, 68] | scene/practical_low_urban |
| P0_ratio12 | 41/2253 | 1.819796 | [77, 77, 81, 90, 82, 83] | scene/practical_low_urban |

## 配对与完整分组

| 主−参考 | 已知正确gain | 已知正确loss | 已知正确net | 已知接受net | 未知拒对gain | 未知拒对loss | 未知拒对net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1_known12 | 255 | 228 | 27 | 265 | 71 | 1424 | -1353 |
| P0_ratio12 | 84 | 55 | 29 | 86 | 17 | 43 | -26 |

主计数：{"known_rejected": 404, "rejected_original_correct": 252, "rejected_original_wrong": 152, "accepted_wrong": 2283, "original_closed_correct": 24565, "known_correct_accepted": 24313, "unknown_rejected": 124}。两条路径的开放分支独立训练、所有阈值均已冻结，配对差异不用于本轮再调参。

| 旧TX | n | 原闭集% | A_K% | C_K% | 失去原正确数 |
| --- | --- | --- | --- | --- | --- |
| 14-10 | 4500 | 89.044444 | 88.200000 | 98.355556 | 38 |
| 14-7 | 4500 | 80.333333 | 79.666667 | 98.288889 | 30 |
| 20-15 | 4500 | 95.911111 | 94.755556 | 98.644444 | 52 |
| 20-19 | 4500 | 85.466667 | 84.777778 | 98.733333 | 31 |
| 6-15 | 4500 | 96.333333 | 95.177778 | 98.200000 | 52 |
| 8-20 | 4500 | 98.800000 | 97.711111 | 98.800000 | 49 |

| 未知TX | n | 拒对 | R_U% |
| --- | --- | --- | --- |
| 19-15 | 1050 | 28 | 2.666667 |
| 19-16 | 1050 | 28 | 2.666667 |
| 7-1 | 1050 | 68 | 6.476190 |

| 组 | known n | novel n | A_K% | C_K% | R_U% | H% | 原闭集损失pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 27000 | 3150 | 90.048148 | 98.503704 | 3.936508 | 7.543258 | 0.933333 |
| rx/1 | 5400 | 630 | 92.129630 | 98.666667 | 4.920635 | 9.342299 | 0.962963 |
| rx/3 | 5400 | 630 | 84.370370 | 97.851852 | 4.603175 | 8.730045 | 0.777778 |
| rx/4 | 5400 | 630 | 90.574074 | 98.629630 | 2.857143 | 5.539542 | 1.055556 |
| rx/6 | 5400 | 630 | 91.851852 | 98.666667 | 1.587302 | 3.120674 | 0.981481 |
| rx/8 | 5400 | 630 | 91.314815 | 98.703704 | 5.714286 | 10.755514 | 0.888889 |
| scene/practical_high | 1103 | 133 | 97.912879 | 100.000000 | 2.773792 | 5.394756 | 0.000000 |
| scene/practical_mid | 1983 | 256 | 97.300618 | 99.641379 | 2.315156 | 4.522699 | 0.142045 |
| scene/practical_low_suburban | 10304 | 1203 | 95.313304 | 99.405409 | 3.401858 | 6.569251 | 0.291614 |
| scene/practical_high_urban | 1171 | 130 | 96.691217 | 99.138942 | 0.724638 | 1.438495 | 0.261185 |
| scene/practical_mid_urban | 1979 | 236 | 91.356788 | 99.333282 | 3.344017 | 6.451870 | 0.357043 |
| scene/practical_low_urban | 10460 | 1192 | 81.628994 | 97.012704 | 5.504180 | 10.312966 | 1.993565 |
| rxscene/1/practical_high | 202 | 20 | 98.569157 | 100.000000 | 4.761905 | 9.084915 | 0.000000 |
| rxscene/1/practical_mid | 437 | 46 | 98.569227 | 100.000000 | 3.030303 | 5.879843 | 0.000000 |
| rxscene/1/practical_low_suburban | 2042 | 245 | 97.016991 | 99.707869 | 3.529412 | 6.811042 | 0.242081 |
| rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 100.000000 | 3.333333 | 6.450127 | 0.000000 |
| rxscene/1/practical_mid_urban | 401 | 44 | 93.397791 | 99.451872 | 4.524887 | 8.631595 | 0.548128 |
| rxscene/1/practical_low_urban | 2082 | 253 | 84.391250 | 96.928276 | 6.300228 | 11.725118 | 2.172988 |
| rxscene/3/practical_high | 208 | 25 | 93.802944 | 100.000000 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/3/practical_mid | 407 | 54 | 93.338106 | 98.128631 | 3.837719 | 7.372316 | 0.833333 |
| rxscene/3/practical_low_suburban | 2007 | 239 | 91.556256 | 98.667791 | 3.971398 | 7.612588 | 0.339553 |
| rxscene/3/practical_high_urban | 241 | 29 | 91.829085 | 97.242148 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/3/practical_mid_urban | 388 | 44 | 87.843317 | 99.066577 | 0.000000 | 0.000000 | 0.292398 |
| rxscene/3/practical_low_urban | 2149 | 239 | 73.437104 | 96.646586 | 7.611552 | 13.793451 | 1.442202 |
| rxscene/4/practical_high | 213 | 26 | 97.848482 | 100.000000 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/4/practical_mid | 403 | 48 | 98.149588 | 100.000000 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/4/practical_low_suburban | 2049 | 248 | 95.864232 | 99.660587 | 2.517227 | 4.905640 | 0.190341 |
| rxscene/4/practical_high_urban | 238 | 30 | 97.883914 | 99.603175 | 0.000000 | 0.000000 | 0.396825 |
| rxscene/4/practical_mid_urban | 392 | 40 | 92.300852 | 99.794239 | 3.703704 | 7.121641 | 0.205761 |
| rxscene/4/practical_low_urban | 2105 | 238 | 82.134306 | 96.923247 | 4.168812 | 7.934880 | 2.411593 |
| rxscene/6/practical_high | 240 | 24 | 100.000000 | 100.000000 | 0.000000 | 0.000000 | 0.000000 |
| rxscene/6/practical_mid | 365 | 47 | 98.543783 | 100.000000 | 2.380952 | 4.649565 | 0.000000 |
| rxscene/6/practical_low_suburban | 2104 | 240 | 96.508712 | 99.720679 | 1.821465 | 3.575449 | 0.236804 |
| rxscene/6/practical_high_urban | 233 | 27 | 98.341523 | 99.170762 | 0.000000 | 0.000000 | 0.450450 |
| rxscene/6/practical_mid_urban | 408 | 57 | 92.441828 | 98.994311 | 0.000000 | 0.000000 | 0.518838 |
| rxscene/6/practical_low_urban | 2050 | 235 | 83.898712 | 97.049299 | 1.882176 | 3.681755 | 2.193726 |
| rxscene/8/practical_high | 240 | 38 | 98.413143 | 100.000000 | 6.267806 | 11.785039 | 0.000000 |
| rxscene/8/practical_mid | 371 | 61 | 97.533251 | 100.000000 | 1.587302 | 3.123766 | 0.000000 |
| rxscene/8/practical_low_suburban | 2102 | 231 | 95.529064 | 99.270199 | 4.942894 | 9.399439 | 0.439971 |
| rxscene/8/practical_high_urban | 223 | 22 | 95.911730 | 99.612403 | 0.000000 | 0.000000 | 0.387597 |
| rxscene/8/practical_mid_urban | 390 | 51 | 90.892673 | 99.303832 | 7.843137 | 14.440226 | 0.268817 |
| rxscene/8/practical_low_urban | 2074 | 227 | 84.611238 | 97.540164 | 7.381381 | 13.578215 | 1.778513 |
| day/1 | 9000 | 1050 | 90.033333 | 98.455556 | 4.285714 | 8.181956 | 1.000000 |
| day/2 | 9000 | 1050 | 89.855556 | 98.455556 | 4.380952 | 8.354573 | 1.044444 |
| day/3 | 9000 | 1050 | 90.255556 | 98.600000 | 3.142857 | 6.074200 | 0.755556 |

主30交叉组超限／缺失明细：{"rxscene/1/practical_low_urban": {"known_count": 2082, "coverage": {"known_complete": true, "unknown_complete": true, "complete": true}, "C_K": 0.9692827564030649, "A_K": 0.8439124954247269, "original_closed_minus_new_open": 0.021729883327704203, "reasons": ["original_F0_loss_above_2pp"]}, "rxscene/4/practical_low_urban": {"known_count": 2105, "coverage": {"known_complete": true, "unknown_complete": true, "complete": true}, "C_K": 0.9692324694050445, "A_K": 0.8213430564885952, "original_closed_minus_new_open": 0.024115926746147043, "reasons": ["original_F0_loss_above_2pp"]}, "rxscene/6/practical_low_urban": {"known_count": 2050, "coverage": {"known_complete": true, "unknown_complete": true, "complete": true}, "C_K": 0.9704929867484561, "A_K": 0.8389871234517837, "original_closed_minus_new_open": 0.0219372575614748, "reasons": ["original_F0_loss_above_2pp"]}}。各类Wilson区间在独立scorer完整结果，不能当成跨独立RX的宏置信证据。

## 成本与执行证据

| 新增计算 | 真实处理行数 | 合成处理行数 |
| --- | --- | --- |
| 原F0/P0 OEteacher | 6134 | 2 |
| 新信号分支训练 | 568320 | 8 |
| 新信号分支评估 | 42584 | 2 |
| DURA | 0 | 0 |

真实新网络处理共617038行＝teacher6134＋signal训练568320＋signal评估42584，是重复曝光/不同网络处理数，不是新增物理样本或F0同成本推理次数。原识别器native新增0/E50，复制信号特征祖先11100＋新增2220＝13320，新MLP2220。F0＋单个新开放分支567714参数，加既有G算术834665；不据参数量推定时延。祖先P1分支、历史失败及恢复器成本保持，不计为本轮新增。

聚合统计存储：{"P0_known12": {"fields": 8, "bytes": 322868}, "P0_ratio12": {"fields": 11, "bytes": 364340}, "P1_known12": {"fields": 8, "bytes": 322868}, "P1_ratio12": {"fields": 11, "bytes": 364340}}；四套六FP64阈值192B，source逐行银行部署0。Pipeline 461.656523s；阶段{"train": 403.4138779640198, "eval": 29.72833514213562, "score": 28.340956687927246}，不是B1或部署吞吐测量。

Actualcode `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`，单552文件/data0 archive SHA `8cdaa414e8ba4e1a383a67c8181f1f51446fe10c44df01174c1c7e6c7829c9d4` 比较、remotecompile/runtime PASS；19项focused测试/16子测试与一次直接P0/P1审查PASS。完整2220 trace与全部四stdout读取；stdout统计[{"name": "driver.log", "lines": 1, "warnings": 0, "errors": 0, "nonfinite": 0}, {"name": "eval.log", "lines": 3, "warnings": 0, "errors": 0, "nonfinite": 0}, {"name": "score.log", "lines": 1, "warnings": 0, "errors": 0, "nonfinite": 0}, {"name": "train.log", "lines": 26, "warnings": 0, "errors": 0, "nonfinite": 0}]。本地测试既有np.trapz弃用及pytest缓存warning单列在verification，不隐去。

四raw由聚合状态及两路径unit160重建：{"known": {"max_abs_error_by_method": {"P1_known12": 2.2737367544323206e-13, "P1_ratio12": 3.410605131648481e-13, "P0_known12": 1.7053025658242404e-13, "P0_ratio12": 1.7053025658242404e-13}, "P1_known_raw_margin_exact": true, "P1_ratio_raw_exact": true, "parent_fields_exact": true}, "novel": {"max_abs_error_by_method": {"P1_known12": 3.410605131648481e-13, "P1_ratio12": 3.979039320256561e-13, "P0_known12": 1.7053025658242404e-13, "P0_ratio12": 2.2737367544323206e-13}, "P1_known_raw_margin_exact": true, "P1_ratio_raw_exact": true, "parent_fields_exact": true}}；原P1 known12 raw/margin、P1 ratio raw和其他publicmetadata/defaultidentity均精确一致，没有重跑NN、fit、τ扫描或scorer。

独立观察记录实际PID/CWD/argv/CUDA/GPU及日志增长，终态本run live/GPU均空；GPU为离散观察，未测峰值。其它owner停止/修改/重启0。训练/预测/评分所有stage COMPLETE，不覆盖、不重启原root。

## 结论与下一步

本轮不支持直接把开放分支迁到P0。主P0known12较P1known12少拒对1353个未知、只净增27个正确已知接收；宏R_U从46.888889%降到3.936508%，A_K只从89.948148%到90.048148%。P0ratio12同样只有4.761905%拒识；这不是换一种密度读出就已解决的问题。

P1ratio12诊断列A_K89.770370%、C_K97.107407%、R_U60.888889%、H72.561330%，较同P1known12的R_U高14个百分点、A_K低0.177778个百分点。它仍未达90%，也不是本轮唯一主，不能替代主晋级或追溯改主。该结果支持后续在P1上改进开放分离和拒识读出，而不能证明某个未实现的新loss会成功。

下一轮保持原P1身份保护，优先检验P1开放表征的训练目标与源OE使用方式；先核对已有条件支持、局部支持和表征消融，避免重复已失败路线。新目标、对照、阈值规则和预算须另行预登记。P0可以作为待研究的辅助证据，当前没有使用其替换主开放输入的证据；不追加这次训练预算、调本轮阈值或从NV真值构造身份关系。

本候选的预登记判断为 `NO_PROMOTION`，不以控制列替换唯一主。它仅决定source早筛；总体和预登记主场景的跨未见RX星地C_K90/R_U90/A_K较同输入原闭集损失≤2pp仍须独立验证。完整无标签域适应、未知类发现、未来匿名注册继续ACTIVE，A_N/H_OW本轮N/A。不得从这一次输入路径差异推断所有恢复器会去除设备指纹，也不按NV真值更改本轮阈值、loss或主方法。
