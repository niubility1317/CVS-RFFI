# PAMF 类别联合偏移无标签适应结果（2026-10-31）

**NO_PROMOTION，完整目标 ACTIVE。** 唯一主 `CLASS_ratio` 虽增加跨 RX 旧类正确接收，却降低星地未知拒识；clean/star 的 C_K 与同 DURA 原 F0 2pp保护也失败。未按较高控制换主。

Run `20261031-phase3-class-shift-s2026092701-r01`，actualcode `74e0a3de37d0f3673375804fb09b06c4f624745b`。三条件六列预测/独立评分 COMPLETE，pipeline `ANALYZED_DEVELOPMENT`；本报告不重新评分、EM、调τ或读取原真值/IQ/权重。

## 固定机制与数据边界

原 F0 E50/11100/default6（232993参数）、P1 signal r03/code dc01a402582f93e452dbaa3ef59a9fbb44151818（232033＋MLP102688）、DURA v1.0（266951、E20/4940）冻结。signal继承特征11100＋2220/MLP2220；源合同392005的L6300/OE6134/U_s56700仅为祖先，新消费0。L为原训练池10%，V27000是额外已知校准监督。

DA0精确复用dura-best/code9fc23a4cb78111aa96cc72fde0d35c036a876e5d；SHARED原样复用mixture-UDA/code d72a7b7c301fec2ff3736e4221dfe5047ab78efc 的DA_M/prior/20步诊断。source known90/OE32、normalizer及原两套六τ均不更新，未使用class-density的新τ。新CLASS从六类×15域逐类中心化残差构建联合先验，由单一系数生成六delta，背景取其均值；所有模式一起参与分数，原F0预测类只路由旧τ，不选择delta。

唯一新拟合输入为原star-U384缓存z160（RX24-1/index26、2021-03-08），固定20次MAP-EM；无真实TX/K/配额/default-logits拟合或clean适应。新源先验半径5.263391与SHARED的2.370508不同，不能解释为类别结构变化而半径完全匹配的单因素消融。source/prior先冻结、三真实checkpoint synthetic smoke后读U特征；CLASS冻结后才打开源/Q旧预测。三条件全固定后独立CPU连接源NV/cleanQ/starQ truth各一次，Utruth0。

源V27000/novel3150为旧六类各4500、19-15/19-16/7-1各1050；源RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，日期03-08/15/23已见。clean/star Q均为RX24-1、03-15的同864物理记录（旧六/未知三类各96），与U互斥且跨日，日期不是源域未见。三条件均经同DURA；source RX不绑定目标adapter，六列重复原两列。源NV/Q历史曝光、单RX/seed及重复物理记录不构成独立盲测；正式RX24-18未访问。

## 全部18个运行点

A_K为旧类正确且接收的逐TX宏平均，C_K为接收率，R_U为未知拒识率，H=2A_KR_U/(A_K＋R_U)。指标为%，损伤为同DURA输入原F0闭集减A_K的百分点；AUROC/OSCR只描述固定分数，没有扫描新运行点。

| 方法 | 条件 | A_K | C_K | R_U | H | 原F0损伤pp | AUROC | OSCR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DA0_known | source_DURA | 89.948148 | 97.522222 | 46.888889 | 61.643672 | 1.033333 | 94.493082 | 87.505278 |
| DA0_known | clean_DURA | 63.541667 | 80.729167 | 88.541667 | 73.986872 | 16.493056 | 90.510465 | 72.142048 |
| DA0_known | star_DURA | 62.847222 | 86.631944 | 45.833333 | 53.008520 | 10.069444 | 77.519170 | 56.334394 |
| DA0_ratio | source_DURA | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 | 87.944738 |
| DA0_ratio | clean_DURA | 68.923611 | 85.416667 | 99.652778 | 81.487441 | 11.111111 | 95.991874 | 77.463710 |
| DA0_ratio | star_DURA | 63.020833 | 84.895833 | 76.388889 | 69.063927 | 9.895833 | 84.728250 | 62.604890 |
| SHARED_known | source_DURA | 89.948148 | 97.522222 | 46.888889 | 61.643672 | 1.033333 | 94.493082 | 87.505278 |
| SHARED_known | clean_DURA | 63.368056 | 80.555556 | 88.541667 | 73.869048 | 16.666667 | 90.609327 | 72.249349 |
| SHARED_known | star_DURA | 62.673611 | 86.458333 | 45.833333 | 52.946667 | 10.243056 | 77.539665 | 56.315104 |
| SHARED_ratio | source_DURA | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 | 87.944738 |
| SHARED_ratio | clean_DURA | 68.576389 | 85.069444 | 99.652778 | 81.244267 | 11.458333 | 95.924961 | 77.416088 |
| SHARED_ratio | star_DURA | 63.020833 | 84.895833 | 75.694444 | 68.778682 | 9.895833 | 84.689067 | 62.573544 |
| CLASS_known | source_DURA | 89.948148 | 97.522222 | 46.888889 | 61.643672 | 1.033333 | 94.493082 | 87.505278 |
| CLASS_known | clean_DURA | 64.756944 | 81.770833 | 87.500000 | 74.429875 | 15.277778 | 90.346499 | 72.051625 |
| CLASS_known | star_DURA | 64.062500 | 87.847222 | 40.625000 | 49.720149 | 8.854167 | 77.146629 | 56.196349 |
| CLASS_ratio | source_DURA | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 | 87.944738 |
| CLASS_ratio | clean_DURA | 70.312500 | 86.805556 | 99.652778 | 82.450204 | 9.722222 | 95.707345 | 77.273220 |
| CLASS_ratio | star_DURA | 64.062500 | 86.111111 | 71.180556 | 67.434211 | 8.854167 | 84.616729 | 62.709780 |

原F0闭集source90.981481%、clean80.034722%、star72.916667%，所有状态逐条不变。主clean正确接收405/576、接受500/576、未知拒对287/288；star369/576、496/576、205/288。clean的R_U过95单项，但C_K86.805556%/损伤9.722222pp失败；star R_U71.180556%/C_K86.111111%/损伤8.854167pp均失败。

## 主方法同ID配对

计数为gain/loss/net；原身份完全相同，新增接受不是身份头纠错。源全部配对净差0。

| 参照 | 条件 | 正确接收 | 接受 | 未知拒对 | ΔA_K pp | ΔR_U pp | ΔH pp |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| DA0_ratio | clean_DURA | 8/0/+8 | 8/0/+8 | 0/0/+0 | 1.388889 | 0.000000 | 0.962763 |
| DA0_ratio | star_DURA | 6/0/+6 | 7/0/+7 | 0/15/-15 | 1.041667 | -5.208333 | -1.629716 |
| SHARED_ratio | clean_DURA | 10/0/+10 | 10/0/+10 | 0/0/+0 | 1.736111 | 0.000000 | 1.205938 |
| SHARED_ratio | star_DURA | 6/0/+6 | 7/0/+7 | 0/13/-13 | 1.041667 | -4.513889 | -1.344471 |

相对DA0，星地新增7条接受中6条正确、1条错身份；三未知TX拒对从72/72/76降到69/70/66（各96），多漏放3/2/10条。clean仍95/96/96。主仍误拒clean56条、star51条原F0正确样本；14-7分别接收95/96、96/96却仅28、32条身份正确，C_K不能替代A_K。

## 主星地场景与覆盖

| 场景 | known/unknown | A_K% | C_K% | R_U% | 原F0损伤pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| practical_high | 33/6 | 72.857143 | 93.333333 | N/A | 3.333333 |
| practical_mid | 42/25 | 69.722222 | 87.222222 | 87.962963 | 12.777778 |
| practical_low_suburban | 203/117 | 65.885177 | 85.309780 | 83.352102 | 10.240015 |
| practical_high_urban | 38/16 | 77.711640 | 93.452381 | 86.666667 | 4.166667 |
| practical_mid_urban | 45/15 | 73.657407 | 92.592593 | 75.555556 | 7.407407 |
| practical_low_urban | 215/109 | 54.563337 | 82.625823 | 49.704301 | 8.966202 |

high未知三TX数0/4/2，缺19-15，R_U/H为N/A；6条全拒对不能认证该景。其余五景R_U都低于90%，六景损伤全超过2pp。源known12保持通过、known42未全过，source R_U60.888889%仅诊断；所有TX/RX/day/42组、覆盖、逐TX Wilson95%描述区间及配对细分保留在results/精简analysis，不事后删景、补样或换主。

## 完整拟合轨迹与成本

完整CLASS iteration0至20共21条均有限、严格增加：181695.052814→182029.344876（＋334.292062），最小及末步增量0.000135359。固定20步未早停，不宣称已经完全收敛。known posterior mass0.507564→0.530641是模型责任质量，不是真实已知比例/纯度。

源rank14/r5.263391448463，系数范数＝六类shift RMS2.044845189513<r，最大类范数3.058136208201、背景均值偏移范数0.887298263929，约束乘子全0。约束是RMS而非maxclass；真实非零适应未撞边界。SHARED的旧21条记录/状态精确继承，不重新EM。新CLASS20步目标mode统计适应与继承SHARED20分开；source密度/normalizer/τ及NN更新0，不代表目标密度完全未拟合。

新IQ及真实G/F0/signal前向全0，缓存U特征读384行；三实际checkpoint synthetic smoke各2条且PASS。旧mix导出U的各384真实前向、原F0/G/signal训练仍为祖先成本。活跃神经834665参数不变；source11数组364340B/两套六τ96B，新prior107768B（NPZ108770B）、CLASS9208B（delta＋背景8960B、系数112B、schema136B；NPZ10236B），另存继承SHARED1528B/旧prior18168B，不部署逐source/U bank。

predict7.573466s、CPUscore6.886858s、pipeline14.578216s为批处理耗时，非B1。3次独立观察捕获driver1043230和predictor1043324的CWD/argv/CUDA7、GPU7/UUID9c4eb614一次374MiB采样，日志482→542B，终态live/owned_gpu空、foreign0。没有CPUscorer live /proc样本，只pipeline记录CPU/gpuNone/ret0及代码CUDAempty设置，不能称独立live核过；显存样本不是峰值。

22份JSON、3观测、8份完整stdout/文本已读。运行stdout为21B/1行、542B/3行、41B/1行，无warning/error/traceback/nonfinite；初始RED模块缺失为预期TDD。既有20unique＋14sub/一次P0P1PASS；scorer旧np.trapz告警全scope4668、末尾重叠单case696，不增加unique。core先前console5PASS/缓存warning1无rawlog由core_test_observation注明。单archive587/data0、SHA/compile/runtime PASS。

结论仅限该源联合先验和固定20步适应：少量旧类接收收益伴随更大的未知保护损伤，U目标改善不等于开放识别成功，也未证伪所有UDA。主未过clean/star保护与场景覆盖，NO_PROMOTION；源/开发曝光限制保持，未知发现和匿名注册未执行，A_N/H_OW为N/A、完整目标ACTIVE。

证据：[方案](PAMF_PHASE3_CLASS_SHIFT_PLAN_20261031.md)、[run记录](../automation_reports/CV-SincNet/20261031-phase3-class-shift-s2026092701-r01/report.md)、[独立评分](E:/type10-7/local_artifacts/pamf_phase3_class_shift_20261031_r01/score/results.json)、[精简分析](E:/type10-7/local_artifacts/pamf_phase3_class_shift_20261031_r01/analysis.json)。
