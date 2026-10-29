# Phase3 G10 common 初始化风险预算结果

唯一主候选 `joint::common_initialized_budget` **NO_PROMOTION**。宏 A_K=90.9222%、R_U=28.8254%、H-score=43.7732%，C_K=98.2630%，相对原 F0 同 P1 闭集损失0.0593个百分点。源端已知42组和结构保护通过，但总体、五源RX、六场景的未知拒识均未达到90%。本轮支持 common 初始化带来局部收益，不支持宣称跨新RX或匿名注册成功。

## 实验身份与固定范围

实际 run：`20261030-phase3-star-common-seeded-risk-s2026092701-r01`；code `119210ba86fb8b569f6cdeb887d314c1458a2007`，预登记 `60c217bc6b55dd6bfe73e11c5a3c731757f7c365`；单一归档 SHA256 `ed449b8d933b66a6d503d19b9768c29143849eafa0ddab308df12c571f2c2563`，326文件/data0，发布 compile/runtime/SHA 已 VERIFIED。

NN父为联合OVA r02/code `5aa041122396d48c49b8cb3d5912649bdc3217e1`，OE公共分数与零初始化阈值父为OE风险run/code `585e10b07be1f6821edea08b4853582b92512aff`。前者提供冻结F/aux，后者不是权重来源。两臂固定 `detach_control/joint`，三策略固定 `common_tail/zero_initialized_budget/common_initialized_budget`；下文简写 common/zero/seeded。只能由 joint seeded 晋级，不能换成较高控制。

common逐值复用r02 ova_max六阈值，zero逐值复用前轮预算六阈值。seeded从common初态出发，仅允许六个模型预测类阈值单调提高；默认身份、raw max(aux)和所有NN参数均不变。新接收集合因此是common的子集：已知正确接收不会凭空增加，未知拒识不会因放松其他类阈值而下降；此性质不保证90%拒识或新RX已知保护。

合法校准输入为源V27000旧六标签/原F0同P1闭集参照，以及OE6134公共score、outside-six成员关系与15个RX/day域。OE真TX/K、novel和target不参与拟合；V不拟合NN或normalizer。算法从真实common初始mask累计known风险及OE已拒prefix，继续按域宏收益/剩余预算代价选择完整分数层；42组C_K≥90%、原F0 closed−A_K≤2pp保持，最后exact-Fraction核对。它是固定贪心路径，不是全局最优或conformal保证。

本轮只加载实际checkpoint做每臂2条合成G10 P0烟测，总F4；真实IQ输入/真实F前向0，G/NN/aux/optimizer更新0。V、OE、novel均复用公共预测，不新读OE IQ或重建信道。三套阈值、calibration和known输出全部写定后才打开novel公共数组，完整预测后独立CPU truth-last。合成P0不计物理样本，也不是新P1/G管线验证。

祖先F0为Phase1 L6300/U56700/V27000、E50/11100（训练标签比例10%），父两臂各新增2220至累计13320；本轮继承而不重训。42584原物理记录为L6300/OE6134/V27000/novel3150，同一G10/P1缓存重复使用不增加独立样本。V校准标签和OE成员监督是额外合法源监督，不能称全系统零源标签。

source RX index1/3/4/6/8对应`1-19/18-2/19-2/2-19/3-19`，源日2021-03-08/15/23，均已见；novel `19-15/19-16/7-1`各1050已有历史源开发曝光，与known同属这五RX。G10 MIX01、2.45GHz、receiver2027、scene/channel seeds2026103001/2026104001及固定DURA v1.0保持原缓存身份；P1仍是星地恢复输入，不是clean。target U/Q及正式RX24-18访问0。

## 六个运行点与精确复现

A_K是按真实旧TX宏平均的正确接收率，C_K只是接收率；R_U按真实未知TX宏平均，H=2A_KR_U/(A_K+R_U)。原F0同P1闭集参照90.981481%，候选自身closed保持control91.548148%/joint91.540741%，不能用候选closed替换原参照。

| arm / policy | A_K % | C_K % | R_U % | H % | 原F0损失 pp | 宏AUROC % | 宏OSCR % |
|---|---:|---:|---:|---:|---:|---:|---:|
| detach_control / common | 91.2926 | 99.1444 | 11.6825 | 20.7143 | -0.3111 | 84.9276 | 79.4583 |
| detach_control / zero | 90.3815 | 97.1926 | 20.2540 | 33.0922 | 0.6000 | 76.0478 | 72.2844 |
| detach_control / seeded | 90.5630 | 97.6222 | 32.3175 | 47.6360 | 0.4185 | 84.2413 | 78.9166 |
| joint / common | 91.0037 | 98.4630 | 24.9206 | 39.1267 | -0.0222 | 88.9786 | 83.3740 |
| joint / zero | 90.9000 | 98.2741 | 20.2857 | 33.1692 | 0.0815 | 77.3276 | 73.0664 |
| joint / seeded | 90.9222 | 98.2630 | 28.8254 | 43.7732 | 0.0593 | 89.0715 | 83.6316 |

六运行点全部known42/42通过、unknown12/12失败，42组类别覆盖完整，无缺类N/A。seeded的最小组C_K/最大原F0损失分别为control94.8664%/1.9876pp、joint94.6446%/1.9701pp；整体高接收不能替代局部分组保护。

逐ID核实两臂common/zero阈值分别精确等于两父冻结值，其known/novel margin逐值精确复现；所有非scores公共字段与r02完全相同，包括default/aux logits、身份、baseline、IDs和两套单位特征。OE八个字段也与OE父逐值相同。新三margin重建误差0，newτ≥common逐元素成立，known和novel的new接受集合均为common子集；未重新运行模型、评分或校准父结果。

主候选接收26531/27000，正确24549、错身份1982；3150个novel拒对908、误接2242。**98.2630%接收率不等于98.2630%身份准确率**。AUROC/OSCR用实际类条件margin；阈值平移改变排序，不能把父raw-score面积直接沿用。

## 从common起步的收益与代价

| arm | OE域宏 common / zero / seeded % | seeded−common pp | novel R_U common / zero / seeded % |
|---|---:|---:|---:|
| detach_control | 36.6480 / 40.1161 / 47.2755 | +10.627541 | 11.6825 / 20.2540 / 32.3175 |
| joint | 56.6686 / 53.1226 / 57.2531 | +0.584554 | 24.9206 / 20.2857 / 28.8254 |

主joint从common起步，OE域宏56.6686%→57.2531%，提升0.584554pp；novel24.9206%→28.8254%，提升3.904762pp。这次OE代理与novel同向，且超过zero的53.1226%/20.2857%。这是预定初始化修改的局部结果，不能将OE域宏（每域1/15权重）等同于未知真实TX宏拒识。

| seeded − 对照 | 已知正确 gain/loss（净） | 接收 gain/loss（净） | 未知拒对 gain/loss（净） | ΔA_K pp | ΔR_U pp | ΔH pp |
|---|---:|---:|---:|---:|---:|---:|
| detach_control - common | 0/197 (-197) | 0/411 (-411) | 650/0 (+650) | -0.729630 | +20.634921 | +26.921676 |
| detach_control - zero | 108/59 (+49) | 261/145 (+116) | 474/94 (+380) | +0.181481 | +12.063492 | +14.543813 |
| joint - common | 0/22 (-22) | 0/54 (-54) | 123/0 (+123) | -0.081481 | +3.904762 | +4.646486 |
| joint - zero | 30/24 (+6) | 64/67 (-3) | 316/47 (+269) | +0.022222 | +8.539683 | +10.604000 |

joint相对common新增123条未知拒对、没有丢失任何原拒对，同时失去22条已知正确接收和54条接收；H增加4.646486pp。相对zero的未知净增269、已知正确净增6，但seeded/zero之间没有集合包含保证。控制seeded R_U32.3175%高于主28.8254%，仍不能换主或宣布晋级。

## 完整选择轨迹与停止原因

| arm | 新选择步数 | 各预测类新增步数 | OE拒绝 common→seeded | V拒绝 common→seeded |
|---|---:|---|---:|---:|
| detach_control | 810 | [227,29,368,72,0,114] | 2654→3464 | 231→642 |
| joint | 31 | [0,2,22,2,5,0] | 3953→3984 | 415→469 |

完整解析新841条选择记录，step连续、全部数值有限，新增收益之和分别0.106275409589/0.005845537528，等于objective_after−before；新增OE拒绝810/31、V拒绝411/54与公开数组一致。另完整解析复用zero报告的6550条旧轨迹；它们不是本轮新增优化或计算预算。

主joint初τ为`[-1.00492048,-1.11341715,-1.53453076,-1.30284131,-2.73408842,-0.62084115]`，终τ为`[-1.00492048,-1.03836465,-1.13450587,-0.99945688,-2.72993422,-0.62084115]`，类0/5保持初值。两臂初态合法、无科学回退，终点maximal=true/global_optimum_claimed=false；六类下一候选均因局部known风险非法，不是OE层耗尽。

| 主预测类index | 下一候选绑定RX/scene | 原F0损失 pp | FRR % |
|---|---|---:|---:|
| 0 | RX3/high, RX3/mid, RX3/midU, RX8/lowU | 2.3136, 3.0812, 2.6969, 3.1458 | 4.5159, 6.4666, 5.0133, 5.8368 |
| 1 | RX4/highU | 2.3686 | 2.2006 |
| 2 | RX3/midU | 2.0761 | 4.3925 |
| 3 | RX4/highU | 2.2423 | 2.0743 |
| 4 | RX3/high | 2.3325 | 4.5348 |
| 5 | RX3/mid | 2.2265 | 5.6119 |

全部绑定都由2pp总身份损失触发，FRR仍低于10%；没有为追求OE收益放松保护。control的下一候选涉及RX3 high/mid/mid urban、RX1/6 low urban、RX4 high urban，完整多组明细在校准文件。没有扫描novel阈值或追加候选。

## RX、六场景与逐真实TX

下表为主模型三策略的真实未知TX宏R_U；high/mid/lowS/highU/midU/lowU为郊区高/中/低、城区高/中/低。source RX都是已见RX；30个RX×scene未知组合只作诊断，全部42组逐策略指标保留在评分JSON。

| 组 | joint common / zero / seeded R_U % | 主A_K / C_K % | 主原F0损失 pp |
|---|---:|---:|---:|
| overall | 24.9206 / 20.2857 / 28.8254 | 90.9222 / 98.2630 | 0.0593 |
| RX1 | 21.5873 / 11.2698 / 28.7302 | 93.2222 / 99.3704 | -0.1296 |
| RX3 | 25.0794 / 26.0317 / 26.3492 | 84.9815 / 96.0000 | 0.1667 |
| RX4 | 26.5079 / 19.2063 / 31.5873 | 91.4074 / 98.4259 | 0.2222 |
| RX6 | 29.6825 / 20.7937 / 34.1270 | 93.1111 / 99.3889 | -0.2778 |
| RX8 | 21.7460 / 24.1270 / 23.3333 | 91.8889 / 98.1296 | 0.3148 |
| high | 42.0373 / 28.5350 / 46.1023 | 97.2379 / 98.0483 | 0.6750 |
| mid | 33.4590 / 27.2980 / 35.0587 | 97.0276 / 98.4882 | 0.4151 |
| lowS | 29.5858 / 22.8601 / 33.7916 | 95.5083 / 98.5757 | 0.0966 |
| highU | 28.9483 / 23.0955 / 36.6171 | 96.5412 / 98.4060 | 0.4112 |
| midU | 22.3291 / 19.3269 / 26.1538 | 91.8643 / 98.5411 | -0.1505 |
| lowU | 16.4911 / 15.2128 / 20.1103 | 83.7350 / 97.8638 | -0.1124 |

| 已知TX（各4500） | control seeded正确 | joint common / zero / seeded正确 | 主C_K % |
|---|---:|---:|---:|
| 14-10 | 4020 | 4012 / 4023 / 4012 | 98.4667 |
| 14-7 | 3511 | 3573 / 3567 / 3567 | 98.1556 |
| 20-15 | 4278 | 4323 / 4332 / 4319 | 98.5556 |
| 20-19 | 3905 | 3918 / 3881 / 3906 | 97.6667 |
| 6-15 | 4319 | 4324 / 4324 / 4324 | 97.8000 |
| 8-20 | 4419 | 4421 / 4416 / 4421 | 98.9333 |

| 未知TX（各1050） | control seeded拒对 | joint common / zero / seeded拒对 | 主R_U % [95% Wilson] |
|---|---:|---:|---|
| 19-15 | 479 | 273 / 229 / 341 | 32.4762 [29.7120, 35.3681] |
| 19-16 | 192 | 174 / 75 / 203 | 19.3333 [17.0582, 21.8320] |
| 7-1 | 347 | 338 / 335 / 364 | 34.6667 [31.8487, 37.5964] |

Wilson区间按真实类别描述这组源记录，不代表新RX/seed/episode总体区间；场景内类别数不等，宏平均不能以pooled替代。结构单位特征完全不变，跨source-RX1NN仍为joint79.968254%、control79.206349%、原F078.793651%，guard通过；79.97%不是未来匿名A_N。

## 成本、实际资源与最终边界

主部署NN仍为F232993＋aux966＋G266951=500910参数，本轮更新0。每策略只部署6个FP64阈值（48B），每臂保存三策略18τ（144B），不是新增NN参数；OE行bank、风险轨迹和结构诊断特征不部署。父模型训练、OE前轮12272F、G缓存等历史成本保留；本轮增量仅4个合成烟测F行前向，真实新IQ/F/G前向0。

| 阶段 | 实际PID / CUDA_VISIBLE_DEVICES | 进程墙钟 s | source内部 s |
|---|---|---:|---:|
| control source | 3106706 / 4 | 4.440793 | 3.722287 |
| joint source | 3106707 / 2 | 4.119844 | 3.432634 |
| control / joint CPU score | 3107130 / 3107129 | 2.852142 / 2.793525 | N/A |

driver3106658，总墙钟10.645742s，两臂并行；不是逐包推理时延。五次独立observations的前四次确认两个source live PID、正确CWD/argv与CUDA_VISIBLE_DEVICES4/2，但owned nvidia列表始终空，第五次已terminal；**没有捕获GPU显存占用或峰值**，不能由分配卡号伪造遥测。source日志前四次control562B/joint553B不变，最终679B/652B，只有起末大小证据，没有完整运行期增长曲线；未为补监控重跑。

独立终态source/CPUscore全部COMPLETE、run ANALYZED_SOURCE_ONLY、live/owned GPU均空、foreign stop0。25项测试42子测试PASS7.09s，唯一P0/P1 PASS；6项/10子项已包含其中，不相加。完整读取22份JSON、841条新与6550条复用选择记录、5次observations、7份完整阶段日志和8份当前NPZ；阶段日志无异常/OOM/非有限信号。报告分析没有打开真实IQ、私有truth文件或checkpoint，也没有新增fit/预测/评分。

可复核：[control完整评分](/E:/type10-7/local_artifacts/pamf_phase3_star_common_seeded_risk_20261030_r01/rows/detach_control/score/results.json)、[joint完整评分](/E:/type10-7/local_artifacts/pamf_phase3_star_common_seeded_risk_20261030_r01/rows/joint/score/results.json)、[主校准与全部轨迹](/E:/type10-7/local_artifacts/pamf_phase3_star_common_seeded_risk_20261030_r01/rows/joint/predict/calibration/complete.json)、[pipeline](/E:/type10-7/local_artifacts/pamf_phase3_star_common_seeded_risk_20261030_r01/pipeline_state.json)、[独立观察](/E:/type10-7/local_artifacts/pamf_phase3_star_common_seeded_risk_20261030_r01/observations.jsonl)。

本轮取得小幅同源增益，仍未达到未知拒识要求，不证明已解决阈值、表征或跨RX泛化问题。A_N/H_OW=N/A、target/formal访问0，完整Phase3目标保持ACTIVE；没有换主、扩seed或启动新候选。
