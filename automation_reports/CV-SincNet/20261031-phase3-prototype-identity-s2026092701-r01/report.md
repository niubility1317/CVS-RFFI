# 源原型身份与PCD条件拒识

ANALYZED，执行 VERIFIED；科学 NO_PROMOTION，完整目标 ACTIVE。

# Phase3 源原型身份与 PCD 条件拒识（2026-10-31）

Run `20261031-phase3-prototype-identity-s2026092701-r01`。源端簇几何排障显示 PCD 单位坐标的跨接收机最近源类中心闭集诊断较强，而无标签簇与跨日支持尚未通过筛选。本行检验不依赖目标簇的源原型身份判决，不直接晋级失败的聚类方案。原 PCD 研究基线及所有历史 NO_PROMOTION 保持。

## 固定方法和顺序

严格复用 fresh f4b5 的 TARGET_PCD444、原 F0/default6、DURA v1.0 R69、PCD known90/OE32/origin/whitener 和旧两组六阈值。新神经、GMM、DURA、目标适应更新均0。先实际 checkpoint 合成烟测，通过即继续；本行真实 IQ 前向0，source V/NV/Q 复用已冻结的相同 DURA 特征与原 F0 logits。

从 source-cluster-geometry/074a84c 的 source_features.npz 精确复用最终 PCD 单位坐标 L6300，以同 IDs 对齐原 L 的合法 y/rx/day 元数据。六类中心是各类单位向量的算术均值，不再单位化中心，与源端已核实的最近中心规则一致。其身份 c=argmin_j ||unit(z)−mean_j||；只使用六个已注册源类，不读取目标标签、真实未知类数或类别配额。

六中心先于 V 数值冻结。已知 V27000 用该原型 c 路由 PCD 原两种密度分数：pooled_ratio 与 predicted_ratio；后者取 c 的15个源RX/day模式减 OE32 背景。两种分数均由 c 索引各自六个新阈值。固定复用 star_macro12_risk.calibrate_known，以新 closed、原 F0 baseline_closed 和已知 V 标签校准；硬保护仍是总体＋五源RX＋六场景共known12，30交叉组另报。新原型闭集较强也不能替换原 F0 的2个百分点保护基准。

两个新六τ完全冻结后才读取源 NV 与两个未来 Q。所有4读出×3条件预测完成后，独立 CPU scorer 才连接 source NV、clean Q、star Q 三份真值各一次，U真值0。原 F0 logits 与原型距离证据分开保存；新 closed 不来自原 F0 argmax，不能称原身份判决不变或把距离冒充神经 logits。

## 数据、矩阵与指标

source L6300/OE6134/V27000/NV3150、五源 RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、03-08/15/23和六旧TX均保持。新读 L 特征/合法标签6300，原 Us56700及OE逐行值0；V仍是额外有标签校准角色，不是独立已知类测试。

目标仍为开发 RX3-2/index34、03-15/23未来10238条，旧4508、未注册四类5730；前期 U576 仅为所用 PCD 的继承适应来源，本行不新读U。六旧类为14-10/14-7/20-15/20-19/6-15/8-20，目标非旧为5-5/10-7/2-19/7-11，四类本行均保持未注册。三个测试条件为源已知RX星地、跨RX clean、跨RX星地，均已经过同一冻结DURA；clean不学习，未来同canonical清洁/星地记录不重复算独立样本。

四读出固定 OLD_PCD_pooled_ratio、OLD_PCD_predicted_ratio、PROTO_pooled_ratio、PROTO_predicted_ratio，唯一主为最后一项。旧两列复用 fresh PCD 原预测及阈值，不重拟合；两个新列只改变源原型身份、分数路由及已知V新τ。主报逐TX宏A_K/C_K/R_U/H、原F0同DURA闭集与新原型闭集、原F0闭集−A_K，完整TX/RX/day/六景和配对得失。

source known12保护、目标总体及必需主景的clean R_U≥95%/星地R_U≥90%、C_K≥90%、原F0−A_K≤2个百分点、必需类别完整覆盖仍需同时通过；主星地 A_K须高于 OLD_PCD_predicted_ratio，R_U不得下降。未全过NO_PROMOTION、不扩seed、不降低门槛；03-23缺两个旧TX时完整宏指标N/A。匿名注册未启用，A_N/H_OW=N/A。此为复合判决变化，不是新UDA训练；PCD有U576祖先，不称纯源模型或新增适应成功。

## 执行与成本

EXACT6公开配置为run_id/code_commit/reference_config/source_geometry_root/output_root/batch_size128，不含truth路径。唯一owner root，不可覆盖新root，N607单predict GPU worker再独立CPU scorer；每GPU新增≤2且实际free≥4096MiB，不改变其他用户任务。

真实F0/signal/G新前向均0；checkpoint合成烟测单列。新统计为六中心一组（6×160 FP64=7680 B）与2×6新τ，原GMM/旧τ保持。运行比较可加载F0/r03/PCD/G共1169386参数；主算法最小神经路径为PCD334721＋G266951=601672，F0仍用于本次原始基准/控制，不能把缓存预测时间当新部署B1时延。

角色/祖先不符、冻结状态改变、nonfinite、输出碰撞或无法产生合法预测即保留停止；低科学分数仍完成三条件全部预测和评分。代码commit、聚焦负测、一次直接P0P1、最小登记、单N607preflight/archiveSHA/compile、PID/CWD/argv/GPU/log增长后按实际结果归档。Q已曝光、单seed/开发RX、global packet history UNKNOWN及正式RX24-18未访问的限制保持，完整目标ACTIVE。


实际代码 `89483533937a8b97b51700b984e404d10ecab4c5`；环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-prototype-identity-s2026092701-r01-release`；owner `codex-phase3-prototype-identity-20261031`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.prototype_identity_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-prototype-identity-s2026092701-r01-release/runtime.json`。输入/输出路径见 experiment.json。


# Phase3 源原型身份与 PCD 条件拒识结果（2026-10-31）

**NO_PROMOTION，原 PCD 研究基线保留，完整目标 ACTIVE。** 固定原型身份在目标 clean/星地的宏闭集已较原 F0 下降3.691028/2.337520个百分点；即使不拒绝任何正确已知样本，也无法满足本轮2个百分点保护。这仅约束当前数据与固定分类器，不是所有原型方法的结论。

Run `20261031-phase3-prototype-identity-s2026092701-r01`；actual code `89483533937a8b97b51700b984e404d10ecab4c5`；pipeline ANALYZED_DEVELOPMENT，predict/score均COMPLETE/0。设计见[计划](PAMF_PHASE3_PROTOTYPE_IDENTITY_PLAN_20261031.md)。

## 判决与数据

冻结 fresh/f4b5 TARGET_PCD444、原F0、DURA、PCD known90/OE32/origin/whitener和两旧六τ。以source L6300最终PCD单位特征及合法标签求六类算术均值，中心不再次单位化；最近均值身份c同时用于predicted密度15模式和新τ路由。原F0/defaulthead权重不变，但复合身份判决改变，原型距离不是神经logits。

六中心先于V冻结；V27000仅按known12校准两新六τ，两旧套精确复用。全部τ冻结后才读NV3150及两Q；四方法×三条件全预测后，CPU独立连接source NV和两Q真值各一次。保护基准始终为同DURA原F0，不能换成较低的新原型闭集。

source祖先L6300/OE6134、V27000额外有标签校准、NV3150，六旧TX/五源RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、03-08/15/23保持。目标为RX3-2/index34，未来Q10238：旧4508、四未注册类5730，clean/星地同canonical且均DURA。PCD保留U576祖先，本轮U读取/新UDA为0，clean仅测试。

## 四方法 × 三条件

均为逐TX等权宏百分比；C_K为已知接受率，不是分类准确率。source V与L-only几何及目标Q是不同角色，容量也不等。

| 方法 | 条件 | 原F0闭集 | 当前闭集 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|---:|---:|
| OLD_PCD_pooled_ratio | source_DURA | 90.981481 | 90.981481 | 89.770370 | 97.081481 | 63.619048 | 74.465443 |
| OLD_PCD_pooled_ratio | clean_DURA | 48.567015 | 48.567015 | 32.123647 | 44.791196 | 60.790839 | 42.034854 |
| OLD_PCD_pooled_ratio | star_DURA | 41.721059 | 41.721059 | 29.175335 | 50.230381 | 48.454372 | 36.420916 |
| OLD_PCD_predicted_ratio | source_DURA | 90.981481 | 90.981481 | 89.733333 | 96.748148 | 64.285714 | 74.907247 |
| OLD_PCD_predicted_ratio | clean_DURA | 48.567015 | 48.567015 | 31.661804 | 43.761391 | 61.979647 | 41.912794 |
| OLD_PCD_predicted_ratio | star_DURA | 41.721059 | 41.721059 | 27.951722 | 48.582412 | 49.307698 | 35.678111 |
| PROTO_pooled_ratio | source_DURA | 90.981481 | 91.585185 | 89.970370 | 96.381481 | 62.285714 | 73.611098 |
| PROTO_pooled_ratio | clean_DURA | 48.567015 | 44.875988 | 27.672118 | 40.501360 | 63.319940 | 38.513182 |
| PROTO_pooled_ratio | star_DURA | 41.721059 | 39.383539 | 24.197231 | 43.181528 | 54.180700 | 33.453880 |
| PROTO_predicted_ratio | source_DURA | 90.981481 | 91.585185 | 89.966667 | 96.337037 | 61.555556 | 73.097504 |
| PROTO_predicted_ratio | clean_DURA | 48.567015 | 44.875988 | 27.478806 | 40.621900 | 63.408169 | 38.341705 |
| PROTO_predicted_ratio | star_DURA | 41.721059 | 39.383539 | 24.072020 | 42.895457 | 54.440791 | 33.383082 |

两旧控制六运行点的A_K/C_K/R_U/H/原F0闭集与fresh历史JSON逐值一致；既有producer/scorer确认模型、τ及预测精确复用，本报告不补读NPZ。主星地对OLD_PCD：A_K−3.879703pp、R_U＋5.133092pp、H−2.295029pp；拒识单项改善不能替换主结果。

## 闭集上限与拒识损伤

| 主条件 | 原F0−原型闭集 pp | 原型闭集−A_K pp | 原F0−A_K pp | 原型闭集正确/正确接收 |
|---|---:|---:|---:|---:|
| source_DURA | -0.603704 | 1.618519 | 1.014815 | 24728/24291 |
| clean_DURA | 3.691028 | 17.397181 | 21.088209 | 2141/1630 |
| star_DURA | 2.337520 | 15.311519 | 17.649039 | 2054/1580 |

固定c的正确接收只能是其闭集正确集合的子集，故任意只改变拒识τ、保持c不变的规则都有A_K≤当前闭集。clean44.875988%低于原F048.567015%−2pp，star39.383539%低于41.721059%−2pp；只调τ无法通过本固定规则的绝对保护。本报告未扫描τ或生成新运行点。

此前L-only原型支持留出最近类约97.9%来自训练角色，支持构造也与本轮全L均值不同。当前source V27000原型闭集91.585185%，原F090.981481%；不能把L-only成绩直接当作V或目标泛化证据，V还承担额外校准。

| 主−OLD_PCD_predicted | 闭集正确 gain/loss/net | 正确接收 gain/loss/net | 接受net | 正确但拒绝net | 未知拒识 gain/loss/net |
|---|---|---|---:|---:|---|
| source_DURA | 772/609/163 | 675/612/63 | -111 | 100 | 196/282/-86 |
| clean_DURA | 316/317/-1 | 124/141/-17 | -33 | 16 | 298/219/79 |
| star_DURA | 356/246/110 | 165/149/16 | -110 | 94 | 405/114/291 |

星地样本闭集正确净＋110、正确接收净＋16，但宏闭集与宏A_K都下降：各旧TX容量不等，大类增益不能掩盖小类损伤。星地正确但被拒由380增加到474，已知总拒绝1875→1985；身份错误与拒识损伤同时存在，不能只看总正确条数。

## 逐类、场景、日期与校准

| 星地旧TX | 条数 | 原F0闭集% | 原型闭集% | OLD A_K% | 主A_K% |
|---|---:|---:|---:|---:|---:|
| 14-10 | 1491 | 52.850436 | 63.916834 | 45.070423 | 50.033535 |
| 14-7 | 1182 | 14.805415 | 14.974619 | 11.844332 | 13.113367 |
| 20-15 | 150 | 41.333333 | 26.000000 | 2.666667 | 2.000000 |
| 20-19 | 406 | 5.172414 | 1.970443 | 2.216749 | 0.738916 |
| 6-15 | 1186 | 70.573356 | 69.224283 | 58.600337 | 54.890388 |
| 8-20 | 93 | 65.591398 | 60.215054 | 47.311828 | 23.655914 |

星地四未知5-5/10-7/2-19/7-11各1454/1439/1427/1410条，主拒识分别34.181568/55.802641/60.686755/67.092199%；四类保持未注册，R_U四类等权，目标角色不用于预测路由。

| 星地场景 | 当前闭集% | A_K% | C_K% | R_U% |
|---|---:|---:|---:|---:|
| practical_high | 38.440767 | 23.122406 | 38.118991 | 59.759699 |
| practical_mid | 47.593048 | 24.802887 | 40.007898 | 61.022952 |
| practical_low_suburban | 43.061167 | 26.021737 | 41.834271 | 56.598558 |
| practical_high_urban | 44.690022 | 36.751770 | 59.004861 | 60.405656 |
| practical_mid_urban | 48.524379 | 26.236406 | 39.381261 | 52.550097 |
| practical_low_urban | 32.540827 | 20.965399 | 44.267260 | 50.001003 |

两Q中03-15为5525条、03-23为4713条；后者缺20-19与8-20，完整已知闭集/A_K/C_K/H保留N/A，R_U独立可报clean64.228290%、star54.938490%。总体与六主景覆盖齐全，cleanRU95/starRU90、C_K90、原F0损伤2pp及星地AK严格增益未联合通过。

source known12通过、known42未全过。超2pp交叉诊断为RX3/high、RX3/mid、RX3/mid_urban、RX4/high_urban、RX8/low_urban；30交叉不是本轮硬保护项，但不能声称42全通过。主source AUROC91.687857%/OSCR85.737108%，旧对照94.730673%/87.940433%；曲线不能替代固定运行点。

已存主校准q=55/1498，六预测类prefix为165/169/165/161/163/166、人数4503/4622/4494/4387/4466/4528。下一已存q=82/2233因RX8原F0损伤2.018519pp失败；pooled新校准下一候选也受同一约束。仅整理已存V经验校准，不重新扫q，不提供独立或conformal保证。

完整TX/RX/day/场景、source42、配对及逐类Wilson95%区间见[analysis](../local_artifacts/pamf_phase3_prototype_identity_20261031_r01/analysis.json)。区间描述固定输出，不能补足缺类或历史曝光。

## 成本与执行

新真实F0/signal/G前向均0；L特征/标签读取6300，无OE逐行值、Us或U新读取。新统计为六中心一组960个FP64值/7680 B及两新六τ/96 B；旧GMM/旧12τ/NN保持，共24τ192 B、单主6τ48 B。source原型fit1，不能笼统称统计拟合0。

实际四checkpoint合成烟测PASS：F0/G各2、signal共4（r03/PCD各2）。本轮比较可加载1169386神经参数，最小主PCD334721＋G266951=601672；F0另保留作基准/旧控制，中心不是NN新增权重。源行bank不部署；F011100/r032220/PCD444、G4940训练与源特征/缓存成本继承。
实际predict 22.134874 s、CPUscore 19.585845 s、pipeline 41.845286 s，属于缓存处理walltime，不是部署B1时延。4次观测独立核实driver1714247/predictor1714268的CWD/完整argv/CUDA5，GPU UUID ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331重复两次408 MiB，非峰值。
CPU1714989未独立live捕获，仅pipeline argv/gpuNone/ret0、CUDAempty实现和终态产物支持完成，不虚报遥测。三完整生产stdout为867/48/21 B、9/1/1行，warning/error/nonfinite均0；log0→542→575→867，终态live/owned GPU空、foreign0。

测试10 unique＋20sub：producer4＋4sub（初目录bug后仅定向1PASS/6.69s）、scorer4＋9sub/8.77s、launcher2＋7sub/3.05s；6AST/3CLI/实际config及唯一P0/P1 PASS。初始RED保留，测试2306条旧trapz警告与生产零警告区分，未新跑测试。
包773files/data0；SHA256 `841492de86435509a9b2716d0a04710b4e995a504b6540d0a7f79c6eb00b94bf` 单次比对、compile/runtime VERIFIED；真实pipeline无技术失败。

source V非blind，NV/Q历史曝光、单开发RX/seed、全局物理历史UNKNOWN，正式RX24-18未读。无新UDA或匿名注册，A_N/H_OW为N/A；原PCD与所有历史结果保持。源端最近中心较强不能直接决定目标身份规则，也不能据本Q挑下一阈值或原型参数。

完整[评分](../local_artifacts/pamf_phase3_prototype_identity_20261031_r01/score/results.json)、[预测终态](../local_artifacts/pamf_phase3_prototype_identity_20261031_r01/predict/complete.json)、[已存校准](../local_artifacts/pamf_phase3_prototype_identity_20261031_r01/predict/calibration/complete.json)可核查。
