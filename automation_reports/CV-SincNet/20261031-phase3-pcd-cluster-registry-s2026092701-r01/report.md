# PCD 无标签全簇关联与匿名注册

ANALYZED，执行 VERIFIED；科学 NO_PROMOTION，完整目标 ACTIVE。

# Phase3 固定 PCD 的无标签簇关联与匿名注册（2026-10-31）

Run `20261031-phase3-pcd-cluster-registry-s2026092701-r01`。使用已选 PCD 表征推进真实未来匿名注册，不继续只改身份头。原 signal/HIER 注册已显示 append-only 的固定旧 margin 不能提升同一后来群体的 R_U 或纠正旧类身份。本行改为全 U 原型判决及源支持回退，算法和参数在本次查询前固定，旧结果均保留。

## 冻结表示与新状态

严格复用 fresh f4b5/r01 的 TARGET_PCD444、原 F0/default6、DURA v1.0 R69、PCD known90/OE32/whitener，以及原六行阈值矩阵中的 PCD 两行（每行六阈值）。网络不更新、原 GMM/阈值不重拟合；PCD 有 U576 神经祖先，不是 source-only。先 actual F0/r03/G/PCD synthetic no-query smoke，再使用原 L6300 的最终 PCD z160 计算源统计；不能使用训练缓存中的 parent_z 冒充最终 PCD。

在原固定白化坐标下，用合法 L 标签建立六个均值，源支持半径 r_s 为 L 到自身类中心欧氏距离的总体95%分位，正数floor1e−6。该统计只用源 L，不读 OE/V/NV/U 真值，原源类中心不是另训神经头。U576 使用既有 views[0] P1 和当前最终 PCD 再前向一次，F0 同 P1 给出原生 closed；不重抽 received 或重做实际 DURA。

全 U 按 opaque ID 排序执行与旧 HIER 相同固定 [HDBSCAN 参数](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.HDBSCAN.html)：min_cluster_size16/min_samples8、euclidean/brute、eom、n_jobs1/alpha1/epsilon0/allow_single_clusterFalse/copyTrue/max_cluster_sizeNone。簇数不预设，保留所有非噪声簇。若簇内原六类 closed 的最高票占比≥0.8，且原 predicted_ratio margin 的中位数≥0，则关联该预测旧类；否则获得独立匿名槽。0.8 是本行事前固定一致性门槛，不来自目标标签或真实 K。

每簇均值和半径来自全部成员；r_j=min(r_s,该簇成员到均值距离95%分位)，同floor。原型包含旧类关联和匿名两类，匿名名按最小opaque ID规范排序，不携带真实TX。预测先选最小距离/r_j 的受支持原型（比值≤1）；命中则输出其旧类或匿名标签。无原型支持时，只有旧margin≥0且到其F0预测源类中心距离≤r_s才接受旧类，否则拒识。该规则能改判旧类，也能拒绝原来接受的无支持样本，不使用旧max(old,new)接受下界。所有计算逐query独立。

## 数据、矩阵与评分

仍用同一 RX3-2/index34、03-01 U576；未来03-15/23 Q10238，旧4508/可发现新4320/仅未来未知7-11为1410。仅未来未知从U整类排除。源L/OE/V/NV和Phase1角色不改，source seenRX、crossRX clean、crossRX star全经原DURA。source V27000/NV3150与两Q复用f4b5完整冻结logits/z/原PCD margin，不重读IQ或用它们拟合注册状态，clean不学习。该Q已曝光，非新的独立盲测。

四读出pooled_REG0/predicted_REG0/pooled_REG1/predicted_REG1，唯一主predicted_REG1；REG0精确复制PCD两列。REG1共用同一U簇状态，分别使用旧pooled/predicted margin回退。注册状态在任何V/NV/Q数值读取前冻结，全部4×3预测完后CPU scorer读U及sourceNV/clean/star四truth各一次；U真值不进入任何模型或注册状态。

新对齐采用U-only严格多数新类单元的一对一Hungarian，已知簇不匹配、旧类不重命名。未匹配匿名接受是错误身份，不算拒识；未来三可发现设备全部进入A_N，包括未发现者0。源NV19-15/19-16/7-1与目标5-5/10-7/2-19不同，source只报旧类与其全部未注册NV拒识，A_N/H_KN/H_OW=N/A。未来REG0对全四非旧类报R_U，REG1主R_U对固定future-only类9；另报相同later群体REG0/1配对及未成功注册真实类集合，不能把分母变化写成提升。

主报宏A_K/R_U/H、C_K、A_N和H_KN/H_OW；旧类正确接收相对同DURA原F0下降≤2pp、注册前后下降≤2pp，旧及预期注册集合C_K≥90%、总体A_N≥80%、注册后later R_U≥95%，与sourceknown12/必需景完整覆盖同时验收。A_N 的 80% 门槛沿用总体口径，各景完整报告；旧类与拒识保护同时检查总体和必需主景。原REG0仍按cleanRU95/starRU90。03-23缺两旧类的完整宏AK/CK/H保持N/A，later RU独立可报。零簇/漏发现不能绕过完整A_N分母。未全过不晋级/扩seed，完整目标ACTIVE。

## 执行与预算

唯一owner root，新的不可覆盖root/代码commit、聚焦负测及一次直接P0P1后登记；N607一次preflight/archiveSHA/compile，依次producer→独立CPUscore。source L新PCD前向6300、U新PCD576/F0576，真实signal6876/F0576/G0，原Q及源V/NV前向0；合成检查独列。新统计是源支持均值/半径、U HDBSCAN和K个全簇支持原型，无NN/新GMM/τ优化。部署继承834665神经参数，另六源中心和K簇中心/半径/标签，source/U逐行bank不部署。

本行只将U真值单独传入新release的score_truth路径，供预测全冻后的独立scorer；未来truth沿用既有路径，不传给producer。每GPU新增≤2、本行顺序worker且实际free≥4096MiB，不停止他人。非法来源/角色/状态改变/nonfinite/输出碰撞或不能闭合法预测即保留停止，低科学分数仍完成全部矩阵。不依据Q重选半径、簇阈值或读出；正式RX24-18不访问，结果不替代旧RX24-1困难集合。


实际代码 `16e6f04e7475827e7c8e24b398fb029a39a8138c`；环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-pcd-cluster-registry-s2026092701-r01-release`；owner `codex-phase3-pcd-cluster-registry-20261031`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.pcd_cluster_registry_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-pcd-cluster-registry-s2026092701-r01-release/runtime.json`。输入路径和角色见 experiment.json/runtime。


# Phase3 固定 PCD 无标签簇注册结果（2026-10-31）

**NO_PROMOTION，完整目标保持 ACTIVE。** HDBSCAN 形成 3 个簇、2 个匿名槽，但匿名槽均不满足 U-only 新类严格多数匹配；未来 A_N 为 0，且旧身份出现严重破坏。此次完整执行不改变原 PCD 研究起点或既有基线，不升级草案。

Run `20261031-phase3-pcd-cluster-registry-s2026092701-r01`；actual code `16e6f04e7475827e7c8e24b398fb029a39a8138c`；pipeline `ANALYZED_DEVELOPMENT`，预测与独立评分均 COMPLETE/0。设计与权限见[本轮计划](PAMF_PHASE3_PCD_CLUSTER_REGISTRY_PLAN_20261031.md)。

## 方法、数据与匹配

冻结 fresh/f4b5 的 TARGET_PCD444、原 F0/default6、DURA、known90/OE32 密度和 PCD 两套六阈值。用 L6300 最终 PCD 表征拟合六源中心与 95% 支持半径，再对 U576 全量 HDBSCAN；原型支持优先于旧判决，无支持时才按旧 margin 与源支持回退。因此原 F0 不变，但复合身份判决确实改变。没有新 NN、GMM、normalizer 或阈值优化。

源 L/OE/V/NV 为 6300/6134/27000/3150，源五 RX、三 day。目标 RX3-2/index34：U 为 03-01 的 576 条，future-Q 为 03-15/23 的 10238 条；其中旧类4508、可到达新类4320、future-only 1410。三个主条件均经同一 DURA，星地用于学习、clean 仅测试；source V 是既有校准角色，不是独立测试。

源 RX 为 1-19、18-2、19-2、2-19、3-19，日期为 2021-03-08、03-15、03-23；L 每 TX×RX×day 为70条。原 Phase1 隐藏 U_s56700 仅为继承训练来源，本行新读0。目标 U 在2021-03-01按六旧类＋三可发现新类各64条固定，真实配额只用于 builder/scorer；future 两日分别5525/4713条。U与future物理互斥，clean/star未来使用同一批10238条物理记录和相同顺序，不能相加为独立样本。

固定 HDBSCAN 1.7.0：min_cluster_size16/min_samples8、euclidean/brute、eom、n_jobs1、alpha1、epsilon0、allow_single_cluster=False。簇 378/25/26、noise147；旧类关联要求原头最高票≥0.8 且 predicted margin 中位数≥0。

| 簇 | 大小 | 原头最高票/比例 | margin 中位数 | 部署角色 | 评分端真组成 |
|---|---:|---|---:|---|---|
| 0 | 378 | 类4：173/378（45.7672%） | 94.846759 | 匿名0 | 旧0..5：49/52/5/42/51/13；新6/7/8：54/53/59 |
| 1 | 25 | 类2：23/25（92%） | 13.690792 | 关联旧类2 | 未保存真组成，N/A |
| 2 | 26 | 类5：25/26（96.1538%） | −26.204137 | 匿名1 | 全部真实旧类5（8-20），26条 |

匹配只使用已保存的 U truth-side 结果，严格多数分母包括簇内旧类污染，再做一次一对一 Hungarian。匿名0 的旧类共212条，三个新类均未过189条；匿名1 是旧类而非到达新类。映射为空，三个到达类均漏匹配；未匹配匿名接受按错误身份计，不是拒识。关联旧类25簇的真组成无已存字段，本报告不补读真值。

A_N 以真实到达5-5/10-7/2-19（标签6/7/8）全部三类为分母；两种未来条件分别为0/1454、0/1439、0/1427，合计0/4320。source NV19-15/19-16/7-1使用独立命名空间100/101/102，不套目标匿名映射，source A_N/H_KN/H_OW 均 N/A。

## 四读出 × 三条件

下表均为百分比。REG0 的 R_U 对全部非旧类；目标 REG1 的 R_U 仅对固定 later9（7-11，1410条），相应 H 的分母也改变，不能横比为注册收益。source 两态均对三个 source NV 设备。

| 读出 | 条件 | A_K | C_K | R_U | H | A_N | H_KN | H_OW |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pooled_REG0 | source | 89.770370 | 97.081481 | 63.619048 | 74.465443 | N/A | N/A | N/A |
| pooled_REG0 | clean_DURA | 32.123647 | 44.791196 | 60.790839 | 42.034854 | N/A | N/A | N/A |
| pooled_REG0 | star_DURA | 29.175335 | 50.230381 | 48.454372 | 36.420916 | N/A | N/A | N/A |
| predicted_REG0 | source | 89.733333 | 96.748148 | 64.285714 | 74.907247 | N/A | N/A | N/A |
| predicted_REG0 | clean_DURA | 31.661804 | 43.761391 | 61.979647 | 41.912794 | N/A | N/A | N/A |
| predicted_REG0 | star_DURA | 27.951722 | 48.582412 | 49.307698 | 35.678111 | N/A | N/A | N/A |
| pooled_REG1 | source | 33.448148 | 92.833333 | 69.365079 | 45.132976 | N/A | N/A | N/A |
| pooled_REG1 | clean_DURA | 7.414701 | 33.905864 | 68.865248 | 13.387928 | 0.000000 | 0.000000 | 0.000000 |
| pooled_REG1 | star_DURA | 4.221721 | 38.168624 | 54.042553 | 7.831647 | 0.000000 | 0.000000 | 0.000000 |
| predicted_REG1 | source | 33.437037 | 92.818519 | 69.301587 | 45.109417 | N/A | N/A | N/A |
| predicted_REG1 | clean_DURA | 7.414701 | 33.905864 | 68.865248 | 13.387928 | 0.000000 | 0.000000 | 0.000000 |
| predicted_REG1 | star_DURA | 4.221721 | 38.168624 | 54.042553 | 7.831647 | 0.000000 | 0.000000 | 0.000000 |

REG0 六个运行点与 fresh 的 PCD 对应历史 JSON 在 A_K/C_K/R_U/H/同 DURA 原 F0 闭集上逐值一致；原 producer/scorer 已完成控制复用核验，本报告未重读任何预测 NPZ。主 REG1 的原 F0 闭集损伤依 source/clean/star 为57.544444/41.152314/37.499337个百分点，相对同方法 REG0 的旧 A_K 损伤为56.296296/24.247103/23.730001个百分点。

## 同一物理群体的配对与身份损伤

| predicted_REG1 − predicted_REG0 | 旧正确 gain/loss/net | 旧接受 net | 固定 later 拒识 gain/loss/net | 全部非旧拒识 net |
|---|---|---:|---|---:|
| source | 0/15200/-15200 | -1061 | 0/0/0 | 158 |
| clean_DURA | 0/1515/-1515 | -467 | 96/65/31 | 326 |
| star_DURA | 0/1481/-1481 | -402 | 79/107/-28 | 232 |

source 旧正确从24228降到9028，旧接受仅从26122降到25061；损伤包含大量错误身份接收，不能仅称“误拒增加”。source/clean/star 未匹配匿名接受总数为16580/3490/4554。pooled 注册分支也分别净损旧正确15207/1533/1501；完整两套配对、各 TX/RX/day/场景计数见私有 analysis。

固定 later 的 predicted REG0→REG1：clean940→971/1410（66.666667%→68.865248%，净+31），star790→762/1410（56.028369%→54.042553%，净−28）。因没有匹配成功的新类，实际仍未注册集合为6..9；该四类宏拒识 clean61.979647%→67.655082%、star49.307698%→53.315950%。这些局部拒识变化不补偿 A_N=0 与旧身份损伤。

## 分组、覆盖与门槛

主 source 六旧类 A_K 依14-10/14-7/20-15/20-19/6-15/8-20为2.777778/3.044444/91.400000/7.822222/0.377778/95.200000%；五 RX 的宏 A_K 为31.537037%至36.277778%，三 day 为33.211111%至33.877778%。source known12 与完整42组均失败，不能继承旧阈值的保护结论。

未来主旧类正确数依相同六类顺序，clean为77/11/0/0/9/35，star为39/10/2/0/14/18；各类分母1491/1182/150/406/1186/93，宏指标不能用总正确率替代。源 NV 三类拒识为79.714286/65.333333/62.857143%。

| 星地场景 | 条数 | A_K（%） | C_K（%） | later R_U（%） | A_N（%） |
|---|---:|---:|---:|---:|---:|
| practical_high | 462 | 6.222222 | 37.020226 | 69.565217 | 0.000000 |
| practical_mid | 736 | 3.676404 | 32.335127 | 62.385321 | 0.000000 |
| practical_low_suburban | 3882 | 4.717277 | 37.243140 | 58.118081 | 0.000000 |
| practical_high_urban | 441 | 1.061628 | 49.497944 | 54.838710 | 0.000000 |
| practical_mid_urban | 712 | 6.550300 | 37.933647 | 53.191489 | 0.000000 |
| practical_low_urban | 4005 | 2.882953 | 39.406825 | 46.254682 | 0.000000 |

本池总体和六主景类覆盖齐全；各景门槛仍全部失败。03-23 缺旧20-19与8-20，完整 A_K/C_K/H/预期注册集合宏指标保留 N/A，later R_U 可独立报告：clean64.033850%、star50.211566%；03-15 为73.751783%/57.917261%。不得删除缺类后认证。

总体 A_N≥80%、旧及预期注册集合 C_K≥90%、原 F0/REG0 旧 A_K 损伤≤2个百分点、注册后 later R_U≥95% 均未通过；scene 的 A_N 仅报告而不额外设80%硬门槛。主预期注册集合（旧6＋全部真实到达3）的接受率 clean33.520012%/star41.087833%，准确率4.943134%/2.814481%；它不是“已成功注册的9类”，实际仅6旧＋2无语义匿名槽且匹配0。

## 成本、执行与声明边界

新增真实 signal 前向6876（L6300＋U576）、F0前向576、G前向0；U既有 DURA P1复用，source V/NV/Q无新前向。实际合成烟测 F0/G各2、signal共4（r03/PCD各2）PASS。新优化/GMM/阈值拟合均0，但六源支持中心/半径和全U簇原型是本轮新统计；不能统称“无拟合”。

部署神经参数仍834665，外加六源中心/源半径及3簇中心/半径/标签；源/U逐行bank不部署。JSON给出簇半径16.612922204/15.471611624/16.612922204，但未提供源半径精确值与数组总字节，保留缺项而不读NPZ。旧 F0、r03、PCD、G训练及缓存成本继承。
实际 walltime：predict 9.193640 s、CPUscore 4.929865 s、pipeline 14.234630 s；不作B1时延。4次独立观测核实driver1681561、predictor1681587的CWD/argv/CUDA5，GPU UUID ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331单次376 MiB为离散样本、非峰值；CPU1682212独立live/CWD/argv/CUDAempty已核。
三份生产 stdout 全读：predict841 B/9行、score50 B/1行、driver21 B/1行，warning/error/traceback均0；预测日志0→522→841 B，终态live/owned GPU均空、foreign0。全部预测先冻结，scorer随后U/sourceNV/两Q真值各连接一次；Q不用于映射。

验证为producer6 unique＋6sub（初3过、3修复定点8.93s）、scorer6＋7sub/10.87s、launcher2＋7sub/3.13s；共14 unique＋20sub。保留初始RED，scorer最终测试1978条旧trapz警告不等于生产警告。8 AST/3 CLI/实际config与一次P0/P1审查PASS，未新增本轮报告测试。
发布包763files、data1；SHA256 `947ae0872c4fd666dda2d824f2ef085bc136dcf0cfa2d09ad5fe1e273b4cd809` 一次比对、compile/runtime VERIFIED。发布前登记kind与私有模板复用问题已在启动前修正，不构成真实pipeline故障或重复实验。

本轮只验证已曝光RX34、单seed的有限未来注册规则；全项目物理使用历史UNKNOWN，正式RX24-18未读，clean不学习。不同未知集合和日期不能替代旧RX24-1困难结果；没有多节点确权或持续多session注册成功。PCD原研究起点及所有历史结果保持，本次失败仅约束这套冻结支持原型规则。

完整[评分结果](../local_artifacts/pamf_phase3_pcd_cluster_registry_20261031_r01/score/results.json)、[注册元数据](../local_artifacts/pamf_phase3_pcd_cluster_registry_20261031_r01/predict/registry.json)与[精简全分组分析](../local_artifacts/pamf_phase3_pcd_cluster_registry_20261031_r01/analysis.json)可核查。
