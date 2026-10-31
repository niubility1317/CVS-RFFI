# 源侧统计子空间拒识

ANALYZED，执行VERIFIED，科学NO_PROMOTION，完整Phase3 ACTIVE。

# 源侧无标签子空间与冻结拒识

Run `20261101-phase3-ground-geometry-s2026092701-r01`，owner `codex-phase3-ground-geometry-20261101`。上一轮尾部损失配对实验为实测进展，但源RU仅小幅提高且跨RX退化，不能晋级。本轮在完整原三视图F0上学习源侧拒识几何，神经模型保持冻结；完整Phase3的无标签目标适应、发现与未来匿名注册目标不缩减，当前仍优先地面开放能力。

## 固定方法

参考[ViM原论文](https://arxiv.org/abs/2203.10807)及[官方实现](https://github.com/haoqiwang/vim/blob/master/benchmark.py)：将原生特征残差与六类logits能量结合。标准分支沿用低维160空间的80维残差、uncentered二阶矩和源meanMaxLogit/meanResidual定alpha；源域均值约束分支是本项目变体，不声称官方复现或已保证接收机不变性。

唯一神经识别器为完整原F0 Tri-SIG-TIME SSL E50/11100 reader/default6，232993参数，头960，无bias、Python scale30；DURA固定R69 v1.0、266951参数/E20/4940/reuse=True/graph=True/native=False，FP32/TF32off。F0/r03/G全部frozen eval、参数/buffers不变，无优化器、教师更新、EMA、替代头、源OE或目标U学习。新状态是统计拟合，不称新增神经训练或全系统无标签。

必须取当前F0分类头的实际FP32坐标x=F.normalize(raw160,eps1e-4)，并在同一次forward逐项验证logits=30*F.linear(x,F.normalize(W,eps1e-4))。不复用旧FP64 unit/raw特征冒充此坐标，不再归一化。原head bias0，因此ViM原点为精确0，不作全样本均值减法或协方差shrinkage。

只用旧G(clean)源Us56700的IQ和公共RX/day拟合两套状态，不读Us真实TX、不生成伪标签、不读取L/OE值。Us在原Phase1合同中属于六已注册身份，已知成员资格来自原合法角色，并非混合未知池；这不代表整个系统从未用过标签。原F0及旧SSL对照的源Us祖先保留。

standard：M=X'X/N的最低80特征向量作为残差基R。domain_orthogonal：按公共15个RX/day求源均值μd；14行μd−μ0与居中的15均值张成同一空间，用FP64 fullSVD及eps*max(shape)*smax数值rank得Nuisance基B、秩r≤14；在B的正交补C（160−r维）上作uncentered (X C)'(X C)/N，取最低80向量并映回R=C Vlow。两分支残差均固定80维；后者principal维为160−r−80（r14时66），不是再次折半。约束R'R=I80、B'R=0。若域均值无差异r0，等价标准分支；数学残差退化或alpha非正则明确失败，不静默回退。

每分支alpha=mean(max六logits)/mean(||xR||)，分数s=logsumexp(logits)−alpha||xR||，高分接收。B只作拟合/审计，推理只需R、原点和alpha，不使用query统计、目标RX标签、TX、真实K或配额。该约束只消除选定源均值方向对残差的贡献，原logits路径仍可能受域影响，不能据B'R=0宣称整个系统域不变。

固定四读出F0_energy、OLD_SSL_energy、F0_vim、F0_domain_vim；唯一主F0_domain_vim。前两能量控制逐值复用旧ground r03的col0/2和对应六阈值；标准ViM隔离新残差约束的作用，不依据结果改主。两新读出用同一次当前F0原默认头和坐标；保存freshF0 logits与旧canonical F0独立基准，逐条件报告FP32重放最大logit差和argmax差。小数值差不改标成头重训，也不伪称与旧cache逐值相同；原F0基准不能被fresh候选闭集替代。

## 数据与角色

直接复用`20261101-phase3-ground-open-s2026092701-r03` / code `d7e0f875397b2f9801df4605ba0051d221aa4e2a`的VALIDATED_ONCE胶囊与物理ID，不重建。源split392005、eq1、256点、输入RMS一次；六注册TX14-10/14-7/20-15/20-19/6-15/8-20，五源RX index1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23。Us56700只读四公共metadata字段ids/rx_i/day_i/domain，原合同每公共域3780，不读sig_i/base_index或TX标签。

CAL13500和已知TEST13500来自原V27000每y×RX×day格150/150、seed2026110101，物理互斥。旧builder合法用V已知y分层后将TEST真值导出私有侧；本轮拟合/校准/预测不读TEST标签。仅CAL13500已知y用于阈值身份保护，源NV3150（19-15/19-16/7-1各1050）只评分。TEST加NV为源Q16650，clean/原固定G10星地同序，public不含known/unknown角色。历史V/NV/Q已曝光，本轮不是新的盲测。

跨RX仍为开发RX3-2/index34、未来03-15/23 Q10238（5525/4713），六已知4508（1491/1182/150/406/1186/93），四未注册5730（5-5/10-7/2-19/7-11各1454/1439/1427/1410）。03-23缺20-19/8-20，完整已知指标N/A，保留present-only诊断而不替代完整名单。未知表示未注册，不证明未在OE或历史出现；不新增OE TX读取去核对。正式RX24-18未访问。

全部source_clean_DURA/source_star_DURA/cross_clean_DURA/cross_star_DURA四条件保持固定DURA。源G(clean)/G10 P1复用；目标clean仅冻结测试，星地为原单物理记录单固定practical residual received：G10 MIX01 600km/25MHz/2.45GHz、post_sync/noEQ/refSNR30dB/reflect、源scene3001/channel4001、future3133/3134、receiver2027、输出不再RMS。跨RX两条件同10238物理记录，逐记录经同G，不重抽信道或用clean参考恢复query。

## 冻结与正式验收

神经原F0/G始终冻结；两个统计模型先于CAL冻结。新两方法只用CAL已知24组（overall/5RX/3day/15RXday）固定公共低尾规则，保护CK≥95%、同输入canonical原F0闭集−AK≤2pp；rank/r/alpha不由CAL/NV/target挑选。无缺预测类或零尾仍无法保护时记录失败，合法预测继续，不以测试修复。新12阈值＋旧12=24个FP64/192B，主方法六值48B。只保存聚合basis/alpha、域方向审计和必要预测坐标，不部署source行bank。

固定4读出×4条件共16点全部完成后，独立CPU连接source_knownTEST/sourceNV/futureclean/futurestar四真值各一次；旧F0/SSL四overall共8项必须精确保持。CPU只重建分数/阈值与指标，不重做PCA/SVD、不读Us数组或运行NN。source三未知字符串与目标四标签使用各自名单，宏AK/CK/RU/H按真实TX等权；报告原与fresh闭集、逐RX/day/scene/day×scene/TX、CI/AUC/OSCR和配对净变化。

正式门槛不变：全部条件宏RU≥95%、每预登记主场景≥95%、CK≥95%、同canonical原F0损≤2pp且所需名单完整。源clean全部24组；crossclean overall＋clean；source/cross星地overall＋六主景。ground_source_success/cross_clean_success/ground_stage_success与四条件frozen_rejection_success分别报告；scientific_success仅四条件全部通过且主CAL保护通过。通过仅是冻结拒识子阶段，后续UDA/发现/未来匿名注册仍未完成；否则NO_PROMOTION。旧90仅历史诊断，不作为晋级；完整Phase3 ACTIVE，AN/H_OW=N/A，不按新Q改主、调参或扩seed。

## 执行与成本

默认N607，卡0至7启动free≥4096MiB，每卡至多新增2本任务worker，不停止或修改他人任务。单owner/不可覆盖run，GPU predict进程顺序实际CP无query smoke→源Us拟合→CAL→四Q，随后独立CPU score。原F0真实前向123976=Us56700＋CAL13500＋两源Q33300＋两跨RXQ20476，G新20476、SSL新0、NN优化0；合成F0/G各2单列。旧数据G99284/F036450、原F0/SSL/G训练及缓存为继承成本，不重计新导出。

主部署神经参数499944，残差读出12961个FP64数字/103688B，加六阈值48B；域方向B审计额外160*r*8B，不将其或原点数据隐藏。完整两模型统计与四读出阈值审计矩阵另报，不把统计值计作神经参数。whole-stage墙钟不是B1时延。

具体commit/精确命令/Python/CWD/路径/GPU/预期产物写在同run登记；一次独立P0/P1、聚焦协议负测和实际CP首smoke PASS即继续，单archive一次本地/远端SHA及compile，实际PID/CWD/argv/GPU/log增长核对。停止仅协议/来源/输出冲突、非有限或无法合法预测；不按低分技术重跑。代码/登记/汇总Git交付，IQ/weights/原日志/运行时和私有进程状态不提交。


code `9e0a97852a1a8688e4a5410490fc0d8a7a49ef34`；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261101-phase3-ground-geometry-s2026092701-r01-release`；command `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.ground_geometry_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261101-phase3-ground-geometry-s2026092701-r01-release/runtime.json`。


# Phase3 源域几何读出结果（2026-11-01）

**本轮执行 VERIFIED，科学结论为 NO_PROMOTION。** 固定主方法 `F0_domain_vim` 达到了预定的源域正交约束，却没有达到未知拒识目标：源 clean 的 A_K/C_K/R_U/H 为 97.874074%/98.866667%/26.349206%/41.520465%，跨接收机 star 为 32.778756%/65.775654%/31.046922%/31.889343%。源已知保护通过不能替代未知拒识；本轮不替换原有方法，不宣称完成 Phase3，完整目标仍为 ACTIVE。

Run 为 `20261101-phase3-ground-geometry-s2026092701-r01`，实际代码为 `9e0a97852a1a8688e4a5410490fc0d8a7a49ef34`。矩阵、主方法和门槛在访问本轮查询前固定，见[计划](../../../docs/PAMF_PHASE3_GROUND_GEOMETRY_PLAN_20261101.md)；全部结果见[评分 JSON](../../../local_artifacts/pamf_phase3_ground_geometry_20261101_r01/score/results.json)和[完整私有分析](../../../local_artifacts/pamf_phase3_ground_geometry_20261101_r01/analysis.json)。

## 方法与数据权限

原 F0 默认六类分类头、骨干和 DURA v1.0 全部冻结。本轮没有神经优化、另训分类头、投影层更新、目标适应或匿名注册；两套源域子空间和尺度 α 是新统计学习，不能称为“没有学习”。`OLD_SSL_energy` 来自 ground-open r03，作为继承对照；此前未运行的 ViM 草稿、ground-tail 学生和其他历史候选不属于本矩阵。

新拟合只使用源 Us 的 56700 条记录及公开 RX/day，不读取其 TX。Us 属于原六个已注册身份的源角色，并非未知目标混池。坐标为实际原生分类器同一次前向产生的 Torch FP32 `F.normalize(raw160, eps=1e-4)`；保留坐标，不用旧密度归一化缓存替代，不再次归一化。默认 logits 满足原头 `30·linear(x, normalized_W)`。

标准 ViM 对未中心化二阶矩 `M=XᵀX/N` 取最低 80 个特征向量为残差基 R；原点为 0，不减经验均值，不加收缩。域正交分支先取 15 个公共 RX/day 均值差的张成空间 B，再在其正交补内取最低 80 维残差；实际 r=14，因此有效空间 146 维、主空间 66 维，残差仍为 80 维。

两分支分别以源 Us 的 `α=mean(max logits)/mean(‖xR‖)` 冻结尺度，分数为 `logsumexp(logits)−α‖xR‖`，高分接收。主方法始终为域正交分支；标准 ViM 仅为机制对照。B 的约束只针对残差通道，不保证默认头 logits 不随域变化，也不保证未知类可分。

本轮不新读取 L/OE 的值，但不能把完整流程称为无标签：原 F0 具有六类有标签及源半监督训练祖先；旧 SSL 对照另有 L/Us/OE 训练，其中 OE 只提供 outside-six 成员监督，不能将该对照的 OE 训练归到原 F0。本轮 CAL 仍使用 13500 条已知真实标签保护阈值。OE 的具体 TX 本轮未读，“未知”只指未注册进六类，不自动代表从未在 OE 或历史中出现。

| 角色 | 容量与使用范围 |
|---|---|
| 源祖先 | L 6300、Us 56700、OE 6134；源 5 RX × 3 日、6 个已知 TX。本轮只有 Us 值进入几何拟合，L/OE 新读取为 0。 |
| CAL / known TEST | 原历史 V 27000 按真实类 × RX × day 分成各 13500；每个单元 150/150，TEST 每类 2250。源 V 不是新的盲测集。 |
| 源 NV | 3150，19-15、19-16、7-1 各 1050；与 known TEST 合成每条件 16650 条角色盲查询。 |
| 跨 RX 查询 | 开发 RX 3-2（索引 34），03-15/03-23 各 5525/4713，共 10238；known 4508、四个未注册 TX 共 5730。clean/star 使用同一物理集合。 |

源六类依次为 14-10、14-7、20-15、20-19、6-15、8-20；源 RX 索引 1/3/4/6/8 对应 1-19/18-2/19-2/2-19/3-19，日期为 2021_03_08、03_15、03_23。跨 RX 六类容量依次为 1491/1182/150/406/1186/93；未注册 5-5/10-7/2-19/7-11 分别为 1454/1439/1427/1410。源 NV 字符串与目标标签 6 至 9 是不同命名空间。

四条件均经过同一冻结 DURA：源 clean 为 G(clean)，源 star 复用固定星地缓存；跨 clean/star 经相同 G 处理。G10-MIX01 物理、输入 RMS 与信道种子保持，不重抽信道；目标 clean 只测试。本轮不读目标 U 或 U IDs，也不把源 Us 当目标适应。构建者曾合法读取历史 V 已知 y 做分层分割和私有 TEST 真值导出；本轮拟合、校准和预测不读取 TEST 标签。

## 子空间与重放证据

| 已存源统计 | 标准 ViM | 域正交 ViM（主） |
|---|---:|---:|
| 拟合条数 / 公共域数 | 56700 / 15 | 56700 / 15 |
| nuisance rank / 有效维数 / 主维数 / 残差维数 | 0 / 160 / 80 / 80 | 14 / 146 / 66 / 80 |
| α | 409.621150856 | 394.677980402 |
| 源平均最大 logit / 平均残差 | 7.135830891 / 0.017420563 | 7.135830891 / 0.018080134 |
| 特征值最小 / 最大 | 4.054602e−7 / 0.755636870 | 4.176093e−7 / 0.367144278 |
| 二阶矩迹 / 最低 80 维迹 | 1.000000000 / 0.000366905 | 0.427033553 / 0.000395316 |
| 最大 BᵀR 元素绝对值 | 0 | 2.943260e−16 |
| 残差空间域中心最大距离 | 0.004192403 | 1.217537e−16 |

15 域各有 3780 条；两个谱均完整有限、升序且无负特征值。以上来自实际 producer 的完整 JSON 诊断，本次分析没有打开数组或重算 PCA/SVD。正交和域均值消除约束达到数值精度，不能据此推导跨 RX 不变性或开放识别成功，见[模型诊断](../../../local_artifacts/pamf_phase3_ground_geometry_20261101_r01/predict/models/complete.json)。

新前向与原缓存的 5 组重放均无 closed 改变：CAL/source clean/source star 的最大 logit 绝对差分别为 5.245209e−6/5.364418e−6/5.245209e−6，跨 clean/star 为 0。新 logits 与原基准分别保存，未伪称源 logits bit exact；两个 ViM 共用新 F0 身份输出，原基准始终是同 DURA 输入的原 F0。

## 全部 16 个运行点

以下均为百分比，按真实 TX 等权宏平均；C_K 为已知接收率，H 为 A_K 与 R_U 的调和均值。“闭集”为该行分类器自身无拒识结果，保护指标仍使用原 F0 基准。条件依次表示源 clean、源 star、跨 clean、跨 star。

| 方法 | 条件 | 闭集 | A_K | C_K | R_U | H | AUROC | OSCR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F0 energy | source clean | 98.414815 | 97.911111 | 98.637037 | 51.523810 | 67.517732 | 97.156614 | 96.408056 |
| F0 energy | source star | 90.918519 | 89.451852 | 96.370370 | 22.380952 | 35.803763 | 80.457738 | 76.517416 |
| F0 energy | cross clean | 48.567015 | 39.679774 | 62.791866 | 34.335698 | 36.814809 | 47.115393 | 29.003672 |
| F0 energy | cross star | 41.721059 | 34.843320 | 67.736189 | 31.390511 | 33.026917 | 44.653202 | 23.058292 |
| OLD_SSL energy | source clean | 98.518519 | 97.755556 | 98.303704 | 69.365079 | 81.148828 | 97.115273 | 96.266180 |
| OLD_SSL energy | source star | 91.000000 | 90.059259 | 98.014815 | 38.380952 | 53.823644 | 81.138768 | 75.505275 |
| OLD_SSL energy | cross clean | 49.210482 | 37.155045 | 53.978578 | 55.609637 | 44.546664 | 56.840596 | 35.639902 |
| OLD_SSL energy | cross star | 42.379209 | 35.447581 | 62.520184 | 40.971269 | 38.009794 | 54.569913 | 29.455873 |
| F0 ViM | source clean | 98.414815 | 97.800000 | 98.725926 | 26.253968 | 41.395501 | 96.540778 | 95.908564 |
| F0 ViM | source star | 90.918519 | 89.948148 | 97.755556 | 17.968254 | 29.953022 | 85.388247 | 79.827184 |
| F0 ViM | cross clean | 48.567015 | 35.360765 | 59.714642 | 34.126850 | 34.732852 | 42.865903 | 25.818258 |
| F0 ViM | cross star | 41.721059 | 30.635462 | 61.860054 | 32.855555 | 31.706693 | 41.871586 | 21.464122 |
| F0 domain ViM | source clean | 98.414815 | 97.874074 | 98.866667 | 26.349206 | 41.520465 | 96.775941 | 96.127356 |
| F0 domain ViM | source star | 90.918519 | 90.133333 | 98.022222 | 16.857143 | 28.402350 | 85.854140 | 80.180783 |
| F0 domain ViM | cross clean | 48.567015 | 36.674881 | 63.548753 | 31.916523 | 34.130653 | 43.797100 | 26.380247 |
| F0 domain ViM | cross star | 41.721059 | 32.778756 | 65.775654 | 31.046922 | 31.889343 | 42.368116 | 21.745591 |

两个旧能量方法的 logits、closed、分数和阈值精确复用；评分后 8 个完整 overall 对象与[原 ground-open r03 结果](../../../local_artifacts/pamf_phase3_ground_open_20261101_r03/score/results.json)逐值一致。该核对读取已有 JSON，没有重新连接真值或评分，也没有修改旧门槛或旧结果。

## 相同 ID 配对与身份、拒识分解

下表均为“主方法减对照”；宏变化单位为百分点，正确接收和未知拒对为实际净条数。相对 F0 energy/标准 ViM，身份 closed 完全相同，变化来自接收决策；相对 OLD_SSL，还包含不同冻结分类器的身份差异。

| 对照 | 条件 | ΔA_K | ΔR_U | ΔH | 正确接收净数 | 未知拒对净数 |
|---|---|---:|---:|---:|---:|---:|
| F0 energy | source clean | −0.037037 | −25.174603 | −25.997267 | −5 | −793 |
| F0 energy | source star | +0.681481 | −5.523810 | −7.401413 | +92 | −174 |
| F0 energy | cross clean | −3.004893 | −2.419175 | −2.684157 | −49 | −136 |
| F0 energy | cross star | −2.064564 | −0.343588 | −1.137573 | −45 | −18 |
| OLD_SSL energy | source clean | +0.118519 | −43.015873 | −39.628364 | +16 | −1355 |
| OLD_SSL energy | source star | +0.074074 | −21.523810 | −25.421294 | +10 | −678 |
| OLD_SSL energy | cross clean | −0.480164 | −23.693114 | −10.416011 | +103 | −1352 |
| OLD_SSL energy | cross star | −2.668825 | −9.924347 | −6.120451 | +20 | −567 |
| F0 ViM | source clean | +0.074074 | +0.095238 | +0.124964 | +10 | +3 |
| F0 ViM | source star | +0.185185 | −1.111111 | −1.550672 | +25 | −35 |
| F0 ViM | cross clean | +1.314117 | −2.210327 | −0.602199 | +38 | −127 |
| F0 ViM | cross star | +2.143294 | −1.808632 | +0.182650 | +30 | −104 |

源 clean 相对 F0 energy，正确接收为 46 增/51 失，错误接收净增 36，未知拒对 286 增/1079 失；已知接收率稍高并不代表开放性能改善。跨 star 相对标准 ViM，正确接收 42 增/12 失、错误接收净增 64、未知拒对 41 增/145 失，H 的微增伴随 R_U 下降，不能据此晋级。

相对 OLD_SSL，跨 clean 的正确接收净增 103，但宏 A_K 下降 0.480164 个百分点；跨 star 净增 20，宏 A_K 仍下降 2.668825 个百分点。各已知类容量不等，宏平均与总条数可以方向不同。跨 star 的 closed 纠错为 93 增/99 失，错误接收净增 183、未知拒对净减 567，不能将更多接收解释为身份纠错或成功拒识；全部配对分组保存在评分 JSON。

## 分组、困难类别与正式判定

CAL 的四方法均通过已知保护；主方法共同尾比例为 12/1121。新子空间先冻结，随后仅 CAL 拟合两套 6 类阈值，12 个新值与 12 个旧值合计 24 个 FP64、192 B。CAL 的已知约束不包含未知拒识承诺，见[校准结果](../../../local_artifacts/pamf_phase3_ground_geometry_20261101_r01/predict/calibration/complete.json)。

主方法源 clean 的已知 24 组全部通过：C_K 为 95.666667% 至 99.888889%，原 F0 闭集减 A_K 为 0 至 1.444444 个百分点；同 24 组 R_U 仅 0.952381% 至 55.238095%，没有一组达到 95%。源 star 的 known-24、known-12-scene 和 known-42 诊断均未全过，不能用 clean 或 CAL 的通过替代。

源 clean 五 RX 的 R_U 依索引 1/3/4/6/8 为 19.365079%/39.523810%/16.666667%/29.206349%/26.984127%；三日为 31.714286%/19.904762%/27.428571%。源 star 三日降为 20.190476%/16.571429%/13.809524%，困难不限于某一个接收机或日期。

源 NV 三类的主方法拒识：clean 为 19-15 363/1050（34.571429%）、19-16 182/1050（17.333333%）、7-1 285/1050（27.142857%）；star 为 232/1050、127/1050、172/1050（22.095238%/12.095238%/16.380952%）。19-16 仍最困难；这些事实不能回流为按 NV TX 调阈值的权限。

| 主要星地场景 | 源 C_K | 源 R_U | 源原 F0 损失 pp | 跨 C_K | 跨 R_U | 跨原 F0 损失 pp |
|---|---:|---:|---:|---:|---:|---:|
| practical_high | 98.755411 | 22.971465 | 0.216450 | 61.696534 | 32.230255 | 10.785395 |
| practical_mid | 99.276566 | 22.282707 | 0.107527 | 61.024726 | 34.430924 | 15.166303 |
| practical_low_suburban | 98.483869 | 18.041140 | 0.648150 | 67.179947 | 30.204941 | 10.404341 |
| practical_high_urban | 98.571553 | 20.156076 | 0.379229 | 67.322496 | 29.358681 | 10.854976 |
| practical_mid_urban | 98.166898 | 15.683761 | 0.910983 | 54.958167 | 32.392901 | 13.787625 |
| practical_low_urban | 97.149277 | 13.659893 | 1.123552 | 67.248592 | 31.090571 | 5.372918 |

主方法跨 clean/star 的六类 A_K 依上述已知顺序为 53.789403/11.252115/11.333333/2.216749/66.188870/75.268817% 和 48.490946/11.082910/13.333333/2.463054/63.237774/58.064516%；四未知 R_U 依 5-5/10-7/2-19/7-11 为 29.848693/52.744962/20.462509/24.609929% 和 26.272352/50.034746/22.845130/25.035461%。总体宏数没有掩盖某些已知类几乎无法正确接收的问题。

03-15 的跨 clean/star A_K 为 36.576669%/32.235964%，R_U 为 33.180302%/31.492988%；03-23 缺 20-19、8-20，完整六类 A_K/C_K/H 为 N/A，R_U 仍可报 30.672954%/30.599495%。该缺类规则同样用于日 × 场景，不用 present-only 宏数补认证。总体和六个场景的完整名单检查与日期缺类是不同层次，均保留。

95% Wilson 区间按真实类给出，属于已曝光开发池描述，不代表独立 RX 泛化置信度。例如源 clean 19-16 的 R_U 区间为 15.163886% 至 19.740933%，跨 star 2-19 为 20.741454% 至 25.094614%。完整 TX/RX/day/RX×day/scene/day×scene、24/42 组及 CI 全部保存在评分与分析文件，没有只抽首尾组。

排序与操作点也应分开：主方法源 clean AUROC/OSCR 为 96.775941%/96.127356%，低于 F0 energy 的 97.156614%/96.408056%；源 star 的曲线提高却仍只有 16.857143% 的当前拒识率。跨 clean/star 主 AUROC 为 43.797100%/42.368116%，不能凭正交约束或某一曲线局部变化宣称分离增强。

正式门槛保持 C_K≥95%、R_U≥95%、原同 DURA F0 闭集−A_K≤2 个百分点及完整预登记名单：源 clean 检查 24 组，跨 clean 检查总体与 clean，两个 star 条件检查总体与六主景。主方法四条件原 F0 损失为 0.540741/0.785185/11.892134/8.942303 个百分点，四条件均未通过联合门槛；`ground_source_success`、`cross_clean_success`、`ground_stage_success`、`frozen_rejection_success`、`scientific_success` 均为 False。旧 90% 诊断不参与晋级。

## 成本、执行和证据边界

新真实 F0 前向 123976 条＝Us 56700＋CAL 13500＋两源查询 33300＋两跨查询 20476；新 G 前向 20476 条，SSL/student 真实前向为 0。两统计模型共享一次 Us 前向，另有 1 次 nuisance SVD；新 NN/GMM/normalizer/optimizer 更新均为 0。实际无查询 smoke 在真实数据前通过，合成 F0/G 各 2 条，真实与合成计数分列。

原 F0 11100 步、DURA 4940 步、OLD_SSL 的源训练和 Us 曝光、旧 D7 数据 G 99284/F0 36450 前向及缓存成本继续继承，不因本轮复用清零。无本轮神经训练曲线。源 Us 行库、CAL/查询记录及 nuisance 审计基不部署。

主神经部署为 F0 232993＋G 266951＝499944 参数；主几何运行状态 12961 个 FP64 数值、103688 B，另 6 个阈值 48 B。域 nuisance 基另有 2240 个审计数值、17920 B，域完整统计状态为 121608 B；标准状态 103688 B，两状态合计 225296 B。上述为数值载荷，不是带 Unicode/schema 或压缩容器的文件大小，也不是已测时延收益。

预测、独立 CPU 评分、整条流水线实耗为 41.124949 s、8.486256 s、49.953804 s，不能当作 B1 推理延迟。[流水线](../../../local_artifacts/pamf_phase3_ground_geometry_20261101_r01/pipeline_state.json)两阶段均 COMPLETE/return 0；5 次[独立观测](../../../local_artifacts/pamf_phase3_ground_geometry_20261101_r01/observations.jsonl)捕获 driver、预测和 CPU 评分 live，预测使用 GPU 3，304/576 MiB 是离散样本而非峰值，评分 CUDA 为空；终态 owned live/GPU 为空，foreign 操作为 0。私有进程、设备标识和完整命令留在本地证据。

三份真实 stdout 全量读取：predict 959 B/10 行、score 45 B/1 行、driver 21 B/1 行，无 warning/error。预测日志记录真实原模型 smoke、Us 完成、两模型冻结及四查询完成；评分日志终态为 NO_PROMOTION。测试日志独立归类，未将 `launch_fixture` 等合成树当实测。

本轮新验证合计 16 unique/83 subtests：core 6/25、run 2/9、score 5/35、launch 3/14；另旧 ViM core 6/19 是此前验证复用，不计新实验。scorer 包含先 RED 后定向 GREEN 和一次完整合成 producer→16 点→CPU 四真值交接；未声称初轮全绿。run/score 的 972/1134 条旧 `np.trapz` 弃用警告只出现在合成测试，真实运行无警告；不重复测试或追加评审。

一次独立 P0/P1 [审查 PASS](../../../local_artifacts/pamf_phase3_ground_geometry_20261101_r01/review.json)，真实 CP 首 smoke、preflight、单包比较及远端 compile/runtime 均已核实；[发布验证](../../../local_artifacts/pamf_phase3_ground_geometry_20261101_r01/release_verification.json)为 VERIFIED。包含 908 个文件、数据 0，SHA256 为 `35ee1634edad42fbda52838a4c88abf3f62bd23899c5832c6eabca0492d8e8b0`，没有实际技术失败。

两子空间先于 CAL 冻结、所有旧新阈值先于查询消费冻结，全部 16 点固定后独立 CPU 只连接 source known TEST、source NV、cross clean、cross star 四真值各一次；Us 真值读取为 0。此分析只整理既有 JSON/JSONL/stdout，没有读取数组、权重或私有真值，没有重拟合、重评分或扫描查询阈值。

本轮仅否定这套固定源域几何读出的晋级资格。跨 RX 原 F0 闭集本身为 48.567015%/41.721059%，可能构成身份能力限制，但本轮未分离其因果贡献；也不能将拒识下降归结为唯一机制，或据此否定全部几何方法。源 CAL/TEST 来自历史 V，NV/Q 已曝光，单模型 seed、单开发 RX、部分日期重合、全局物理历史 UNKNOWN，均不属于新独立盲测。

正式 RX 24-18 未读；本轮无目标 DA、发现、匿名注册或持续注册，`adaptation_success=None`，A_N/H_KN/H_OW 为 N/A。旧方法保持原身份，完整目标 ACTIVE；当前结果不用于按未知 TX 或目标真值选择 rank、α、阈值或下一轮参数。
