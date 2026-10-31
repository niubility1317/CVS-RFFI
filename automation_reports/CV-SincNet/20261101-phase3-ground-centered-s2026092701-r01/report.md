# 地面中心化证据训练

ANALYZED，执行VERIFIED，科学NO_PROMOTION，完整Phase3 ACTIVE。

# 地面默认头的中心化证据训练

Run `20261101-phase3-ground-centered-s2026092701-r01`，owner `codex-phase3-ground-centered-20261101`。本轮继续用户指定的地面训练优先；完整跨RX星地开放识别、无标签适应、发现及未来匿名注册目标保持，正式RU≥95%、CK≥95%、原同DURA F0闭集−AK≤2pp及主场景95%不降低。原方法、数据与其他owner健康任务保持。

## 假设与固定方法

上一轮源Us子空间数值约束成立但拒识失败，不将其作为初始化。本轮从完整原F0独立训练一个原生学生CPAIR，默认六类头与骨干联合训练，Sinc48和全部buffers固定，232993参数/232945可训练/头960。原F0 Tri-SIG-TIME SSL E50/11100和DURA R69v1.0均不更新。没有投影、另训头、EMA、目标U或新数据增强。

原能量a(z)=logsumexp(z)。本轮证据c(z)=logsumexp(z)−mean(z)−log(6)，等于KL(Uniform6 || softmax(z))。对每条记录同时加同一常数，分类、源CE、teacher KL和c均不变，但a变化。该代数说明原排序有共同平移自由度；它不是既有训练实际走了捷径的证明，也不保证中心化将改善拒识。

参考[能量OOD原论文](https://arxiv.org/abs/2010.03759)及[OE原论文](https://arxiv.org/abs/1812.04606)的能量和均匀分布背景；本轮是自己的公共域内中心化配对排序变体，不声称复现任一论文的完整训练或保证。推理c用FP64 logits计算稳定LSE后减均值与log6，训练用TorchFP32同一公式，保持原六logits和预测类别，不改成中心化分类头。

共同CE为L真实六类交叉熵，KD=4*KL(softmax(original same G(clean) cached logits/2)||softmax(student logits/2))，两者权重1。每个公共RX/day域d，计算全部L/OE配对softplus(c(OE)−c(L))的均值，然后对15域等权；权重1。CPAIR只改变原DMEAN损失中排序使用的标量，不增加OE真实TX、K或未知语义。每step L192/OE64，同SourceSampler种子2026103017/18，逐步比对旧DMEAN全部2220步L/OE索引；同完整原F0初始化和2220更新、AdamW lr/wd1e−4/eps1e−8/clip5，最终步冻结，不选checkpoint。

旧DMEAN为`20261101-phase3-ground-tail-s2026092701-r01` / code `315f8ec1be994c40a97cbede4e27942e14267f12`的匹配控制，原D7 ground r03数据/教师不变。只用旧DMEAN的源训练祖先，不继承其权重进新学生。旧DMEAN真实模型仅在CAL新增前向，所有query列使用其既有完整cache；原能量列与阈值精确复用。

固定四方法：`DMEAN_energy`、`DMEAN_centered`、`CPAIR_energy`、`CPAIR_centered`；主方法预定`CPAIR_centered`。0/1同旧DMEAN身份，2/3同新CPAIR身份。相同模型内差异隔离评分规则，同读出跨模型差异隔离训练目标。保存canonical原F0 baseline_logits/closed；旧tail的col0是SSL，不得拿它充当原F0。原F0基准由旧tail的独立baseline_logits继承。

## 数据、校准与评分

复用VALIDATED_ONCE `20261101-phase3-ground-open-s2026092701-r03-source-clean-v1` / `sourceV-half-y-rx-day-s2026110101`，不新增导出或重验链。新训练只读合法G(clean) L6300及OE6134，L可读六类y与公共RX/day，OE只允许outside-six成员监督及公共RX/day，不读OE TX。源Us新增读取0，原F0/旧控制的Us祖先不清零，不称全系统无标签。

六已知TX14-10/14-7/20-15/20-19/6-15/8-20；源RX index1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23，split392005/eq1/256点。原历史V27000每y×RX×day格300按固定seed2026110101分CAL150/TEST150，共CAL13500/TEST13500，物理互斥。CAL只在模型冻结后保护接收率和原F0身份；TEST标签只在独立CPU评分侧。

源Q16650=knownTEST13500+NV3150，未知19-15/19-16/7-1各1050，public角色隐藏。跨RX仍为开发RX3-2/index34未来03-15/23 Q10238（5525/4713），known4508六类1491/1182/150/406/1186/93，未注册5730四类5-5/10-7/2-19/7-11=1454/1439/1427/1410。03-23缺20-19/8-20，完整已知指标N/A，不能present-only晋级。未注册不等于从未在OE或历史出现，不新增OE真TX去证明互斥。

四条件source_clean_DURA/source_star_DURA/cross_clean_DURA/cross_star_DURA全部固定DURA v1.0、FP32/TF32off/reuse+graph/nativeoff。源G(clean)/G10P1复用，跨RX两条件逐记录经同G。G10MIX01、600km/25MHz/2.45GHz、post_sync/noEQ/refSNR30dB/reflect、输入RMS一次/输出无RMS、源scene3001/channel4001/future3133/3134/receiver2027保持，不重抽观测、不用clean参考恢复star，目标clean只冻结测试。

四方法用相同CAL known24组规则；旧DMEAN_energy六阈值继承，新三方法18值冻结，合计24个FP64/192B，主6/48B。rank/学习率/损失和主方法不由CAL/NV/target选。零尾仍不满足已知保护时记录失败，继续合法预测，不借测试修复。源NV、TEST、目标U/Q均不参与训练或早停。

4方法×4条件共16点完整固定后，独立CPU连接source_knownTEST/sourceNV/crossclean/crossstar四真值各一次，旧DMEAN_energy四个overall与旧JSON完全一致。CPU重建raw/margin/指标，不加载NN、源训练数组或拟合新状态。主方法与三控制的配对净数和宏变化全部报告；对照原能量与中心化证据的两类输出不混列。

formal各条件CK95/RU95/原F0损≤2pp且完整名单；源clean all24组，跨clean overall+clean，源/跨star overall+六主景。`ground_source_success`、`cross_clean_success`、`ground_stage_success`与`frozen_rejection_success`分开，scientific只在四条件联合通过为True。通过也只完成冻结拒识子阶段；完整Phase3 ACTIVE，目标UDA/发现/注册仍需后续独立验证。历史开发数据已曝光，本轮不是新盲测；正式RX24-18不访问，不按Q结果改参数/主方法/扩seed。

## 成本、执行与产物

新训练2220×256=568320学生前向；新学生CAL13500+两源Q33300+两跨Q20476=67276预测；旧DMEAN只新增CAL13500前向。新真实NN前向总649096（新学生635596+旧CAL13500），原F0真实新前向0，G20476；旧源data G99284/F036450、原F0/G/旧DMEAN训练分别继承。合成smoke单列：train student6/F02/G2；predict student2/legacy2/F02/G2，r03/PCD真实及合成0。主部署仍499944神经参数+6阈值48B，无辅助统计模型/源行bank；全stage墙钟不冒充B1时延。

默认N607，启动free≥4096MiB，每卡至多新增2个本任务worker，不停止他人。单owner、单不可覆盖root，train首真实CP无query smoke PASS即继续→固定2220步→predict首smoke→CAL→四Q→CPUscore→同row全日志分析/报告/Git。具体commit、命令、环境/CWD、远端路径、GPU与期望产物写在同run登记。训练trace全步记录CE/KD/centered域风险/原能量域风险、原能量/中心化证据/共同均值、索引与梯度；不拿实际损失下降当测试成功。

停止只限输入/祖先/协议、非有限、输出冲突或无法合法预测；低分不是技术失败。不覆盖历史输出，不因低分重跑。一次独立P0/P1、聚焦负测、一次预检/发布包SHA比较/远端compile、实际PID/CWD/argv/GPU/log增长。无需另加审批或签名链。


code `1ae125733a0f25eb9c887351eb63f11f730d8cea`；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261101-phase3-ground-centered-s2026092701-r01-release`；command `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.ground_centered_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261101-phase3-ground-centered-s2026092701-r01-release/runtime.json`。


# Phase3 地面中心化证据训练结果（2026-11-01）

**本轮执行 VERIFIED，科学结论为 NO_PROMOTION。** 固定主方法 `CPAIR_centered` 的源 clean 拒识率由旧 DMEAN energy 的 69.523810% 提高到 73.968254%，同时保持源已知保护，但仍未达到 95%。跨 RX clean/star 的拒识率提高，却伴随 A_K、C_K 和 H 下降，所有正式联合门槛均未通过。原方法保留，完整目标 ACTIVE。

Run 为 `20261101-phase3-ground-centered-s2026092701-r01`，实际代码为 `1ae125733a0f25eb9c887351eb63f11f730d8cea`。主方法、训练预算、四条件和门槛按[冻结计划](../../../docs/PAMF_PHASE3_GROUND_CENTERED_PLAN_20261101.md)执行；[完整评分](../../../local_artifacts/pamf_phase3_ground_centered_20261101_r01/score/results.json)与[全产物分析](../../../local_artifacts/pamf_phase3_ground_centered_20261101_r01/analysis.json)保存全部分组和配对，未用结果更换主方法。

## 方法与比较含义

CPAIR 从完整原 F0 原生 reader 和默认六类头独立初始化，232993 参数、232945 可训练，默认头 960 参数随骨干联合更新；Sinc 的 48 参数及全部 buffers 冻结。原 F0、DURA、旧 DMEAN 和历史权重保持，无投影、另训头、EMA、PL、源 Us 或目标 U 新训练。

原能量为 `a(z)=logsumexp(z)`；中心化证据为 `c(z)=logsumexp(z)−mean(z)−log(6)=KL(Uniform6‖softmax(z))`。给同一记录的六 logits 加同一个常数，分类、CE、教师 KL 和 c 不变，a 会变化。这是代数性质，不证明旧模型曾利用共同偏移捷径，也不保证 c 的未知拒识效果。

训练目标为 `CE_L+4·KL(teacher_T2‖student_T2)+mean_d mean_(L,OE同域) softplus(c(OE)−c(L))`。15 个公共 RX/day 域等权；L192/OE64、同输入原 F0 缓存教师、AdamW lr/wd=1e−4、eps=1e−8、普通 norm clip=5，固定 2220 步后冻结。旧 DMEAN 的配对排序使用 a，其余源采样、初始化及优化预算匹配；CPAIR 不继承旧 DMEAN 学生权重。

四读出为旧 DMEAN energy/centered、新 CPAIR energy/centered，唯一主方法是最后一项。同一模型的两个读出共用 logits/closed，差异包含评分公式及各自 CAL 阈值；相同读出跨模型比较训练目标变化。不能将训练效应与读出效应合并归因，更不能事后把源分数更高的 CPAIR energy 改为主方法。

## 数据、标签与冻结顺序

复用 ground-open r03 的 VALIDATED_ONCE `source-clean-v1`、`sourceV-half-y-rx-day-s2026110101`。新训练只读合法 G(clean) L6300 和 OE6134；L 有六类真实 y，OE 仅提供 outside-six 成员监督及公共 RX/day，不读其真实 TX。L 缓存教师也是同一 G(clean) 的原 F0 输出，没有复用旧 G10 教师。

源六类为 14-10、14-7、20-15、20-19、6-15、8-20；源 RX 索引 1/3/4/6/8 对应 1-19/18-2/19-2/2-19/3-19，三日为 2021_03_08/15/23。原 Us56700 只保留 F0 等祖先成本，本轮 Us 值、IDs 和真值新读取为 0；全流程有 L/CAL 真标签与 OE 成员监督，不能称无标签学习。

历史 V27000 按 y×RX×day 每格 300 固定分为 CAL150/TEST150，合计各 13500；known TEST 每类 2250。构建时曾合法读取 V 已知 y 做分层分割和私有 TEST 真值导出，训练、校准、预测不读取 TEST 标签。CAL 与 TEST 物理互斥，但历史 V 不是新的独立盲测。

源查询每条件 16650＝known TEST13500＋NV3150；源未知 19-15/19-16/7-1 各 1050，public 不暴露 known/unknown 角色。跨 RX 是开发接收机 3-2（索引 34），未来 03-15/23 共 10238（5525/4713）：已知 4508，六类容量依次 1491/1182/150/406/1186/93；未注册 5-5/10-7/2-19/7-11 共 5730，依次 1454/1439/1427/1410。

四条件全部接入冻结 DURA v1.0：源 clean 为 G(clean)，源 star 为原固定 G10 P1；跨 clean/star 使用同一物理查询和各自既定输入经 G，目标 clean 只测试。G10-MIX01、输入 RMS 一次、输出无 RMS、物理与信道种子均保持，不新增信道观测。源未知字符串与目标标签 6 至 9 分开解释；“未注册”不自动代表从未在 OE 或历史中出现。

模型先于 CAL 冻结；旧 DMEAN energy 六阈值精确继承，另三个读出用 CAL 已知 24 组规则拟合 18 个新值，共 24 个 FP64/192 B，主方法只需六值/48 B。阈值先于所有查询消费冻结；16 点全部固定后独立 CPU 连接 source TEST、source NV、cross clean、cross star 四真值各一次，U/Us 真值读取为 0。

## 全部 2220 步训练证据

本地完整读取新 CPAIR 的 2220 步以及旧 DMEAN 的 2220 步 JSONL，逐步独立核对 L/OE 索引和 15 域配对数，全部一致。首步 CE=0.018558674、KD≈1.494e−9、原能量域风险=0.131502032，两模型对应共享输入量及 15 域原能量风险逐值相同；CE/KD、能量均值及原能量风险从第 2 步开始分歧，符合更新目标不同的设计，非新增 bit-exact 验收条件。

| 区间均值 | 总损失 | CE | KD | 中心化域风险 | 原能量域风险诊断 | 裁剪前梯度范数 |
|---|---:|---:|---:|---:|---:|---:|
| 第 1 至 100 步 | 0.073043 | 0.017471 | 0.010222 | 0.045350 | 0.052148 | 0.724270 |
| 第 1 至 740 步 | 0.050886 | 0.013197 | 0.008604 | 0.029085 | 0.033931 | 0.436346 |
| 第 741 至 1480 步 | 0.037741 | 0.009451 | 0.007983 | 0.020307 | 0.024386 | 0.268617 |
| 第 1481 至 2220 步 | 0.034503 | 0.008781 | 0.007769 | 0.017953 | 0.022023 | 0.211410 |
| 第 2121 至 2220 步 | 0.033174 | 0.008674 | 0.007497 | 0.017003 | 0.020993 | 0.194307 |

单步总损失首/末为 0.141632378/0.041208595，中心化域风险为 0.123073697/0.020788798。每步均有 15 个非空真实公共域，各域配对数范围 48 至 105；逐域中心化风险和原能量风险已全量分析，不以两个不同学生的绝对损失高低排名测试性能。

首/末 100 步，L 的平均原能量为 7.383317→7.712892、中心化证据 4.841239→4.786374、logit 均值 0.750319→1.134759；OE 为 3.605808→3.312405、0.963303→0.238556、0.850746→1.282089。共同均值确实变化，但这些新轨迹不能证明旧模型的捷径或解释唯一因果。已存 FP32 均值满足 `LSE≈c+mean(logits)+log6`，L/OE 全步最大绝对残差为 1.528e−6/4.551e−7；总损失分解最大残差为 1.118e−8，作为数值描述保留。

全部记录有限，head/浅层/中层/body 每步梯度非零，Sinc 每步为 0，无未参与梯度的可训练参数。裁剪前范数 0.083507 至 2.833585，普通 clip 调用 2220 次、实际超过阈值 5 的步数为 0，物理 clip/Sinc optimizer step 为 0；不能把“配置 clip5”写成发生了物理残差裁剪。实际唯一 L6300/OE6134，曝光 426240/142080，学生前向 568320，原模型与学生 buffers/Sinc 保持。

## 全部 16 个运行点

以下均为百分比、真实 TX 等权宏平均。C_K 为已知接收率，H 为 A_K 与 R_U 调和均值；学生闭集和原同 DURA F0 基准分列，不能用较低学生闭集替代保护基准。

| 方法 | 条件 | 原 F0 闭集 | 学生闭集 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|---:|---:|
| DMEAN energy | source clean | 98.414815 | 98.503704 | 97.725926 | 98.296296 | 69.523810 | 81.247108 |
| DMEAN energy | source star | 90.918519 | 90.925926 | 89.948148 | 98.000000 | 38.349206 | 53.772583 |
| DMEAN energy | cross clean | 48.567015 | 49.307831 | 36.784968 | 54.184457 | 57.038937 | 44.725819 |
| DMEAN energy | cross star | 41.721059 | 42.320882 | 35.634408 | 62.797839 | 41.782126 | 38.464169 |
| DMEAN centered | source clean | 98.414815 | 98.503704 | 97.977778 | 98.629630 | 62.666667 | 76.441370 |
| DMEAN centered | source star | 90.918519 | 90.925926 | 89.600000 | 96.777778 | 34.888889 | 50.222064 |
| DMEAN centered | cross clean | 48.567015 | 49.307831 | 37.545178 | 55.675683 | 37.469819 | 37.507461 |
| DMEAN centered | cross star | 41.721059 | 42.320882 | 31.920693 | 60.272003 | 35.006122 | 33.392286 |
| CPAIR energy | source clean | 98.414815 | 98.518519 | 97.918519 | 98.548148 | 78.380952 | 87.067156 |
| CPAIR energy | source star | 90.918519 | 90.940741 | 89.266667 | 96.192593 | 41.333333 | 56.503658 |
| CPAIR energy | cross clean | 48.567015 | 48.510471 | 34.865990 | 51.799510 | 66.694593 | 45.792825 |
| CPAIR energy | cross star | 41.721059 | 41.826343 | 32.161828 | 58.536072 | 49.520204 | 38.996589 |
| CPAIR centered | source clean | 98.414815 | 98.518519 | 97.970370 | 98.637037 | 73.968254 | 84.294000 |
| CPAIR centered | source star | 90.918519 | 90.940741 | 89.985185 | 97.511111 | 40.698413 | 56.047649 |
| CPAIR centered | cross clean | 48.567015 | 48.510471 | 35.113860 | 51.440298 | 59.511258 | 44.167342 |
| CPAIR centered | cross star | 41.721059 | 41.826343 | 31.731299 | 57.081174 | 45.573459 | 37.413093 |

旧 DMEAN energy 四个完整 overall 对象与[旧 tail 结果](../../../local_artifacts/pamf_phase3_ground_tail_20261101_r01/score/results.json)逐值一致，身份、原能量、margin、阈值及 canonical 原 F0 baseline 均精确复用。旧 tail 的第 0 列是 SSL，不能充当原 F0；本轮继承的是其独立 `baseline_logits/closed`。

## 配对结果与训练、读出效应

下表为主方法减各控制；宏变化单位为百分点，净数为相同 ID 的真实条数。完整增/失、TX/RX/day/scene 配对留在评分 JSON。

| 控制 | 条件 | ΔA_K | ΔR_U | ΔH | 正确接收净数 | 未知拒对净数 |
|---|---|---:|---:|---:|---:|---:|
| DMEAN energy | source clean | +0.244444 | +4.444444 | +3.046892 | +33 | +140 |
| DMEAN energy | source star | +0.037037 | +2.349206 | +2.275066 | +5 | +74 |
| DMEAN energy | cross clean | −1.671108 | +2.472322 | −0.558476 | −21 | +143 |
| DMEAN energy | cross star | −3.903108 | +3.791333 | −1.051075 | −38 | +218 |
| DMEAN centered | source clean | −0.007407 | +11.301587 | +7.852630 | −1 | +356 |
| DMEAN centered | source star | +0.385185 | +5.809524 | +5.825585 | +52 | +183 |
| DMEAN centered | cross clean | −2.431318 | +22.041439 | +6.659881 | −158 | +1263 |
| DMEAN centered | cross star | −0.189394 | +10.567337 | +4.020807 | −65 | +607 |
| CPAIR energy | source clean | +0.051852 | −4.412698 | −2.773155 | +7 | −139 |
| CPAIR energy | source star | +0.718519 | −0.634921 | −0.456009 | +97 | −20 |
| CPAIR energy | cross clean | +0.247870 | −7.183335 | −1.625483 | +55 | −410 |
| CPAIR energy | cross star | −0.430529 | −3.946745 | −1.583495 | +30 | −226 |

同模型读出比较中，旧 DMEAN 从 energy 改 centered，源 clean R_U 降低 6.857143 个百分点；新 CPAIR 同样降低 4.412698 个百分点。相同 energy 读出跨训练比较，源 clean R_U 提高 8.857143 个百分点、C_K 提高 0.251852 个百分点；相同 centered 读出则 R_U 提高 11.301587 个百分点。源收益支持本次具体训练目标有局部效果，不支持“只改中心化读出便改善拒识”，也不自动推广到跨 RX。

主方法相对旧 DMEAN energy，源 clean 正确接收 50 增/17 失、未知拒对 379 增/239 失；跨 star 正确接收 55 增/93 失、未知拒对 434 增/216 失。跨 star 闭集纠错为 66 增/129 失，净少 63；开放接收损失不能全归为阈值，未知拒识增加也不能掩盖身份损伤。

两个 CPAIR 读出闭集完全相同。跨 star 主方法比 CPAIR energy 正确接收净增 30，却宏 A_K 低 0.430529 个百分点；未知拒对净减 226、错误接收净减 66。类容量不等时总条数与按 TX 等权的宏数可以方向不同，不能混用。所有方法、所有配对均保持原主列，不挑更有利的控制晋级。

## 源未知、场景与正式门槛

| 源未知 TX | DMEAN energy clean 拒对 | 主方法 clean 拒对 | DMEAN energy star 拒对 | 主方法 star 拒对 |
|---|---:|---:|---:|---:|
| 19-15（每条件 1050） | 916（87.238095%） | 795（75.714286%） | 503（47.904762%） | 535（50.952381%） |
| 19-16（每条件 1050） | 399（38.000000%） | 697（66.380952%） | 184（17.523810%） | 201（19.142857%） |
| 7-1（每条件 1050） | 875（83.333333%） | 838（79.809524%） | 521（49.619048%） | 546（52.000000%） |

源 clean 的总体收益主要伴随 19-16 改善及另外两类取舍，三类均未达到 95%；不能据该结果按 NV 身份定阈值。主方法源 clean AUROC/OSCR 为 98.730504%/97.799897%，较旧 energy 的 96.543193%/95.690408% 提高；该排序改善与当前 73.968254% 拒识率分开报告，不声称已满足高已知覆盖区的尾部风险目标。

| 六个主星地场景 | 源 C_K | 源 R_U | 源原 F0 损失 pp | 跨 C_K | 跨 R_U | 跨原 F0 损失 pp |
|---|---:|---:|---:|---:|---:|---:|
| practical_high | 97.552501 | 57.356927 | 0.834565 | 52.020581 | 58.257815 | 10.452850 |
| practical_mid | 98.589219 | 48.361369 | 0.604720 | 54.259454 | 50.227558 | 14.054477 |
| practical_low_suburban | 98.438580 | 46.881844 | 0.612928 | 55.134733 | 48.041241 | 11.775246 |
| practical_high_urban | 98.711055 | 48.869070 | 0.413338 | 62.708465 | 50.532126 | 9.202700 |
| practical_mid_urban | 97.755169 | 41.837607 | 0.811151 | 52.623283 | 43.594460 | 14.174231 |
| practical_low_urban | 96.174320 | 29.823384 | 1.410159 | 59.721311 | 40.512968 | 6.911455 |

四读出 CAL 已知保护均通过，主共同尾比例 5/376。主方法源 clean 的 known-24 全过，但这些组的 R_U 为 43.809524% 至 94.761905%，没有达到正式 95%；源 star 的 known-12-scene 通过，known-24/42 未全过。CAL 和局部 known 通过不能替代源未知、RX×场景或跨 RX 联合保护，见[校准记录](../../../local_artifacts/pamf_phase3_ground_centered_20261101_r01/predict/calibration/complete.json)。

源 clean 五 RX 的主 R_U 为 85.555556/67.460317/75.238095/83.650794/57.936508%，三日为 78.857143/76.285714/66.761905%。跨 clean/star 的 03-15 A_K 为 35.531173%/32.316978%，R_U 为 58.205842%/46.637217%；03-23 缺 20-19、8-20，完整六类 A_K/C_K/H 及各日×场景对应指标为 N/A，R_U 仍为 60.833982%/44.548522%，不以 present-only 数值认证。

主跨 clean/star 的 20-19 仅正确接收 1/406、9/406；20-15 为 23/150、23/150。四目标未知主 R_U 依 5-5/10-7/2-19/7-11 为 61.623109/56.497568/59.215137/60.709220% 和 49.312242/51.077137/36.159776/45.744681%。逐 TX/RX/day、24/42 组和全部日×场景均保留在完整 JSON。

主方法源 star、跨 clean、跨 star 的 AUROC/OSCR 分别为 85.456087/79.747431%、56.022855/34.388922%、51.877800/26.740907%。Wilson 95% 区间是按真实类计算的已曝光池描述，不是独立 RX 泛化置信度；例如源 clean 19-16 拒识区间为 63.468449% 至 69.174032%。曲线和 CI 未用于扫描目标阈值。

正式要求始终为 C_K≥95%、R_U≥95%、原同 DURA F0 闭集−A_K≤2 个百分点及完整主景名单；源 clean 检查全部 24 组，两 star 检查总体与六主景。主方法四条件原 F0 损失为 0.444444/0.933333/13.453155/9.989759 个百分点。`ground_source_success`、`cross_clean_success`、`ground_stage_success`、`frozen_rejection_success`、`scientific_success` 均为 False；旧 90% 诊断不参与晋级。

## 成本与执行证据

新 CPAIR 训练前向 568320，预测 67276＝CAL13500＋两源查询33300＋两跨查询20476，学生合计 635596；旧 DMEAN 只新增 CAL13500 前向，总新增原生 NN 前向 649096。新 G 前向 20476，原 F0 真实前向及新数据导出为 0；教师 L6300 logits 来自旧合法缓存，不能把缓存读取计为新教师计算。

合成计数另列：训练 student6/F0 2/G2，预测 student2/旧 DMEAN2/F0 2/G2，r03/PCD 均为 0。首真实 CP 无查询 smoke 的初始原 F0 logits 最大差为 0，梯度到达默认头及浅中层，smoke 不做 optimizer step。旧源数据 G99284/F0 36450、原 F0 11100、DURA 4940 和 DMEAN 2220 训练成本继续继承，不计作新导出也不清零。

主部署为学生 232993＋G266951＝499944 神经参数及六个 FP64 阈值 48 B；默认头包含在学生内，wrapper 之外的 667729 参数、源教师、训练行库和另一个读出均不进入主部署。本轮没有密度拟合，18 新阈值与六旧阈值合计 192 B；不能把状态简单或批处理墙钟当作实测 B1 加速。

[流水线](../../../local_artifacts/pamf_phase3_ground_centered_20261101_r01/pipeline_state.json)实际训练/预测/独立 CPU 评分为 159.408028/25.959402/6.376850 s，整条 192.306549 s，三阶段 COMPLETE/return 0，没有真实技术失败。六次[独立观测](../../../local_artifacts/pamf_phase3_ground_centered_20261101_r01/observations.jsonl)捕获 driver、训练 GPU7（346/738 MiB）及预测 GPU1（512 MiB）；均为离散样本而非峰值。CPU scorer 未捕获独立 live，仅有 job、代码 CUDA 空设置、完整 stdout 和终态产物，未补跑遥测；终态 owned live/GPU 为空、foreign 操作为 0，私有 PID/设备标识不写入本文。

四份真实 stdout 全读：train2879 B/27 行、predict759 B/6 行、score45 B/1 行、driver21 B/1 行，无 warning/error。测试保留了初 RED、stable 公式 ULP 对齐和 test-only 缺旧 tail score 的定点修复；它们均发生在实测前，不是本次运行故障。最终验证 union15 unique/80 subtests（core3/11、train2/10、run3/12、score4/33、launch3/14），不把受影响重验重复计 unique；旧 `np.trapz` 与缓存权限警告只在测试日志。

一次独立 P0/P1 [审查 PASS](../../../local_artifacts/pamf_phase3_ground_centered_20261101_r01/review.json)，真实 smoke、preflight、单包 SHA 比较、compile/runtime 均通过，[发布验证](../../../local_artifacts/pamf_phase3_ground_centered_20261101_r01/release_verification.json)为 VERIFIED。包 919 文件、数据 0，SHA256 为 `da449474097e299d2925ce679382dd10508a6d17e8b657dcb207af055d969eff`。本报告只整理 JSON/JSONL/stdout，未读取数组、IQ、权重或原私有真值，未新增拟合、评分、实验、测试或审查。

这次具体中心化排序训练带来源端局部收益，但未消除未知类取舍，也未保持跨 RX 已知接收和身份性能；不能据此宣称普遍泛化改善或否定全部中心化方法。历史 NV/Q 已曝光、单模型 seed、单开发 RX、源与目标日期部分重合、全局物理历史 UNKNOWN，均不是新盲测。正式 RX24-18 未读，目标 U 新读取为 0，无目标 DA、发现或匿名注册；`adaptation_success=None`、A_N/H_KN/H_OW 为 N/A，完整目标 ACTIVE，不按本 Q 结果调参、换主或扩 seed。
