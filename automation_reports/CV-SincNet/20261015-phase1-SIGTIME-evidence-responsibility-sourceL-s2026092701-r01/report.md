# 真实证据责任：等损失值的梯度路由源原型

状态 **ANALYZED / VERIFIED，源资格 INITIAL3/4、FINAL20 0/4，支持从零 E20 验证**。下文保留原预登记。目标仍为提升实际 joint160 身份特征及默认头三种子 clean 均值至84%，当前未达成；原默认 cosine 头随主 CE 正常联合训练，U真值隔离、无星地训练增强。上一轮代码/结果执行与推送已核实，两个实际源候选关闭，是取得新证据的 progress，不是等待或技术阻塞。

## 源证据与方向选择

复用上一轮完整900L的 RX/targetTX/referenceTX 风险，不增加 IQ/CP前向。E20 held180原样本准确率100%，原CE .0105142；同RX三端差分 virtualCE .0645498、72/3600错，跨RX .0980904、112/3600错。跨RX相对同RX的CE差在目标RX1/4/6为+.05833/+.09045/+.17561，RX3/8为−.07362/−.08306。RX6的 factualCE仅.0007629，却有crossCE .18806/40错；同RX为.012447/0错。按originTX差有5正1负，refTX差六组都正，幅度.00967至.07030。这说明不能把所有参考差分视为共同、可靠、类无关的RX作用；非线性TX/RX交互及packetnoise是解释假设，尚无因果分离证据。完整条件表见[preceding_conditional_risks.json](preceding_conditional_risks.json)，这不是新seed或独立newdata确认。

三条可执行方向：一是显式学习类相关RX运输，但需要新的生成模型、身份保真对照，当前简单双差证据不足；二是**直接修复当前真实证据中损害身份判别的增量**，不依赖delta身份保持假设，作为本轮主攻；三是以真实L标签学习证据弱化后的主路径补偿，但历史COMP/可信U补偿有负结果与稀疏入口，暂不重开。每条都必须实际进入joint160，不以辅助embedding、另训头或目标成绩选择来源。本轮仅新增第二条核心假设，后续不得通过λ/容差/分支筛选扫描救活。

## 明确机制

真实四项 joint=((a+tanh(gPA)p)+tanh(gREF)r)+.2tanh(gSIG)s，来自同一次原模型前向。对每个分支 b，按原浮点加法树将该项置零，得到 z_without_b；没有跨包或跨TX重组。定义 h_b=relu(CE_W(z,y)−stopgrad(CE_W(z_without_b,y))−1e-6)，Laux=.1×sum(h_b)/720，覆盖4branch×180packet，固定分母，不按active量归一化。真实y只来自L，既有默认W.detach/scale30/eps1e-4，三gain在辅助重建中detach，主CE仍正常训练它们。达到不增加身份CE的条件即停止，不再压绝对p=.9。

ROUTED_GAP每个h_b只保留负责分支b的rawcomponent边，其余三项context.detach；虚拟full数值必须与实际joint逐项exact相同。JOINT_GAP控制保持全部component边，使用完全相同的full/ablation数值、h_b、active掩码、λ与720分母；两loss scalar严格相同，差异仅是梯度去向。ZERO连接四rawcomponent的零图，实际辅助为0。无生成器、拟合head/diagnostic分类器、原型、持久统计、推理路由或部署特征变换，新增参数0。路由在component张量边发生：若上游共享参数，仍可能收到不同路径的梯度，不能声称参数模块完全隔离。gain.detach限制直接缩小融合gain的逃避，主CE和共享上游仍会改变其他证据，因此不能省略独立held原身份检查。

这与历史随机COMP撤去路径训练、EVIDENCE_STOP只弱化PA/ref并训练a/s、全教师关系复制不同：当前以真实y测量actualcomponent加入是否增加损失，把修复交回损害的component，而非强化已经正确的全体特征。特征挑战已有[RSC](https://arxiv.org/abs/2007.02454)先例，其机制丢弃主导特征并训练剩余特征；本候选的增量是CVS实际四项证据的责任判定与同值梯度路由，不宣称通用理论首创或已有效。

## 固定矩阵与资格

INITIAL/FINAL20×四currentL180×ZERO/JOINT_GAP/ROUTED_GAP，共24单元。使用同一ownBASE_CE-s2026092701的原untrained initial_smoke和final_ssdg epoch20，source_proof在两个load前核对实际roles/physicalIDs、scratch ancestry、fixedfinalfreeze；初始scratch_only/finalactualargs及188参数严格核验，不按target最高选择父、不回退未知权重。当前与held仍为prior900L（ord2/3,19/20,37/38,54/55；held66/67），物理ID不交，父FINAL见过held，原L6300/U56700hidden/V27000与RX1/3/4/6/8/day1/2/3/6TX不变。U/V/target不读。VALIDATED_ONCE handles复用，不做数据重验/签名链。

INITIAL空原moments、FINAL保留savedm/v/step，两状态固定stageLR2e-4/原AdamW options/globalclip5。两realCP各2全零IQ noquery smoke，PASS立即继续；模型eval，48步AdamW只在临时副本，原live模型/CP/optimizer/grad/RNG不更新。after全188进入真实临时模型，保留after3gains，ACTUAL afterW与PRE_W beforeW分析同一afterjoint/aftercomponents，不重新拟合头。

每state至少2/4 ROUTED同时相对ZERO和JOINT满足：current≥36/720 active、≥18/180不同origin、nonzeroaux；abs(auxF/mainCE cosine)≤.95，abs(auxF/JOINT cosine)≤.95，防仅缩放；heldPRE_W positiveburden和fullCE均下降≥1e-5；原ACTUAL/PRE_W CE各下降≥1e-5、准确率不降，原full错数不增；actualF差norm>1e-6、heldjointRMS差>1e-5、至少1浅中层tensor改变。此新任务的1e-5幅度在本候选结果前固定：burden按4个ablation平均、原fullCE量级约.01，旧virtualCE约.1任务的1e-4门槛不移植，不重判旧loss。完整资格值见[唯一登记](experiment.json)。FINAL通过优先同父E20＋20，只有INITIAL通过scratch20；qualified仅说明值得正式训练，必须继续固定同路径纯CE/oldhiddenU SSL/JOINT/ROUTED及逐row自动clean预测/独立truth-last，累计40成本明确。失败不把control替为candidate、不扫描λ/容差/路由子集。

900uniqueL/6300metadata/6124网络packet前向（4smoke＋2×[180before＋4×(180current＋3×180after)]）/48临时CPUAdamW/60objectivecalls，objectiveFP64 97200、FP32比较54000判别。每call sharedfull180＋4routedfull720＋4ablation720=1620FP64，FP32只比较sharedfull180和4ablation720=900；路由full数值已经exact验证。native main/evaluator默认头前向随既有模型路径执行，不能把objective数字冒充全部head计算。0真实训练/头拟合/EMA/U/V/target。完整branch/RX/TX/day active/burden/去除前后错误、aux各层/实际gain、actualFi71/浅中层/truejoint/Sincwholeproposal核差保留；gap变小不替代原identity改善，也不称DG。

run `20261015-phase1-SIGTIME-evidence-responsibility-sourceL-s2026092701-r01`，一个owner codex/root/evidence-responsibility-20261015，actualcode/config Gitcommit写release_commit.txt。普通N607/CVS-RFFI Python/CPU2/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_sourceL_20261015_r01/experiments/cvs_evidence_responsibility20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-evidence-responsibility-sourceL-s2026092701-r01/source_diagnostic`、日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_sourceL_20261015_r01/source_probe.stdout.log`。权限/来源/非有限技术失败保存且只停止ownedassay，先核对再恢复；健康任务不触碰。一个focusednegativecheck/唯一P0P1review、一个preflight/归档SHA比较/remotecompile、PID/CWD/cmd/GPU/loggrowth和独立终态。REJECTED_EXTRA_GATE：brainstorming额外设计审批/固定plan/specreview由用户CVS最小白名单覆盖，已有明确研究授权直接推进，不新增审批。

历史SIG_TIME/CENTER25K/COMPACT_LINEAR/LONG_SILU正收益按各自同row与预算保留，SIG_TIME CUDA初始化混杂继续披露；不能相加、择target最高父或以source资格冒充84%三seed结果。Phase1/defaulthead/hiddenU/cleanonly测试范围保持。


## 真实源结果

执行 `34b8309871822c2756169f39183133d1a4af839a`，wall 17.14 s，24cells/16paired/120RX/96branch，所有CPsmoke、oncearchive/compile、PID/CWD/argv/GPU/loggrowth、live不变与独立自然终态均 VERIFIED，无技术失败。资格由完整24文件及summary逐项一致复算。

| state | ordinal | active/720 | distinct/180 | auxF/main cos | auxF/JOINT cos | ROUTED−ZERO CE | ROUTED−JOINT CE |
|---|---|---:|---:|---:|---:|---:|---:|
| INITIAL | 2 | 261 | 179 | 0.933353 | 0.948977 | -0.00176465 | -0.00125773 |
| INITIAL | 19 | 257 | 178 | 0.932302 | 0.949846 | -0.00261569 | -0.00202279 |
| INITIAL | 37 | 250 | 177 | 0.932412 | 0.951916 | -0.00116669 | -0.000695456 |
| INITIAL | 54 | 259 | 180 | 0.932854 | 0.948726 | -0.00189188 | -0.00147235 |
| FINAL20 | 2 | 130 | 117 | 0.407263 | 0.407877 | -7.97402e-07 | +1.05745e-06 |
| FINAL20 | 19 | 125 | 119 | 0.501953 | 0.672260 | -9.57696e-07 | +3.03901e-06 |
| FINAL20 | 37 | 131 | 120 | 0.676765 | 0.672858 | +5.01574e-07 | +5.59748e-06 |
| FINAL20 | 54 | 130 | 121 | 0.351874 | 0.521627 | -1.98932e-08 | -4.18356e-06 |

资格：{"passed": true, "stage_signal": "SCRATCH20_SIGNAL", "qualified_ordinals_by_state": {"INITIAL": [2, 19, 54], "FINAL20": []}, "cell_conditions": [{"state": "INITIAL", "ordinal": 2, "qualified": true, "failed_conditions": []}, {"state": "INITIAL", "ordinal": 19, "qualified": true, "failed_conditions": []}, {"state": "INITIAL", "ordinal": 37, "qualified": false, "failed_conditions": ["near_JOINT_gradient_rescaling"]}, {"state": "INITIAL", "ordinal": 54, "qualified": true, "failed_conditions": []}, {"state": "FINAL20", "ordinal": 2, "qualified": false, "failed_conditions": ["ZERO:no_held_positive_burden_difference_gain", "ZERO:no_held_full_CE_difference_gain", "ZERO:ACTUAL_no_original_identity_CE_gain", "ZERO:PRE_W_no_original_identity_CE_gain", "JOINT_GAP:no_held_positive_burden_difference_gain", "JOINT_GAP:no_held_full_CE_difference_gain", "JOINT_GAP:ACTUAL_no_original_identity_CE_gain", "JOINT_GAP:PRE_W_no_original_identity_CE_gain"]}, {"state": "FINAL20", "ordinal": 19, "qualified": false, "failed_conditions": ["ZERO:no_held_positive_burden_difference_gain", "ZERO:no_held_full_CE_difference_gain", "ZERO:ACTUAL_no_original_identity_CE_gain", "ZERO:PRE_W_no_original_identity_CE_gain", "JOINT_GAP:no_held_positive_burden_difference_gain", "JOINT_GAP:no_held_full_CE_difference_gain", "JOINT_GAP:ACTUAL_no_original_identity_CE_gain", "JOINT_GAP:PRE_W_no_original_identity_CE_gain"]}, {"state": "FINAL20", "ordinal": 37, "qualified": false, "failed_conditions": ["ZERO:no_held_positive_burden_difference_gain", "ZERO:no_held_full_CE_difference_gain", "ZERO:ACTUAL_no_original_identity_CE_gain", "ZERO:PRE_W_no_original_identity_CE_gain", "JOINT_GAP:no_held_positive_burden_difference_gain", "JOINT_GAP:no_held_full_CE_difference_gain", "JOINT_GAP:ACTUAL_no_original_identity_CE_gain", "JOINT_GAP:PRE_W_no_original_identity_CE_gain"]}, {"state": "FINAL20", "ordinal": 54, "qualified": false, "failed_conditions": ["ZERO:no_held_positive_burden_difference_gain", "ZERO:no_held_full_CE_difference_gain", "ZERO:ACTUAL_no_original_identity_CE_gain", "ZERO:PRE_W_no_original_identity_CE_gain", "JOINT_GAP:no_held_positive_burden_difference_gain", "JOINT_GAP:no_held_full_CE_difference_gain", "JOINT_GAP:ACTUAL_no_original_identity_CE_gain", "JOINT_GAP:PRE_W_no_original_identity_CE_gain"]}], "controls_not_fallback": true, "interpretation": "Actual component responsibility, distinct gradient and source identity benefit, not20epoch/unseenRX/DG"}

INITIAL250至261/720active，177至180/180不同origin。ROUTED与JOINT loss同为.03036至.03148，但auxnorm约.533至.547对JOINT1.125至1.183，F/main cos .93230至.93335对JOINT .97434至.97639。ROUTED/JOINT cos .94873至.95192；ordinal37超过固定.95，即使该组原accuracy多对1包也不例外。其余2/19/54全部条件通过，符合至少2/4的预登记scratch20资格。

INITIAL heldPRE_W CE相对ZERO四降.001167至.002616、相对JOINT四降.000695至.002023；ACTUAL CE也四组均优于两控制，positiveburden四降。三组accuracy维持共同主CE后的23.8889%，ordinal37到24.4444%；不能把共同从19.4444%的提升归给辅助。F区别约.00778至.00871，14至17浅中层tensor及实际joint改变；source rawgeometry margin只有微小混合变化，不能据此宣称泛化表征已经提升。等scalar/mask下的路由产生了真实方向和实际默认判别优势，仍可能部分与较小gradient强度相关，尚无完整训练效力证明。

FINAL20 current125至131/720active、117至121不同origin；ROUTED/main cos .35187至.67676，ROUTED/JOINT .40788至.67286，新的方向存在。held原accuracy皆100%，相对ZERO原CE三微降一微升，幅度最多约9.58e-7；相对JOINT三升一降，不能通过任何一组1e-5的实际identity与burden资格。固定FINAL0/4，故不从final20续训，也不通过提高λ或放宽门槛挽救晚期。该结果支持把机制放到特征形成阶段检验，而非已训练特征后期修整。

继续执行原授权的完整scratch20实验。新run `20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r01`，BASE_CE/BASE_SSL/JOINT_GAP/ROUTED_GAP四条件三seed，先首seed源健康/实际信号，再另两seed；所有4条件重新从零构造，不继承sourceassay权重。纯CE20L/0U真实关闭U/EMA，其余13L+7U保留原oldgate/ownEMA，默认头联合训练。新辅助日程在main真实结果前固定为epoch1至13、每16step、180calls/32400额外L180packet前向/model；进入pseudo后辅助实际关闭，原U任务继续。两baseline零额外bank/IQ，JOINT和ROUTED成本/标量相同；成本差明确披露。source当前证据给出scratch资格与晚期0/4，不据target或主训练结果改日程。每rowfinal20后自动clean预测、独立truth-last评分；测试分数不能影响扩seed或停止健康任务。

本source-only原型完整分析，并不等于完整main训练完成、clean新准确率或84%目标达成。完整[24单元](source_cells.csv)、[120RX](source_RX_metrics.csv)、[96branch](source_branch_metrics.csv)、[16paired](paired_regimes.csv)、[证据](result_evidence.json)保留；历史正结构/CENTER25K路线及全部相关负结果继续保留。0actualtrain/headfit/EMA/U/V/target。
