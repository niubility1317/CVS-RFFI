# PAMF 固定开放表征：12组与42组已知风险校准

Run `20261030-phase3-star-macro12-risk-s2026092701-r01`。唯一主为`signal_known12`。本轮不训练网络、聚合统计或分数归一化，不改任何历史阈值/判定；只检验预先固定的已知风险分组策略。完整跨未见RX星地/零目标标签适应/未知发现/未来匿名注册目标保持。

## 假设与固定矩阵

source V中旧known42规则同时保护overall、5个RX、6个scene与30个RX×scene，每组按真实已知TX宏平均约束C_K≥90%及原F0 closed−A_K≤2pp。signal-open的下一共同尾率先被208条的RX3/practical_high交叉组绑定，总体损伤仅0.592593pp。此为已有V风险轨迹事实，不是由未知分数选阈值的理由，也不证明放宽后会达到拒识目标。

新known12规则继续对overall＋全部5RX＋全部6scene分别要求相同C_K90/原F0总身份损失2pp与六类完整覆盖；30个RX×scene完整计算、明确作为诊断而非硬约束。known12通过不能写成known42通过，任何交叉组损伤都逐项报告，不隐藏在均值里。正式总体、接收机与预登记主场景验收不因本研究改变。

固定四列2×2矩阵：`frozen_known42`（原冻结前端控制）、`frozen_known12`（相同冻结前端仅新校准）、`signal_known42`（原信号主）、`signal_known12`（唯一新主）。原known42两列模型/阈值/raw/margin逐值复用。两个新known12各独立只用同一known V建立六阈值；不比较后挑表征或主分数。密度比不在本轮新增矩阵，不按先前较高对照更换主。

## 输入与模型身份

父预测为`20261030-phase3-star-signal-open-s2026092701-r03/rows/E_SIGNAL/predict`，实际执行code`dc01a402582f93e452dbaa3ef59a9fbb44151818`。其frozen_known_support列0与signal_known_support列1，以及各自known-only8字段聚合状态，为本轮唯一两个固定原分数；源known和novel特征/原F0默认身份及metadata原样复用。新run不加载比这些更好的其它候选，不读取新的IQ或改变表示。

主信号分支334721参数（headless信号232033＋MLP102688），原识别器232993，原DURA266951身份均保持。第一步严格加载父真实signal checkpoint，对两条合成P1形状IQ做无query smoke，state不变则立即进入缓存校准；不运行G或原F0，不执行真实IQ/新神经更新。父train config保持原r03/原code/原root，候选run/code与父模型来源分别记录，不把父状态改写为新训练。

复用known V27000（旧六设备各4500）、source novel3150（19-15/19-16/7-1各1050）；本轮source L6300/OE6134训练或新拟合读取0，原Us56700/target U/Q/正式RX24-18读取0。L/OE只作为原模型和密度祖先，不重新导出source逐样本bank。V标签只用于固定经验风险阈值，novel trueTX/K/未知配额不进入任何校准。

源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23均已见，novel有历史开发曝光，不能称盲测。四原source角色物理互斥，共42584条；本轮复用V/NV30150个既有记录，没有新增物理数据。known V每TX×RX×day300，novel70。真OE TX/K仍隐藏。

全部输入原固定G10MIX01/P1，P0是叠加星地信道后的received，P1为同received经DURA：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、郊区/城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001、原一次输入RMS/单received不变。无P0/clean实值、二次RMS、信道/G/真实F0或signal前向。本轮只做两条synthetic signal smoke，原源训练和全部真实forward成本不抹去。

## 12组共同尾率规则

与原known42保持同一common-tail算法，不新增OE代理目标或阈值搜索超参数。按冻结原F0预测类c建立六条raw score的完整tie-prefix曲线。q候选为各预测类prefix比例的精确Fraction并集（含0和1）；对每类拒掉不超过floor(q*n_c)的最大完整tie前缀。接受仍为raw≥tau[closed]，真实y不参与部署路由。

对q二分，选择全部12硬组完整、六预测类完整、各组精确宏FRR≤1/10且原F0 closed−open≤1/50的最大q。所有风险以Fraction计算，不能因浮点误差越过2pp。q=0不合法或硬组缺已知类时，保持原算法的接受全部fallback并明确科学失败；诊断交叉组缺类不作为新硬约束，但必须显示缺项，不用0补齐。部署仅六FP64阈值，不导出V曲线/逐样本状态。

所有42组均计算selected状态及下一q失败证据；hard_names12与diagnostic_names30显式保存。新阈值相对同原分数known42逐项不减，已知接受集合应为旧接受集合的子集，这是输入一致性可核查性质，不使用novel真值。新阈值由V独立确定，不能从未知拒识或H/AUROC选q、换组或调限额。

## 冻结顺序、独立评分与推进

actual checkpoint synthetic smoke后，只读父complete/model/threshold/calibration元数据，不先打开父novel NPZ。先冻结两个known-only聚合统计，复用V的raw/defaulthead/ID/metadata分别完成两个known12校准。两套新τ及两套历史τ完整固定后，才打开父novel预测、复制其原raw和特征并生成四列margin，全部本row预测固定后独立CUDAempty CPU scorer才唯一连接novel truth。

scorer从两份固定known8状态和保存frozen learned_features/signal_features重建四列raw，检查old42控制/τ/pred逐值复用、new12 state和分组、全部margin与predicted-class路由。不得重载NN、重新校准/拟合或遍历novel阈值。报告四列宏A_K/R_U/H、C_K、同P1原F0损失、AUROC/OSCR、逐TX/RX/day/scene及全部30交叉组；主相对signal42、主相对frozen12作配对，另列哪些交叉组超2pp/接收90。

唯一主signal_known12必须source known12和整体宏R_U≥90%才另预登记跨未见RX星地。source只是早筛，不认证最终主景全部通过，不因任何控制更高替换主。完整最终总体/主场景C_K90/R_U90/A_K原闭集损失2pp、后续零标签DA/未知发现/未来匿名注册继续，A_N/H_OW本轮N/A。

N607单个短GPU预测worker只用于synthetic checkpoint smoke，启动free≥4096MiB且每卡新增本任务≤2；随后一个独立CPU scorer，其他owner不停止/修改/重启。不可覆盖独立root、一次实际代码commit、聚焦Fraction/覆盖/分组与truth-last负测、一次直接P0/P1审查、最小登记/单preflight/archive比较/compile、PID/CWD/argv/GPU/log增长及终态证据。输出碰撞/非法来源或角色/非有限/不能产生合法预测保留不自动重启；UNKNOWN先核原root，不按科学低分停止健康运行。

## 实际发布预登记

Actualcode `26fbecff1df7e5476bcc5f459d8367c6fa49fd36`，唯一owner `codex-phase3-macro12-risk-20261030`，四列单row矩阵在experiment.json。环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-macro12-risk-s2026092701-r01-release`，独占root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-macro12-risk-s2026092701-r01`，预测predict、独立score、日志logs。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_macro12_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-macro12-risk-s2026092701-r01-release/runtime.json`。父真实模型/r03 trainconfig保留原code和路径，与本候选校准code分开。

Focused验证：{"core_run": {"unique_tests": 7, "subtests": 17, "seconds": 6.61, "stdout": "local_artifacts/pamf_phase3_star_macro12_risk_20261030_r01/core_run_tests.log"}, "score": {"unique_tests": 4, "subtests": 10, "initial_passed": 3, "targeted_fixture_scope_passed": 1, "targeted_seconds": 4.36, "production_changed_for_fixture": false, "legacy_trapz_warnings_initial": 864, "legacy_trapz_warnings_targeted": 1296}, "launcher": {"tests": 7, "seconds": 2.94}, "AST_files": 8, "mirror_bytewise_matches": 8, "unique_tests_total": 18, "subtests_total": 27, "real_data_weights_truth_read_in_local_tests": 0}；一次直接P0/P1审查PASS。单次N607preflight只核父预测/CP完成和来源元数据、文件存在、实际free显存，IQ/private truth实值0；没有新增NN/几何或源数据验证链。

预期actualparentCP synthetic_signal2 smoke、两个固定known8模型/四条件τ/两新known12全42诊断及next-q、四列known/novel预测、独立CPUtruthlast、PID/CWD/argv/GPU/log-growth/终态。rawIQ/权重/原始日志/逐行预测/source银行不入Git。当前无新性能；source历史曝光不改盲测身份，完整跨RX星地/无标签DA/发现/未来REG目标ACTIVE。


## 已完成实际结果

# PAMF 固定分数的known12／known42校准：完整结果

**执行VERIFIED，科学结论NO_PROMOTION。** 唯一主signal_known12：宏A_K89.948148%、C_K97.522222%、R_U46.888889%、H61.643672%，距同P1原F0闭集90.981481%损失1.033333pp。保留的overall＋5RX＋6scene共12组均通过；原42保护不再全部通过，5个交叉组损伤超过2pp。未知拒识仍低于90%，不晋级正式目标、不换控制为主、不改历史判定。

Run `20261030-phase3-star-macro12-risk-s2026092701-r01`，actualcode `26fbecff1df7e5476bcc5f459d8367c6fa49fd36`；固定设计 `docs/PAMF_PHASE3_MACRO12_RISK_PLAN_20261030.md`。原F0/default6head、独立signal与两个known聚合状态均冻结，没有新增训练、真实IQ前向或原分数拟合。完整跨未见RX星地、零目标标签DA、发现和未来匿名注册目标继续ACTIVE。

## 数据和来源

| 角色 | 数量 | 本轮使用 |
| --- | --- | --- |
| known V | 27000，旧六TX各4500 | 只用已冻结raw score/defaultpred/labels确定两套known12六阈值 |
| source novel | 3150，19-15/19-16/7-1各1050 | 全部新旧τ固定后复用父预测，独立CPUtruth-last |
| source L/OE | 6300/6134祖先角色 | 本轮拟合/真实IQ新读取0，原已训练表征和已拟known8仅复用 |
| 原Us | 56700祖先记录 | 本轮新读取0 |
| target U/Q、正式RX24-18 | 本轮0 | 无跨未见RX、适应、发现或注册新结果 |

四source原角色42584条物理互斥；本轮只复用30150条V/NV输出，不增加物理记录。source RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23均已见。V每TX×RX×day300，novel70；source novel已有历史开发曝光，本轮不是独立盲测、新RX或未来注册测试。

父为signal-open r03，actualcode dc01a402582f93e452dbaa3ef59a9fbb44151818；只取其frozen_known_support(col0)、signal_known_support(col1)及两份known-only8聚合状态。V/NV原F0默认身份、opaqueID、public metadata、frozen learned_features和signal_features逐值复用；不选择其它表征、密度比或更好的候选。原F0 Tri-SIG-TIME SSL/sourcecontract392005、DURA v1.0及独立signal的合法L/OE祖先保持，父2220更新不改写为本轮新增。

所有父预测均来自固定G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、郊区/城区各0.5及原六面积权重，receiver2027/scene2026103001/channel2026104001，原一次RMS/单received不变。P0是星地received，不是clean；P1是同P0经DURA。本轮不新读P0/clean/P1实值，不重抽信道、RMS/EQ、G或真实F/signal前向。首步两行合成checkpoint smoke的actual metadata为G10_P0_interface_only，不能冒充真实P1系统测试。

## 固定校准规则与实际冻结顺序

原known42同时保护overall、5RX、6scene和30RX×scene，均按已知真实TX宏平均施加C_K≥90%与原F0 closed−open≤2pp。新known12保留前12组相同限额/覆盖要求，30交叉组全部计算并明确为诊断。总体、RX与主场景保护并未删除；known12通过不能写成known42通过。

两套新阈值仍按冻结预测类路由：完整tie-prefix/Fraction q并集，对每类拒绝不超过floor(q*n_c)的最大完整前缀，精确FRR≤1/10/损失≤1/50，二分选全部硬组可行的最大q。真实y只用于已知V风险，不路由部署阈值；无未知指标、OE代理目标、V分数/几何/归一化拟合或τ超参数搜索。部署只需要六FP64阈值。

严格加载父真实signal checkpoint并执行2行无query合成smoke，state不变后立即复制冻结known8。先读V完成两个新known12；两新τ与两旧τ全部固定后才读父novel NPZ，输出四列margin。独立CPU scorer先验证models、τ、raw重建、metadata、控制和接受子集后，才唯一连接novel trueTX，不重新校准、fit、NN推理或遍历未知τ。

| 方法 | q | q% | 六预测类拒绝前缀 | 已知拒绝总数 | known12 | known42 |
| --- | --- | --- | --- | --- | --- | --- |
| frozen_known42 | 56/4241 | 1.320443 | [56, 56, 59, 65, 59, 60] | 355 | True | True |
| frozen_known12 | 97/4506 | 2.152685 | [91, 91, 96, 106, 97, 98] | 579 | True | False |
| signal_known42 | 73/4506 | 1.620062 | [68, 68, 72, 80, 73, 74] | 435 | True | True |
| signal_known12 | 56/2253 | 2.485575 | [105, 105, 111, 123, 112, 113] | 669 | True | False |

两新12下一q均被RX3整体（5400条）绑定：原F0损失2.018519pp，多误拒一个原本正确样本；主下一点C_K仍94.148148%，不是接收90约束。当前主RX3已使用108/5400=2pp正确拒绝预算。总体损失1.033333pp，距全局2pp还有0.966667pp、相当于261个正确样本；这不等于仍存在同时符合RX3保护的新q。

## 四路宏指标与配对

A_K为误拒计错的旧六TX宏正确接收，C_K为宏接收，R_U为三未知TX宏拒识，H=2A_KR_U/(A_K+R_U)。原同P1闭集24565/27000=90.981481%与所有身份输出保持。

| 方法 | A_K% | C_K% | R_U% | H% | 原F0损失pp | AUROC% | OSCR% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_known42 | 90.496296 | 98.685185 | 23.015873 | 36.698290 | 0.485185 | 89.584163 | 83.137141 |
| frozen_known12 | 90.044444 | 97.855556 | 40.095238 | 55.484282 | 0.937037 | 91.670685 | 84.872965 |
| signal_known42 | 90.388889 | 98.388889 | 31.301587 | 46.500199 | 0.592593 | 93.115866 | 86.509364 |
| signal_known12 | 89.948148 | 97.522222 | 46.888889 | 61.643672 | 1.033333 | 94.493082 | 87.505278 |

| 主−参考 | 正确gain | 正确loss | 正确net | 接收net | 未知拒对gain | 未知拒对loss | 未知拒对net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| signal_known42 | 0 | 119 | -119 | -234 | 491 | 0 | 491 |
| frozen_known12 | 118 | 144 | -26 | -90 | 432 | 218 | 214 |

主正确接收24286、接受26331、已知拒绝669；较原闭集少279个正确，接受中2045身份错误。未知拒对1477/3150=46.888889%，距90%的2835条仍少1358条。主较同signal42多拒对491、少正确接收119、少接受234；新增拒对loss0、正确gain0，符合更高τ的接受子集。

相同分数的校准scope效应：冻结前端未知净+538/正确−122/接受−224；信号分支+491/−119/−234。相同scope的信号−冻结效应：42组未知+261/正确−29/接受−80，12组+214/−26/−90。两因子有交互，不能把不同参照的增益直接相加。

本轮raw score与表征完全不变，AUROC/OSCR变动来自六预测类阈值平移改变了跨预测类margin排序，不是新学习了特征。Wilson逐类区间仅描述重复源池样本，不构成独立目标泛化或盲测保证。

## 必须显式报告的交叉组代价

| 方法 | 交叉组 | known n | C_K% | 原F0损失pp |
| --- | --- | --- | --- | --- |
| frozen_known12 | rxscene/3/practical_high | 208 | 94.377924 | 2.788807 |
| frozen_known12 | rxscene/3/practical_mid | 407 | 93.237836 | 2.817405 |
| frozen_known12 | rxscene/3/practical_low_suburban | 2007 | 95.109427 | 2.143623 |
| frozen_known12 | rxscene/3/practical_mid_urban | 388 | 94.686046 | 2.585498 |
| frozen_known12 | rxscene/4/practical_high_urban | 238 | 97.023810 | 2.579365 |
| signal_known12 | rxscene/3/practical_high | 208 | 94.526008 | 3.698924 |
| signal_known12 | rxscene/3/practical_low_suburban | 2007 | 95.221129 | 2.030089 |
| signal_known12 | rxscene/3/practical_mid_urban | 388 | 94.564787 | 3.294993 |
| signal_known12 | rxscene/4/practical_high_urban | 238 | 97.528860 | 2.074315 |
| signal_known12 | rxscene/8/practical_low_urban | 2074 | 95.050250 | 2.411531 |

两新12各有5个交叉组超原2pp，接收率均仍高于90%。这些是实际局部身份风险代价，不按known12通过删去、合并或改写历史known42保护。原42两列仍42通过，新12两列只12通过。

## 设备、日期与主方案完整分组

| 未知TX | n | 拒对 | R_U% |
| --- | --- | --- | --- |
| 19-15 | 1050 | 641 | 61.047619 |
| 19-16 | 1050 | 369 | 35.142857 |
| 7-1 | 1050 | 467 | 44.476190 |

| 旧TX | n | 原闭集% | A_K% | C_K% | 额外失去正确 |
| --- | --- | --- | --- | --- | --- |
| 14-10 | 4500 | 89.044444 | 87.755556 | 97.000000 | 58 |
| 14-7 | 4500 | 80.333333 | 79.288889 | 97.022222 | 47 |
| 20-15 | 4500 | 95.911111 | 94.933333 | 98.066667 | 44 |
| 20-19 | 4500 | 85.466667 | 84.311111 | 97.200000 | 52 |
| 6-15 | 4500 | 96.333333 | 95.466667 | 97.288889 | 39 |
| 8-20 | 4500 | 98.800000 | 97.933333 | 98.555556 | 39 |

以下均为已见sourceRX/day，同MIX scene切片而非新观测。全部required TX覆盖完整，无N/A或缺类填0；五RX、六scene、三day的R_U全部低于90。

| 组 | known n | novel n | A_K% | C_K% | R_U% | H% | 原F0损失pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 27000 | 3150 | 89.948148 | 97.522222 | 46.888889 | 61.643672 | 1.033333 |
| rx/1 | 5400 | 630 | 92.703704 | 99.259259 | 47.777778 | 63.057093 | 0.388889 |
| rx/3 | 5400 | 630 | 83.148148 | 94.166667 | 45.714286 | 58.994046 | 2.000000 |
| rx/4 | 5400 | 630 | 90.925926 | 98.277778 | 45.238095 | 60.417071 | 0.703704 |
| rx/6 | 5400 | 630 | 92.388889 | 99.240741 | 53.015873 | 67.371626 | 0.444444 |
| rx/8 | 5400 | 630 | 90.574074 | 96.666667 | 42.698413 | 58.037023 | 1.629630 |
| scene/practical_high | 1103 | 133 | 96.818816 | 97.972470 | 67.256496 | 79.374456 | 1.094063 |
| scene/practical_mid | 1983 | 256 | 96.991898 | 98.512758 | 56.262915 | 71.215341 | 0.450765 |
| scene/practical_low_suburban | 10304 | 1203 | 94.691081 | 98.046361 | 53.314635 | 68.219263 | 0.913837 |
| scene/practical_high_urban | 1171 | 130 | 95.953300 | 97.819516 | 61.088815 | 74.650973 | 0.999102 |
| scene/practical_mid_urban | 1979 | 236 | 90.607329 | 97.611245 | 49.102564 | 63.689866 | 1.106501 |
| scene/practical_low_urban | 10460 | 1192 | 82.378804 | 96.714192 | 33.883336 | 48.016812 | 1.243755 |
| rxscene/1/practical_high | 202 | 20 | 98.569157 | 99.549550 | 60.317460 | 74.838792 | 0.000000 |
| rxscene/1/practical_mid | 437 | 46 | 97.860429 | 99.291201 | 60.975299 | 75.135097 | 0.708799 |
| rxscene/1/practical_low_suburban | 2042 | 245 | 97.016033 | 99.410839 | 57.573529 | 72.263034 | 0.243039 |
| rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 99.645390 | 55.714286 | 71.377214 | 0.000000 |
| rxscene/1/practical_mid_urban | 401 | 44 | 93.472511 | 99.298281 | 49.989226 | 65.140972 | 0.473409 |
| rxscene/1/practical_low_urban | 2082 | 253 | 86.037884 | 99.042714 | 34.172103 | 48.915993 | 0.526354 |
| rxscene/3/practical_high | 208 | 25 | 90.104019 | 94.526008 | 62.962963 | 74.127234 | 3.698924 |
| rxscene/3/practical_mid | 407 | 54 | 92.829171 | 94.768528 | 44.407895 | 60.076307 | 1.342269 |
| rxscene/3/practical_low_suburban | 2007 | 239 | 89.865719 | 95.221129 | 53.134703 | 66.782856 | 2.030089 |
| rxscene/3/practical_high_urban | 241 | 29 | 90.253555 | 93.356962 | 72.685185 | 80.522242 | 1.575530 |
| rxscene/3/practical_mid_urban | 388 | 44 | 84.840722 | 94.564787 | 38.888889 | 53.331800 | 3.294993 |
| rxscene/3/practical_low_urban | 2149 | 239 | 73.183702 | 93.098091 | 34.253785 | 46.665626 | 1.695603 |
| rxscene/4/practical_high | 213 | 26 | 96.744493 | 96.744493 | 77.777778 | 86.230504 | 1.103989 |
| rxscene/4/practical_mid | 403 | 48 | 98.149588 | 99.726776 | 57.037037 | 72.147476 | 0.000000 |
| rxscene/4/practical_low_suburban | 2049 | 248 | 95.427648 | 98.538679 | 50.831180 | 66.330354 | 0.626925 |
| rxscene/4/practical_high_urban | 238 | 30 | 96.206425 | 97.528860 | 65.555556 | 77.977107 | 2.074315 |
| rxscene/4/practical_mid_urban | 392 | 40 | 92.506614 | 99.146825 | 51.481481 | 66.149601 | 0.000000 |
| rxscene/4/practical_low_urban | 2105 | 238 | 83.701025 | 97.808440 | 28.872265 | 42.934486 | 0.844873 |
| rxscene/6/practical_high | 240 | 24 | 99.593496 | 99.593496 | 73.333333 | 84.469519 | 0.406504 |
| rxscene/6/practical_mid | 365 | 47 | 98.543783 | 100.000000 | 73.333333 | 84.089660 | 0.000000 |
| rxscene/6/practical_low_suburban | 2104 | 240 | 96.453703 | 99.418387 | 57.341147 | 71.923942 | 0.291814 |
| rxscene/6/practical_high_urban | 233 | 27 | 98.375307 | 99.583333 | 53.981481 | 69.710643 | 0.416667 |
| rxscene/6/practical_mid_urban | 408 | 57 | 92.722570 | 99.474548 | 59.459729 | 72.455981 | 0.238095 |
| rxscene/6/practical_low_urban | 2050 | 235 | 85.388066 | 98.809624 | 40.468348 | 54.912004 | 0.704372 |
| rxscene/8/practical_high | 240 | 38 | 98.016317 | 99.164578 | 54.151404 | 69.761460 | 0.396825 |
| rxscene/8/practical_mid | 371 | 61 | 97.272834 | 98.609640 | 54.761905 | 70.074060 | 0.260417 |
| rxscene/8/practical_low_suburban | 2102 | 231 | 94.569666 | 97.554191 | 46.913434 | 62.715444 | 1.399369 |
| rxscene/8/practical_high_urban | 223 | 22 | 95.484380 | 98.757702 | 56.161616 | 70.724678 | 0.814947 |
| rxscene/8/practical_mid_urban | 390 | 51 | 89.671391 | 95.607339 | 50.374957 | 64.509964 | 1.490099 |
| rxscene/8/practical_low_urban | 2074 | 227 | 83.978221 | 95.050250 | 30.988925 | 45.271974 | 2.411531 |
| day/1 | 9000 | 1050 | 90.077778 | 97.844444 | 53.714286 | 67.298060 | 0.955556 |
| day/2 | 9000 | 1050 | 89.955556 | 97.744444 | 45.047619 | 60.032419 | 0.944444 |
| day/3 | 9000 | 1050 | 89.811111 | 96.977778 | 41.904762 | 57.145933 | 1.200000 |

## 一致性、资源与边界

18份结构JSON、3份完整stdout、全部V/NV四列、两known8、4×42校准/评分组完整读取。旧42模型/τ/raw/margin、全部publicmetadata、frozen learned_features与signal_features逐值父一致；两新τ逐项非减，known/NV接受集合是旧接受集合子集；四raw由两known8与两z重建最大误差3.41e−13。没有重新执行校准、评分或未知τ扫描。

新NN/native/signal/projection/density/normalizer更新0，L/OE拟合读取0，真实IQ/F0/signal/G前向0；实际父CP合成signal2。两known8各322868B、合计645736B，四套六τ192B。父成本包括原F011100、独立signal新增2220/特征累计13320、MLP2220，以及原冻结前端训练，均保留；r03已有610904真实signal与10合成signal/F0合成2不能计成本轮新增，也不被“本轮0”抹去。

Pipeline15.533032s；predict7.145719s（producer内部4.605729）、独立CPUscore8.302036s，非B1。actualcode `26fbecff1df7e5476bcc5f459d8367c6fa49fd36`，543文件/data0 archive SHA18591ad1dae476d9751beb51121df909301b0da653dda08c40ed56b699e17b28单次比较/remotecompile/runtime PASS；18项focused测试/27子项与一次直接P0/P1审查PASS。3份完整stdout共5行，warning/error/nonfinite0。

独立live仅捕获driver881735和CPUscore881974/CUDAempty、日志增长；predict881830配置GPU4只由jobs/smoke/完整产物记录，未捕获其live GPU占用，不补造显存或峰值。3次观察覆盖启动、评分live和终态；全部stage COMPLETE、live/GPU空、foreign停止/修改/重启0。GPU短stage未取到live测量不授权重跑同实验。

## 研究判断与后续

30个交叉guard确实压低旧运行点，但去掉它们仍只得到46.888889%未知拒识；不能把“约束保守”当成全部失败原因，也没有90可达证明。当前12组下一点被源RX3整体身份保护绑定，不能仅因C_K97.52%仍高就继续抬高同一q。保持本轮阈值、模型和历史判定，不按NV trueTX调类阈值、组集合或预算。

下一步优先检验识别与开放表征是否应采用不同信号路径：身份继续同P1原F0，开放分支可单独研究同一固定星地P0 received，比较恢复前后对未知特征的影响。P0不是clean，不能由当前结果断言DURA损坏未知身份；需要匹配训练/冻结比较、合法源标签权限及独立truth-last。这一后续尚未实施，不计本轮性能或成功。

完整目标继续ACTIVE：跨未见RX、Phase1后全为星地观测；总体及预登记主场景宏C_K/R_U≥90%、A_K距同输入原闭集≤2pp，主报A_K/R_U/H并继续提高接收/拒识；零目标TX标签DA、未知发现、未来独立匿名注册（新类≥80%、旧类损失≤2pp、后续未知保护）仍待完成。AN/H_OW=N/A，正式RX24-18新访问0。
