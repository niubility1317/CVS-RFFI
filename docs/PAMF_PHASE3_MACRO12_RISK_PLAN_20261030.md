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
