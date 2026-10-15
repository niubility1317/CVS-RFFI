# SIG_TIME＋RDrop：IQ 混合增强的下一轮设计

状态：已完成历史证据与实际增强路径审计、候选算子和损失原型，以及真实源IQ的冻结默认头验证。独立源run已ANALYZED/安全PASS；五模型混合增强训练尚未接入、发布或正式评分。本文是当前研发方向，不是额外设计审批。

用户最新预算覆盖旧安排：增强前底座为 SIG_TIME＋RDrop，每个新模型从头总计 E20，不加载 E50/E200 父权重，不追加 E20。旧的 HQ 默认头间隔 E50 设计停止推进，历史记录保留。当前工作树起点为 `482e428b85b6b1452bec748adfe92ba0acc32859`。

## 前面失败在哪里

失败是联合目标未达成，不代表每一机制都没有贡献。以下数值只在各自匹配实验内比较，不跨预算、数据契约或 seed 拼接收益。

|已完成实验|实际结果|支持与不支持的判断|
|---|---|---|
|普通 practical，E30、完整 L6300、3 seeds|相对无星地增强，low urban +15.3349 pp、urban 均值 +17.3588 pp；clean −1.6534 pp|直接训练受损 IQ 有效，但没有解决 clean 与星地的联合收益|
|课程＋单侧间隔，E30、同上|相对普通增强，low +0.6587 pp、urban +0.3932 pp、clean −0.1083 pp；low 3/3 正收益|是小幅稳定增量；一个 seed 的 clean 下降超过 0.5 pp，不能称为突破|
|Urban SSL，E30、同上|相对课程＋间隔，low +0.1696 pp；urban 2/3 seed 正、宏 F1 1/3 正|门控接纳增加或有监督来源正确，不足以保证困难观测的身份收益|
|单分支、全部分支及 joint 对齐，E50、5580/720|预登记源规则均未通过；没有扩 seed|特征更近、方差收缩、辅助项下降没有等价转化为默认头判别收益|
|跨 RX 语义方向与双向关系，E50、5580/720、1 seed|单向相对普通增强 low 仅 +0.2054 pp；相对 clean-only，clean 仍 −1.3667 pp；双向 urban 为负增量|不能强迫所有身份证据恢复成 clean，也不能凭目标分数选单向继续晋级|
|独立噪声 HQ 冻结输入干预，L_dev720、0 更新|冻结 EMA low 从 64.8611% 到 70.9722%|证明某些源 BLOCKED 观测可恢复；这是降低新增噪声后的输入效果，不是模型学到的正式收益|
|HQ 时序教学，E50、5580/720、1 seed|相对匹配 CE，正式 low +0.3012 pp、clean +0.0244 pp；相对普通增强 low +1.1488 pp、clean −0.6173 pp；最终 low 54.4202%|额外 HQ/前向和共有 joint 约束也是差异。BRIDGE_CE 不是完全无一致性对照，不能把整个差值归因于时序教学|
|HQ 冻结机制分析，独立源流|SIG 对 CE 净多对 4/720 条，均在 BLOCKED；默认头平均间隔/30 从 0.082212 降至 0.081173|只有小幅纠错，尚无更强身份区分证据；不能把源 BLOCKED 结果外推为正式目标的因果分层收益|
|排除 RX1 的交叉拟合教师，E50、源域小试|RX1 clean 144/144→102/144，low urban 56/144→32/144；六种 practical 均净负收益|本小试没有提供互补教师。clean 对照存在天花板，不据此否定所有交叉拟合教师，但本路线不扩库、学生或 seed|

既有正式 low urban 均未达到 60%。上面后期 E50 机制大多只有一个 seed；不能声称跨 seed 复现。完整结果、分层图、成本和负结果见[已完成证据汇总](../analysis/cvs_phase1_urban_evidence_20261015.md)。

最新 RDrop 完整矩阵已经是 18/18 行完成，不沿用较早的“剩余四行运行中”状态。其 clean 配对结果如下：

|预算|原 SIG_TIME %|匹配两次 CE %|SIG_TIME＋RDrop %|RDrop−原版 pp|正收益 seed|
|---|---:|---:|---:|---:|---:|
|E20|83.115675|82.486111|83.942857|+0.827183|2/3|
|E50，仅历史背景|82.922024|81.986111|84.340079|+1.418056|3/3|

E20 为13L＋7U、4440更新；E50为33L＋17U、11100更新，全部 scratch。RDrop 的两个随机前向复用同一份已生成 IQ；平均平滑 CE＋alpha=1 对称 KL，分类头随原网络联合训练。E20 均值尚未到 84%，且 seed01 比原版下降；RDrop 是随机子网络正则，尚无该矩阵的星地成绩，不能把 clean 增量加到旧 urban 分数上预测结果。

来源：[RDrop 完整匹配报告](E:/type10-7/automation_reports/CV-SincNet/20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01/report.md)。实际模型代码为 `b3ac442e2d33a5b315fb3251c30e0ad7083566bc`；技术恢复只补原未启动行，未重训健康行。

## 实际普通增强审计

已只读核实 E20_RDROP seed2026092701 的 actual_native_args、resolved_config、initialization、completion 和 source_training_mechanism，并在原依赖 release/原环境实际导入、打印模块 `__file__`、调用其 `build_aug_base_cfg` 和逐 epoch 配置。真实实现是依赖 release 的 `experiments/adv3b02_xuc/code/`，不是根 `code/`。完整只读结果为[existing_recipe_audit.json](E:/type10-7/local_artifacts/cvs_mixed_augmentation_design_20261015/existing_recipe_audit.json)，没有读取 IQ/truth、加载 checkpoint 或训练。

原始受核实模块路径以 `PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/` 为前缀，后接：

- `experiments/adv3b02_xuc/code/SSDG/train_ssdg.py`
- `experiments/adv3b02_xuc/code/DataAugmentation.py`
- `experiments/adv3b02_xuc/code/cvsrffi/schedule.py`
- `experiments/adv3b02_xuc/code/leo_practical/channel.py`、`residual.py`

这些是本次实际导入读回，不是根据仓库默认值推测。原 launcher 先放入依赖 release 的路径；只把薄 release 本身放进路径会导入失败。首次只读审计遇到该问题，按真实 launcher 路径修正后成功，没有重复提交训练或预测。

|普通 L 增强|E1 实际值|E13 至 E20 实际值|说明|
|---|---:|---:|---|
|总体 scale|0.1|0.2506003339|使用 min(epoch,13) 配置，E20并未达到配置上限0.35|
|循环移位概率／最大样本数|3.5%／3|8.7710%／8|批次级开关、逐样本 shift；可能引入接缝并改变固定参考窗口内容，尚不能断言已造成准确率损失|
|幅度缩放概率／范围|4.5%／0.9至1.1|11.2770%／0.9至1.1|范围本身不随 scale 缩小|
|相位旋转概率|4.5%|11.2770%|逐包均匀0至2π|
|CFO 概率／最大幅度|3.5%／1000 Hz|8.7710%／2506.0033 Hz|25 MHz；NoExplicitCFO 屏蔽身份坐标，不等于关闭此普通增强|
|相位噪声概率／每步 sigma 上限|3%／0.0006 rad|7.5180%／0.001503602 rad|Wiener 相位噪声|
|AWGN 概率／新增噪声 SNR 范围|4%／20至36 dB|10.0240%／20至36 dB|相对当前波形功率；不是原录音总 SNR|
|多径概率／最大延迟|1.8%／0|4.5108%／1|2至4 taps，原实现同批次共享核；短核不代表与 practical 可重复叠加|
|DC 概率／幅度上限|3%／0.002|7.5180%／0.005012007|在一般信道变换之前|
|端部时域 taper 概率|2.5%|6.2650%|代码名 bandedge_taper 实际乘时域窗，不能直接称频带滤波|
|PA 概率|1.4%|3.5084%|aug_enable_pa_normal=True，确实可进入 PA 模拟；不是仅有未启用的参数|
|DAC／class signature|关闭／关闭|关闭／关闭|不把普通 PA 模拟的有效性当作已验证事实|

实际顺序：native 数据 RMS → DC／时域 taper → 循环移位 → 增益 → 相位旋转 → CFO → 相位噪声 → 普通多径 → AWGN → 可选 DAC／PA → native 前向。RDrop 第二次前向直接复用第一次的增强后 IQ，不重抽普通增强。PA 内部又可包含非线性、记忆、AM/PM，不能把这整组称为单一标签保持 RX 扰动。

RMS 位于数据读取阶段，先于旧普通增益变化，所以不能声称 clean 路径的增益增强必然被该 RMS 消除。若再进入 practical，模拟器会重新单位 RMS；纯整体增益会被消除，但 PA／噪声／DC 与增益的交互不能等同于纯整体增益。新 A_g 明确恢复输入 RMS，只保留非平坦响应变化，并检验它仍改变波形。

旧 native 拼接分支在 concat_active 时先执行普通增强，再构造 satellite：存在普通 AWGN／多径／PA 与 practical 串联的路径。这里确认的是代码路径风险，不声称已经通过受控消融证明它是 clean 损失的唯一原因。

## 当前唯一假设与 IQ 算子

假设：**用有界的轻度线性观测响应训练身份稳定性，同时让完整 practical 单独承担星地退化，能比继续逼近 clean 特征更好地保留 clean 与 urban 的共同收益。** 尚无证据证明这项假设成立。

候选 A_g 对同一接收 IQ 的复数形式 z 使用：

`u[n] = z[n] + a·(z[n−1]+z[n+1])/2`，`|a|≤0.06`，随后把整包 RMS 恢复为输入 RMS。

- 内部 LTI 响应为 `H(ω)=1+a cos(ω)`，RMS 恢复前幅度界为0.94至1.06，最大峰谷比约1.044 dB；没有人为深衰落或新增噪声。
- 长度仍为256，没有整体数值移位；仅首末两个点使用 reflect，未缩放的有效区为[1,255)，参考[80,160)不接触人工边界。RMS标量可把端部功率差传播到整包，不能宣称整网完全不受边界影响。
- a按RX/day/eq及虚拟session epoch固定，所有TX共享规则；下一epoch更新虚拟session，RDrop/view内复用。它不是原真实RX硬件参数的估计或替换。
- 不合成PA、TX IQ、量化、剪裁、CFO或额外多径环境；不反演原录音未知TX效应。线性有界不自动证明身份保持，仍须真实源IQ/default-head验证。

A_s调用原完整 practical residual/post_sync。25 MHz、IQ256、reference_snr_db=30，质量适配、相位跟踪和IQ补偿按原引擎；EQ关闭时G=1，不冒称已经均衡。保留几何／遮挡／传播／新增噪声／接收状态／补偿依赖和原边界；如将来开启EQ，必须保留对应噪声着色。测试IQ及六场景配置保持原样。

首轮固定场景分布为80%原训练三场景均匀先验＋20%全部六场景均匀先验：high/mid/high_urban各1/30，low_suburban/mid_urban/low_urban各0.3。全部状态仍按引擎条件先验生成，不按loss挑最大损伤，不做新课程。这个静态分布是设计选择，不是性能结论；不能按目标成绩调整。

## 最小可归因训练矩阵

保留一条原配方参照 R_LEGACY。四因子组统一关闭旧普通 **L** 增强，原生 U弱／强视图、旧门控、ownEMA规则不改；不能把全局use_aug开关误用成关闭U机制。真实接入时须按调用计数验证。这样新 A_g 与 A_s 都从原 IQ 构造，不把既有PA等未验证变换继承成新的标签保持假设。

|row|原IQ槽|泛化槽|星地槽|L 前向／物理样本|
|---|---|---|---|---:|
|R_LEGACY|原普通增强后的一份IQ，原RDrop|无|无|2|
|B00|x|x|x|6|
|B10|x|A_g(x)|x|6|
|B01|x|x|A_s(x)|6|
|B11|x|A_g(x)|A_s(x)|6|

四因子组对**同一个物理样本**生成三个槽，每槽的RDrop两次随机前向使用完全相同的IQ。记 R(v)=两次平滑CE的均值＋alpha1对称KL：

`L_L = mean_i[0.25 R(x_i) + 0.25 R(g_i) + 0.50 R(s_i)]`。

原始IQ的显式监督权重始终0.25，星地槽固定0.50，g槽0.25，总尺度1。B01因此保留原普通clean:satellite=1:1的监督份额；B11把其中一半clean份额换成轻度A_g，并保留未改IQ。所有有标签satellite参加CE及其自身RDrop，不需要学生先高置信；不使用跨视图KL、伪标签或特征对齐。U总损失按原规则另计。

接入native时该式替换原L身份目标；若沿用现有hook，修正量必须是`L_L−native_identity_loss`，不能在原CE之上再完整叠加该式。六次随机前向的顺序在四组一致，算子使用独立命名流，不消费随机子网络的全局随机流；实际计数必须通过日志核对。

这里的B11是**混合不同IQ视图**，没有调用A_s(A_g(x))。组合／级联视图若有源证据支持，另作独立后续消融，不连续施加两次完整星地信道。A_g是否可交换也不预设。

四因子组B00至B11的物理样本、更新、六次前向和随机子网络预算相同；B00的重复原IQ是必要成本控制。R_LEGACY保留原实际两次前向，既是用户底座参照，也防止只击败重组后的弱B00就宣布突破。B00−R_LEGACY同时包含普通增强移除与前向数变化，不把它单独归因为某一算子。

固定首seed2026092701、full L6300/U56700/V27000、源RX1/3/4/6/8与day1/2/3；所有模型scratch E20=13L＋7U、4440更新，原SIG_TIME/RAW_HZ/joint160/NoExplicitCFO和默认cosine6×160/scale30，原参数联合训练。AdamW与cosine沿核实底座；不另拟合头。完整L的历史模型不伪装成5580/720内折控制，本轮也不继承这些权重。

每行L物理曝光预计568320；R_LEGACY预计1136640个L前向，四因子组各3409920个。首seed五行共100 model-epoch、22200更新、14776320个L前向；U、V、源probe、smoke和七视图预测另记实际计数与墙钟。E20不代表同样前向成本，更不代表收敛。

## 怎样证伪，怎样判断双突破

先做0更新的合法源IQ/default-head检查，验证有界响应的实际身份保留、RMS、参考窗口、边界和session一致性。若固定a策略让源默认头正确率下降超过1 pp，或出现非法角色、非有限值／隐式重复信道，不发布这项候选训练；保留负结果。源正确率安全不等于已证明跨RX泛化。

真实训练接入后，原V仅用于无更新健康及机制诊断，不拆新角色，不更新EMA／BN／归一化。源资格在读取新目标成绩前固定：B10 clean相对B00至少+0.5 pp；B11 clean不低于B00及R_LEGACY；B11 low相对B01至少+1 pp、urban均值非负；clean健康下降不超过0.5 pp、最差源RX下降不超过1 pp，实际梯度和预算完整有限。这是首轮可证伪规则，未通过不扩seed，不按更有利目标分数补救。

固定矩阵每行训练结束后自动原clean＋六种practical预测，全部prediction固定后独立truth-last评分；源FAIL也不删除或选择性漏报已登记的固定对照行。所有目标数值只作冻结后的验证，不回流候选选择、幅度、采样、系数、种子或重跑。

正式探索目标：B11 clean相对**B00和R_LEGACY都至少+0.5 pp**，low urban严格大于60%；每个suburban场景（high、mid、low_suburban）距本模型clean不超过2 pp，且相对B01绝对准确率下降不超过0.5 pp。不能靠降低clean缩小差距。这些是探索目标，不是保证或已实现结果。

报告固定比较：B10−B00的clean、B11−B00及B11−R_LEGACY的clean、B11−B10的clean、B11−B01的low/mid/urban，逐suburban绝对值与差距，以及每个指标的交互项B11−B10−B01＋B00。交互为负时如实报告竞争，不能用总成绩掩盖。

若A_g不改善clean，或联合后吃掉clean增量，或B11相对B01没有urban净收益，则放弃本固定混合假设；不追加对齐损失救场。只有预登记源规则通过才扩后两seed，最终按逐seed配对差、最差RX、宏F1、父参照固定clean正确集合保持率和状态／质量区间报告复现，单seed不宣布目标完成。

## 本轮已经交付与未完成项

已实现 `experiments/cvs_mixed_iq20/augmentation.py`、`objective.py`、`design.py` 和 `train_*.py` 完整管线。代码中的设计状态记录提交时的阶段；实际训练状态以登记run和运行产物为准。另一个独立冻结源验证run已实际完成，不与新训练混为一谈。

本地检查覆盖零扰动／零IQ、RMS保留但波形变化、人工边界有效区、频响界、无TX／全局RNG依赖、非法输入拒绝、三视图RDrop归一化与梯度，以及E20/数据角色/前向预算。它们不替代真实checkpoint smoke或源IQ实验。

实际执行：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_mixed_iq20.checks`，8项PASS；`F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_mixed_iq20.design --output E:/type10-7/local_artifacts/cvs_mixed_augmentation_design_20261015/plan.json`，成功生成五行未发布设计。Torch环境保留既有NumPy ABI警告，本检查没有使用Torch/NumPy桥接，不改环境。

### 已完成的真实源IQ验证

独立run `20261015-phase1-sigtime-rdrop-mixedIQ-source-s2026092701-r01` 已ANALYZED。实际代码为 `7025e13daf9f743438ba2ca7e9871c37201a9fcf`，冻结原E20_RDROP seed01默认头，完整V27000，原父模型fullL6300/U56700/V27000。一次限定P0/P1审查PASS，真实checkpoint无query smoke2后立即继续；单归档比较、远端编译、运行中PID/CWD/argv及终点产物均独立核实，CPU2、CUDA隐藏。

|源输入|正确/27000|准确率 %|相对RAW pp|修复／新增错误|默认头间隔/30|
|---|---:|---:|---:|---:|---:|
|RAW|26425|97.870370|0|0／0|0.216999769|
|G_E1_REFLECT|26409|97.811111|−0.059259|25／41|0.215260550|
|G_E20_REFLECT|26392|97.748148|−0.122222|36／69|0.212779462|
|G_E1_EDGE|26411|97.818519|−0.051852|26／40|0.215271831|
|G_E20_EDGE|26391|97.744444|−0.125926|35／69|0.212773472|

两reflect输入通过预定1pp安全条件，但它们让冻结模型多错16/33条，平均间隔略降；没有产生训练增益。源RX3下降较大，为0.240741/0.388889pp。这里的全部RX已在原训练源集合中，不能称为未见RX泛化验证。

实际135002个前向、74.915770秒、0参数/EMA/头拟合更新，模型state与Torch随机状态保持不变。功率相对误差最大4.768372e-7，恢复RMS后仍有非零波形变化；边界未缩放内部区最大差4.768372e-7。完整135000条源决定在本地独立核对了相同物理ID、混淆矩阵、纠错与间隔。两张实际源分层/响应图已检查，见[完整源验证报告](E:/type10-7/automation_reports/CV-SincNet/20261015-phase1-sigtime-rdrop-mixedIQ-source-s2026092701-r01/report.md)。

第一次本地取回遇到Windows Path到Linux路径的描述转换问题，已用POSIX路径直接取回原完成产物；未重启、重测或修改远端release。原实测结果与冻结策略未变。

### 五组E20已启动，正式结果待完成

训练run `20261015-phase1-sigtime-rdrop-mixedIQ-e20-manysig-s2026092701-m5-r01`，实际代码 `1fe3f00bdddfdb0af494f74c6570db65490ddf72`。本地五个真实native网络检查验证198个同初始state张量、同IQ RDrop对、默认头／融合／SIG有限非零梯度及2／6次前向，共3328个合成前向、10个非法输入／顺序负例；另13项协议负测和774个observer／EMA合成前向检查PASS，均无真实IQ或优化更新。首次本地导入路径问题修正后，完整检查通过。

一次限定P0/P1审查发现最终source panel JSON可能被并行dispatcher提前读取，已改为同目录临时文件完整关闭后原子发布；两项针对性检查与唯一问题定向复核PASS。没有全量重复审查或额外批准。

单归档传输比对、远端编译与真实checkpoint无query smoke通过后，五个scratch模型已各自完成至少首轮。独立PID／CWD／argv／GPU及日志增长核验VERIFIED；星地组每轮222更新、28416物理L星地包、170496个L前向，与登记一致。实际导入仍为原adv3b02_xuc practical，residual/post_sync/noEQ(G1)、25MHz、referenceSNR30、reflect和质量／相位／IQ补偿开启。原测试IQ未改。

训练和正式评分尚未完成，源资格尚未判定，不宣称clean／urban收益或跨seed复现。管线会逐row冻结E20后自动原clean＋六practical预测，七视图预测固定后独立CPU truth-last评分。具体命令、矩阵、来源、预算与停止规则见[训练登记报告](E:/type10-7/automation_reports/CV-SincNet/20261015-phase1-sigtime-rdrop-mixedIQ-e20-manysig-s2026092701-m5-r01/report.md)。旧健康任务未改。
