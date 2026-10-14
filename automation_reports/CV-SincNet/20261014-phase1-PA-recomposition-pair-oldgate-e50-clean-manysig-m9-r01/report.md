# 同身份跨 RX 的 PA 证据重组

本轮检验 CVS 的 PA 实际增量能否成为可跨接收机重组的身份证据。新目标由真实 L 身份标签决定，避免复制教师的全部关系/响应几何。特征插值已有 Manifold Mixup（https://proceedings.mlr.press/v97/verma19a.html）；待验证的机制在于 CVS 的可分离增量、同身份跨 RX 配对及梯度边界，不宣称混合特征本身新颖。

等 U 时序 R02 首次真实阶段测量已完成：MID/SPREAD 当时 ownEMA 在完整 V27000 准确率约96.8%，旧 gate 接受 coverage83.4%/precision99.9956%；真实同 AdamW 状态的一步 U 增量约为总更新范数的1.8%至1.9%，totalshadow误差1.49e−8。这里只是源 V 代理和一步测量，不解封 U、不给出实际 U 精度、不证明整体泛化改善。本 PA 配方 α=.2/λ=.1 已在上述数据及 U target 结果前固定，不选择胜出的日历，独立保留原tail33+17。

## 机制与固定对照

真实部署 `joint160=a+b+c`，其中 b 是已带实际 PA gain 的增量，c 是实际 reference projection 增量。只替换 b：`z̃_i=a_i+b_i+c_i+α(b_j−b_i)`，并对称训练 `z̃_j=a_j+b_j+c_j−α(b_j−b_i)`。同 TX、同 day、不同物理包，保持原a/c，α=.2固定。原现有 cosine W/scale30/normalize eps1e−4 计算真实 L CE；辅助 W 与 PA/ref gains detach，内部 features 有梯度。主 CE 正常联合训练原默认头和 gains，原 train/loss_tx_labeled 不包含辅助；不冻结、替换或重训头。

| 条件 | donor | 辅助系数 |
|---|---|---:|
| ZERO | 同成本计算全部诊断 | 0 |
| SAME_RX | 同 TX/day、同 RX | .1 |
| CROSS_RX（预定主条件） | 同 TX/day、跨 RX | .1 |

仅用每步已有 L128 前向。每 TX/day 至少2个合格 RX cell，每 cell 至少2个不同物理包；各合格 cell 私有随机选相同最小包数。共同 cohort 一对一配对，SAME cell内循环，CROSS cell间循环，每包各作一次anchor/donor，双向端点曝光完全匹配。剔除重复、不足和不平衡部分，记录实际coverage/cell counts；不为覆盖率改变身份/物理规则，不假设不同包同波形或物理硬件反事实。

## 原训练、预算与学习信号

全部新scratch50，33标签+17原pseudo阶段，L始终参与50轮；每轮222步，共11100真实AdamW/EMA更新，L1420800/U963900曝光。固定seed2026092701/02/03，共9模型/450模型轮次/99900更新。原默认CVS架构/844754参数/joint160/ResidualPhysicalHead6×160、NoExplicitCFO与内部补偿、PAIR、ownEMA=.999、旧confidence/temporal/strong gate、noise=.015/Dropout、λU=.16、entropy=.01、L smoothing=.01、AdamW lr=.0002/wd=.0001/clip5/cosine最低.000001均保留。不叠加R/E、TIME、Sinc、U日历、新伪标签链或训练信道增强。

前1110步warmup，新机制此后每4步，共2498calls/模型。新增参数0，额外L/U IQ前向0，额外实际优化更新0。所有conditions仍主 CE 正常训练完整L。在1112/3556/7328/10880四个真实步骤，记录aux/main-L特征梯度、方向、W/gain辅助梯度None；用真实总损失与总损失减去PAaux，复制同当前AdamW参数/moments/step/lr/decay，各自clip5模拟两步。原U/entropy留在base，真实仍更新1次，temporary total对照实际更新，记录各部署featuregroup数值更新与PAaux增量。8临时步骤/模型单列；一步增量不等于累计因果收益。Globalclip/moments可间接改变主W更新，不声称W训练轨迹不变。

三组191初始state和CPU RNG相同，真实 native构建→NoCFO→安装与预测builder一致，EMA在bound训练callback前deepcopy；原CUDA非确定性保留，不声称完整前缀逐位一致。数学、真实CVS与实际native L/coreloss→callback→probe→backward→clip→AdamW本地接线PASS，ZERO loss/gradient/update difference为0，active aux/Lnorm约.1168，actual-shadow误差0。样本是合成源形状128包，不能代替实际运行信号或泛化效果。

## 源资格与自动默认头测试

首seed三个row完整final50，final训练LCE≤.5、共同cohort平均coverage≥.25；两个active条件四点aux/Lfeature normratio均值≥1e−5；相对ZERO的源Vmean下降≤1pp、worstRX≤2pp。只据这些固定source-only条件放行另外6row，不微调α/λ/日历，不按target排序/停止/重跑。已启动健康row完成原预算。

每row固定final50后立即自动默认头clean168000包预测，所有prediction完成后freshCPU independenttruthlast评分。全9row1512000决定/162CM复算，报告CROSS−ZERO、SAME−ZERO、CROSS−SAME逐seed差/均值/SD/正差数、MacroF1/worstRX/day、源曲线/coverage/分支实际更新及负结果。只有CROSS稳定优于SAME与ZERO才支持跨RX机制贡献，不能用合成梯度或某个最高seed替代证据。固定主条件CROSS的三seedcleanmean≥84%是尚未达成的研究目标。

## 最小发布

- 数据：原VALIDATED_ONCE ManySig source contract/physical IDs/capsule/split复用，L6300/U56700隐藏TX/V27000，source RX1/3/4/6/8、day1/2/3、6TX，query clean168000。V仅原源健康/诊断，无梯度、阈值拟合或新头；Phase2不在范围。
- 初始化：9new scratch，无checkpoint/teacher/optimizer继承。首步no-query smoke仅严格读已核实同契约历史default50 student/ownEMA后零输入；不继承训练。数学与运行时observer/counterfactual实际import一起执行，不以compile代替依赖导入。
- run：`20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01`，唯一owner `codex/root/PA-recomposition-20261014`。
- 代码：不可变release的`release_commit.txt`，提交推送并独立核对remoteOID。
- 环境：N607普通账号、`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_PA_recomposition50_20261014_r01`；命令 `python -u -m experiments.cvs_pa_recomposition.dispatch`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01`，独立不可变root。
- GPU：每GPU最多3总实验进程，计入其他owner，至少12000MBfree。不得热改、停止、重启健康旧run。
- 技术停止：本run的role/physical/来源错误、输出碰撞、非有限active训练、固定预算错误、PAcapture或W/gain边界错误、totalshadow不匹配、非法预测或未处理CUDA/data/artifact错误。两row同prediction前错误等预定系统异常仅处置绑定runowned进程，保留所有outputs，无未经修复的自动重试；target低分不触发停止。
- 产物：preflight、实际source args/契约/初始化/完整step与50epoch日志、2498PA调用/cohort/CE/margins、4actual gradient/AdamW/parameterupdates、finalcheckpoint/freeze、W独立featureexport、逐row默认head prediction/score/completion、source-only扩展决定、queue/analysis/finalreport。

唯一独立P0/P1审查PASS，无阻断问题。当前未发布PA、无真实训练/性能证据；等U/source阶段验证完成后按固定方案执行，既有结构交互和正收益路线独立保留。84%目标active，不能用发布或source信号代替验收。

## 实际发布与首个训练信号

RUNNING/VERIFIED，执行commit `76ab0b9944210f8aa3f09f051ea9694f55457cd6`；一次archive传输比较、remotecompile、真实同契约student/ownEMA strict noquery和实际runtime依赖import PASS。3首seed PID/CWD/argv/GPU/参数和日志增长核实，当前轮数[16, 12, 13]。第1112步实际梯度/AdamW/参数更新已观测，cohort约55%，SAME/CROSS aux/Lfeature normratio约.1168/.1038，W/PArefgains辅助grad None，真实t1/PA_b1参数更新有限非零；active增量约4.25e−4/3.24e−4。ZERO梯度0，GPU临时分支差2.44e−8为浮点抵消残差，不能写成bitwise0；totalshadow误差≤1.49e−8。没有本轮完整性能结论，后6row仅按原source条件，84%active。

## 首种子源资格与实际特征证据

VERIFIED：三个条件均已按原固定配方训练至 final50，预定 source-only 健康条件通过，余下 6 行自动启动；未用目标分数排序或修改配方。

| 条件 | 最终 L CE | 源 V 准确率 | 最差源 RX | 同配对队列覆盖率 | 四点辅助/L 特征梯度比均值 |
|---|---:|---:|---:|---:|---:|
| ZERO | 0.287650 | 97.8926% | 94.4074% | 54.8170% | 0.000% |
| SAME_RX | 0.288041 | 97.8667% | 94.2593% | 54.8170% | 10.274% |
| CROSS_RX | 0.286805 | 97.9037% | 94.3704% | 54.8170% | 9.444% |

每模型 2,498 次辅助构造、4 个实际梯度点及 4 个实际 AdamW 更新观测齐全。ZERO 辅助梯度为 0；SAME_RX 与 CROSS_RX 的辅助特征梯度非零，W/PA/ref 增益的辅助梯度均为 None。全部实际 shadow 逐元素误差小于 1e−6，训练没有额外真实更新或 IQ 前向。ZERO 的差分增量约 4.3e−9 至 2.7e−8，来自浮点减法的微小残差，不声称 GPU 上逐位为零。

首三个冻结模型在自身同一合法 L90 上均有初始至最终实际 joint160 的函数变化；full/pure、W 置零/删除误差为 0，参数/buffer/梯度保持。增加 1,350 次源 L 前向、0 拟合/优化/EMA 更新。该见证只说明实际特征被训练改变，不证明 PA 重组的因果贡献；完整三种子默认头 clean 效果仍待返回。

## 固定语义间隔的源结果分析

后续只读分析先使用已保存、实际部署的 L90 特征，不启动新训练、不叠加 PA 损失。固定 margin=0.1；anchor 与 positive 同 TX、同 day、跨 RX，positive 与五个异 TX negative 来自同一参照 RX/day。每个源模型 90 个包产生 1,800 个确定关系。用真实可见 L 标签及原物理 ID 逐包匹配已存 tensor；不从预测反推类别。

记录 gap=cos(anchor,positive)−cos(anchor,negative)、违反排序/未满足固定间隔的比例、[0.1−gap]+ 及其局部特征导数。只对 detached 特征副本求导，0 IQ 前向/模型拟合/训练更新/U 真值/目标访问，CPU 运行；每行独立新 source_semantic_geometry 目录，已完成复用、不覆盖。源特征读取需原 source_proof 与源 witness 都已完成且 ID 对应；输入/输出异常保留产物，不干预健康训练。

本诊断不根据目标结果选择 margin 或排序条件；若基本已满足，不增大间隔制造梯度。90 个已见源包和 1,800 个相关关系不等于独立样本；非零 hinge 只表明可能存在监督空间，不能据此认定新损失有效。后续若独立训练，必须保留匹配原配方与普通 SupCon 对照，默认头正常联合训练；本次尚未发布该训练机制。

三元间隔本身已有 [FaceNet](https://arxiv.org/abs/1503.03832) 等研究；标签正负组织已有 [SupCon](https://papers.nips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) 基础。拟检验的问题是实际 CVS 特征的同参照 RX 身份关系和满足后停止的作用，不把通用间隔公式称为原创理论。代码 experiments/cvs_source_semantic_geometry.py；有限间隔停止、非零局部导数、状态/RNG 不变与五个非法输入本地检查已通过。这是现有源结果分析，不增加发布门槛、候选审批或重复 review。

## 固定源间隔诊断结果

VERIFIED：首种子三个冻结模型的合法 L90 与已存特征物理 ID 完整对应，CPU 分析完成；没有新增 IQ 前向、分类器拟合、训练更新或 U/target 访问。

| 条件 | 未满足 0.1 间隔的关系 | 直接排序错误 | hinge 均值 | 非零局部特征导数包数 |
|---|---:|---:|---:|---:|
| ZERO | 95/1800（5.2778%） | 56/1800 | 0.009905 | 54/90 |
| SAME_RX | 96/1800（5.3333%） | 55/1800 | 0.009622 | 54/90 |
| CROSS_RX | 98/1800（5.4444%） | 56/1800 | 0.009890 | 56/90 |

ZERO 的 94.7222% 关系已满足固定间隔，仍有 95 个未满足关系，其中 56 个正例相似度低于负例。RX3 的未满足率为 37/360=10.2778%，day1 为 62/600=10.3333%；其余分组与全部三个条件保留在 CSV/JSON。SAME_RX、CROSS_RX 的未满足数分别为 96、98，不据这个单种子的几何差异认定 PA 有收益或损害。

这只支持“少量源身份关系仍有待纠正”的机制问题，不能把非零 hinge 当作泛化收益。未满足的是关系，不能把约 5% 关系解释为仅约 5% 样本收到信号；同一关系的三个端点使 54 至 56 个包出现非零局部特征导数。该导数来自 detached 特征副本，尚不是新排序损失的真实骨干梯度或 AdamW 更新。

90 个已见训练包按每 cell 首包确定，1,800 个关系彼此相关；该结果不代表完整 L6300、早期训练状态或目标域。后续应在相同固定间隔下核实合法源学习空间，并独立比较原训练、普通 SupCon 与停止条件排序；不根据本诊断扫描 margin 或堆入现有 PA 训练。当前只有 PA 重组在训练，保持其固定 α/λ/日历，等待完整三种子默认头结果。


## 完整 L6300 固定间隔源分析

补充同 run 的冻结结果分析：选择全部三个预定义 ZERO 对照，各自严格读取 own final50，不依据目标表现选择模型。原可见 L6300/90 个 TX/RX/day 单元、每单元 70 个独立物理包全部参与；输入、标签、物理 ID 和模型来源沿用已核实原契约，U 标签隐藏、V 无梯度，0 目标访问。

固定 margin=.1 不变。anchor 与同 TX、同 day、其他 RX 的同单元序号 positive 配对；五个 negative 使用该参照 RX/day 中其余 TX 的相同序号。每包固定 20 个关系，共 126000/model，三模型 378000。序号依原 L dataset 顺序，不做 hard mining、置信筛选或 margin 扫描。每模型从默认 identity 前向取得真实 joint160，共额外 6300 IQ 前向、三模型18900；完整参数/buffer/gradient 保持、0 拟合/实际更新/EMA更新/额外U前向。

普通 N607 账号、原不可变 source release CWD/projectPython运行已提交分析代码。GPU仍按全机每卡最多3个总实验进程/至少12000MBfree的共同锁分配，保护所有健康训练。输出每ZEROrow/source_semantic_geometry_full_L6300，现有完成结果复用，partial保留不覆盖；未冻结来源、角色/样本数/单元数/有限值/只读状态错误停止该分析，不干预训练。产物为源features tensor/可见metadata、固定几何summary及同row报告，不上传IQ、模型权重或原始features。

代码 experiments/cvs_full_source_semantic_geometry.py 与既有固定间隔分析函数。L90兼容、L6300的126000关系/停止导数/分组总量与不均衡单元负例在本地核验。这不是新训练候选或发布门槛；只用完整源训练池判断可能的监督空间，不证明泛化收益、真实模型新损失梯度或独立样本显著性。

## PA 重组完整三种子结果

ANALYZED / VERIFIED：9 个 scratch50 模型自然完成，默认分类头 clean 预测后独立 truth-last 评分。1,512,000 个决定、162 个混淆矩阵、全部配对差和同物理包纠错/引错计数已复算。无缺项。

| 条件 | clean 均值 | SD（百分点） | Macro F1 | 最差 RX 均值 | 最差 day 均值 |
|---|---:|---:|---:|---:|---:|
| ZERO | 80.6710% | 2.2588 | 0.801581 | 68.4250% | 76.3143% |
| SAME_RX | 80.5823% | 2.5606 | 0.800669 | 68.0917% | 76.2230% |
| CROSS_RX | 80.6165% | 2.1851 | 0.800924 | 68.0403% | 76.3310% |

| 配对 | 均值差（百分点） | seed 01 / 02 / 03（百分点） | 正差数 |
|---|---:|---|---:|
| CROSS_RX − ZERO | -0.0546 | +0.0315 / -0.0179 / -0.1774 | 1/3 |
| SAME_RX − ZERO | -0.0887 | -0.0554 / -0.4060 / +0.1952 | 1/3 |
| CROSS_RX − SAME_RX | +0.0341 | +0.0869 / +0.3881 / -0.3726 | 2/3 |

**固定 PA 重组没有确认泛化收益。** CROSS − ZERO 为 −0.0546 个百分点，仅 1/3 正差；CROSS − SAME 为 +0.0341 个百分点、2/3 正差。CROSS 为 80.6165%，距 84% 仍有 3.3835 个百分点。本轮关闭固定 α=.2、λ=.1 比较，不据目标成绩扫描权重，也不向隐藏 U 延伸硬类别 donor 重组。

| seed | CROSS 比 ZERO 纠错 | 引错 | 净变化 |
|---|---:|---:|---:|
| 2026092701 | 650 | 597 | +53 |
| 2026092702 | 888 | 918 | -30 |
| 2026092703 | 562 | 860 | -298 |

每模型完整 50 轮、11,100 次更新、1,420,800 次 L 观察和 963,900 次隐藏 U 观察；共 450 模型轮次、99,900 次更新、12,787,200 次 L 观察及 8,675,100 次 U 观察。每模型 2,498 次辅助调用只复用现有 L 前向，无新增 IQ 前向、参数或实际优化步。共 22,482 次调用和 2,877,696 次辅助调用中的 L 观察；平衡 cohort 覆盖率另见逐 seed 表，不把全 batch 与实际 cohort 分母混用。0 非有限跳步、0 U 真值读取、0 分类头拟合。

36 个实际特征梯度点验证辅助 W/融合 gain 梯度均为 None，ZERO 辅助梯度为 0，两个重组臂对部署特征有非零梯度。36 个 AdamW 点各在当前参数、动量、lr、weight decay 和 clip 5 下临时计算 base/total 两步，每点仍只有 1 次真实更新，共 72 次临时步、0 次额外真实步；实际 total shadow 最大绝对误差 2.98e-08，低于 1e−6。ZERO 的 GPU 分解舍入可产生极小差，不称逐位等价。

辅助调用中条件 cohort 的部署 CE 三 seed 均值约 0.259，SAME 重组约 0.238 至 0.239，CROSS 约 0.231。固定重组反而降低条件源 CE；这个现象不足以证明制造了更难的身份辨识任务，也不能直接归因为消除接收机信息。这里的数值不是完整 L 或 V 损失。

9 个冻结模型均完成与自身 scratch 初始同 90 个合法 L 包的函数见证，额外 4,050 次 L 前向：实际 joint160 发生变化，full/pure、W 置零、W 移除差均为 0，参数、buffer、梯度不变。该证据说明部署特征确实学习且可独立导出，不证明 PA 的因果增益、未见 TX 迁移或 Phase2 成功。

完整 L6300 几何仅分析三个预定 ZERO 控制，额外 18,900 次 L 前向，固定 margin=.1、同 day、跨 RX、同 ordinal 正负参考，无硬挖掘或参数扫描。每模型 126,000 条相关关系，未满足比例分别为 5.1746%、4.9333%、6.1595%；共 20,497/378,000。未满足关系可以涉及许多端点，不能等同于约 5% 的样本；梯度只在 detached 特征副本上计算，未更新模型。这是已见源域几何诊断，不是泛化性能证明。

按照用户新报告，后续主线改为受控加性证据变化产生候选 TX 集合，并通过达到集合概率质量后停止的损失学习。先做真实 TX 包含率、集合大小、有效未满足数量与 RX/day 分布的源域可行性分析；排序原型保留为未发布备选。B0/B1/S0/S1 作为唯一待检验训练矩阵，分别辨别移除熵、集合监督、学生证据变化的贡献。观察/重复周期残差预测保持独立可行性支线。历史 TIME、LONG_SILU、Sinc 独立正结果保留，不能将不同配方的收益相加。84% 目标继续 active。
