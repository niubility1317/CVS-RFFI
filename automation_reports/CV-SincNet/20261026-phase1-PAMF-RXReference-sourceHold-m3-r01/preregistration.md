# 跨接收机身份参照的单种子源留出验证

PLANNED。三条是三个方法变体，只有student seed2026092701，不是三个student seeds。本run仅为目标访问前的source-only候选证伪，不生成新目标预测，不从目标选择父权重或参数。原clean至少+1pp、raw六景>+1.5pp及同G六景超过两个对照的联合目标保持。

## 矩阵与匹配关系

所有新行从原ordinary-PL own E20+10的E30独立分叉，追加20轮/4440步到E50/11100，累计13L+37LU。原完整L监督、ordinary/raw RDrop、As真实CE且KL0、LS.01、浅中层ORD跨RX、raw U自身eval argmax/confidence≥.95/weight1、默认分类头联合训练保持；native U/EMA关闭。原L6300/U56700/V27000物理角色与合同不变。

| 新方法 | 两次额外输入 | 固定目标 |
| --- | --- | --- |
| RawReference | raw/raw | 与其他源RX的可见L身份参照关系 |
| RestoredReference | G(scene_k)/G(scene_k) | 同上 |
| PairedReference | G(scene_k)/G(scene_k+1) | 同上，两次损失先平均 |

每条使用同一冻结raw三E30平均概率q、同类且各≥.95共识mask。新增分支排除排序首个源RX1的U；原L与raw U主项仍使用全部源RX。参照只来自source L6300，各seed使用自己的固定E30坐标；不跨seed平均特征、更新参照或用U/V/目标拟合参照。按完整U batch平均，参照关系温度.1，原默认头继续由主项联合训练，不增加新投影或身份头。

共同新增权重 **0.07145844915553692**，由已完成12批无更新源梯度的预定最大中位范数规则给出，444步线性升温。证据见 [源剂量](../20261025-diagnostic-PAMF-RXReference-dose-m1-r01/source_decision.json)。该剂量不是AdamW更新保证或目标最优参数。复用原三场景U缓存，不重新加噪或混合IQ；SCENES顺序low suburban/mid urban/low urban，global step轮换，Paired使用相邻不同两景。信道seed20261016421、硬件seed20261016423、namespace `PAMF/restored_dose/U/fixed`，同原缓存。

G保持原F0 G-LU，不更新。原E30、G和主项都见过RX1；只有新增U监督留出RX1，故不能称为真正从未见过的接收机测试。每条预计U曝光1134000，其中RX1曝光226800只进入原主项；新增接受量按固定缓存mask排除RX1计算并在结束核对。

## 源域逐行收尾与选择规则

每条训练固定final E50后立即使用checkpoint内默认头对原完整V27000的八视图预测：raw/Graw、low suburban/mid urban/low urban及对应G恢复。源V的信道namespace为原 `PAMF/restored_source/V/fixed`。复用已完成sourceShift同seed PLControl八视图，核对完全相同物理顺序、标签和场景；不重训该控制。保留逐TX/RX/day、留出/参与RX、净纠错与全部负结果。

V仍为一个原角色；native训练的无状态sourceV健康验证可读，新增objective不读V。固定轮次冻结，无V梯度、归一化/EMA/参照或其他持久状态更新。源八视图预测完成后计算可见V标签指标，不连接目标truth。

科学证伪规则在新预测前固定：相对同seed PLControl，RX1三个G恢复场景平均增益≥+0.3pp，且完整V raw clean差≥−0.1pp。全部三个新行完成后，只在通过者中按留出G平均增益最大选未来三种子候选，精确相同则方法名字母顺序。没有通过者则不扩展；一个负行不取消其他行。该控制器不自动启动未来三种子或目标测试，未来矩阵另行固定，学生仍从对应原E30开始，不继承源筛选F50。

## 来源、发布与停止

加载前核对原E20/own10实际checkpoint/header与相同源角色；L参照和U缓存核对同合同及原G来源。source context只有既有五个源句柄键，不包含target/evaluation句柄。保持VALIDATED_ONCE，不额外重验数据或逐文件哈希。

命令：`PYTHON -u -m experiments.pamf_channel_pl.receiver_reference_training dispatch --config PROJECT/releases/20261026-phase1-PAMF-RXReference-sourceHold-release-r01/runtime.json`。CWD为该不可覆盖release，输出 `PROJECT/runs/20261026-phase1-PAMF-RXReference-sourceHold-m3-r01`。N607既有普通账户Python/Torch2.1，FP32/TF32off。所有物理GPU0至7可准入，每卡CUDA/pending总数<2且空闲≥10000MiB，最多三条本run train/eval子进程，各CPU2threads，保护其他owner。

两条新增配置/隐藏U/留出/配对输入检查PASS；基础关系约束六条检查及真实源E30/G/特征兼容诊断已通过。发布前一次独立P0/P1审查结果追加。launcher第一GPU步骤为真实E30默认头L2和实际L128/U256联合backward smoke，原头梯度有限非零、原L768包和AsKL0核实，通过立即启动三个训练行。Git commit为代码/config身份；一次archive SHA传输比对、远端compile、启动后独立PID/CWD/完整命令/GPU/loggrowth检查。

首个技术、协议、来源、smoke、非有限、预算或状态错误停止本run后续新启动，已健康运行的本run子进程继续完成，保留产物，不自动重提，不热改或干预其他owner。源科学负结果按完整矩阵正常完成。

新增F成本为三条20轮＝60轮；三个历史冻结E30教师共享前缀90轮，own E30包含其中，合计150唯一F轮，不把own30重复加算。G、参照和U缓存历史成本单列复用。预期各行checkpoint/完整日志/机制计数/source预测与源报告，以及固定矩阵、smoke、状态和最终source choice；私有权重、IQ、packet logits、运行配置和日志不提交Git。

发布前审查结果：本候选一次独立P0/P1审查PASS，无P0/P1；核对范围包括实际父链/缓存schema、原生训练与默认头、源曝光计数、并发和无更新V评估。两个新增聚焦检查及相关模块compile PASS。真实joint backward smoke仍由launcher首步执行。
