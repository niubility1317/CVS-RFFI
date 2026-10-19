# 纯 F E20：同轮首次 As 重访

状态：PLANNED。逐行矩阵、seed、预算、命令与路径以 experiment.json 为准；只有一行新增E20，B0/B1完整读出复用。

唯一假设：当前B1的low urban训练命名流重建45/69、同L新信道89/138、源V新观测2237/3600，困难已在训练期出现。当前每次遇见同一L都刷新As，可能缺少对同一受损观测跨优化更新的重复学习。本轮只检验epoch内重访首次As、下一epoch再刷新；不从已见正式分数选择候选，不预判有效或物理上限。whole-raw、sceneRisk、nontrue、PCGrad、旧独立view固定方案不重启。

实际变化：每epoch按physicalID五元组保存第一次真实fullAs IQ与生成step/slot/metadata。后续同epoch遇见相同raw时重用该IQ，下一epoch第一L步强制清空再生成。每步仍照原Config、命名流与RX/day/eq硬件session完整生成fresh128；cache命中时fresh IQ仅为shadow，不入NN。缓存不使用TX标签或学生置信度。它同时固定当epoch内的scene/噪声，实现改变的是观测时间相关性与有效多样性，不能说只改变学习次数；仍每epoch跨轮刷新。

loss为0.5×mean128 ordinary CE＋0.5×mean128 used-As CE（T1，smoothing0.01），所有合法有标签As真实CE完整保留。ordinary原增强真实返回、网络前向和默认头梯度均保留；不改原U弱/强/资格/ownEMA，不增加KL/teacher/全分支对齐或部署参数。纯F三路SIG_TIME＋统一Sinc修正、实际joint160、NoExplicitCFO、内部融合与原cosine默认头联合训练。L6300/U56700隐藏TX/V27000、源RX1/3/4/6/8、day1/2/3、split392005完全不变；不新增Ldev或RX留出。

预算：seed2026092701，同seed随机初始化scratchE20＝13L＋7U、4440更新、原AdamW/cosine20、每步两L128前向。新增20模型epoch，L学生NN1136640，U学生NN396900，自身EMA eval396900，full fresh生成568320，usedAs NN568320；unique、hit、替换差异及epoch reset按实际计数。无额外NN、额外optimizer更新或外部CSIL。旧B1的568320外部教师eval shadow只保留历史成本，不重算；其load使用fork_rng而eval无随机消耗。B0/B1直接复用，不改变NoAs侧，不通过重定义较弱NoAs制造正差。

源规则先固定：V1800物理ID×2draw的mid/low均值相对B1至少+0.5个百分点，fullV27000 clean相对B0不低于−0.2个百分点；仅这两项科学资格，不加逐层/分组门槛。FAIL关闭固定epoch-first-As配方，不扫refresh间隔、seed或日程救援。PASS只允许准备自身E20＋最多30及匹配同预算B0/B1，追加前另固定optimizer/LR/EMA/U缓存/sampler/channel衔接；不自动执行，不由正式成绩决定。

终轮固定FINAL20后源资格、all7预测、独立truth-last评分、同row报告与登记自动串联。固定clean＋六practical received IQ、全部168000物理ID、原SNR/场景及单包默认头推理不变。全部7视图prediction固定且源判定落盘后CPU scorer才开truth；源FAIL也完成预登记正式评分。报告宏F1、最差RX、修正/新增错误、每个suburban绝对值和自身clean差距、固定B0clean正确集合保持、真实骨干/joint160更新、实际成本。只有1模型seed，配对均值/正差数如实报，样本SD不可估计，benchmark已暴露不称新盲测。

原replay记录的差异已核对：viewrefresh仅本step ordinary prePA、LSAM仅本次clip closure IQ/RNG、sceneCycle仅scene排列而channel seed仍逐step刷新，均不测试跨optimizer更新的As缓存。本轮不因更换F基座重试这些固定方案。

代码目录 experiments/cvs_F_epochReplay20；普通N607既有Python3.10/Torch2.1，CPU2、FP32/TF32off；GPU4唯一新增worker，含其他owner每GPU最多2进程、至少12000MB空闲，共享容量锁。唯一owner codex/root/F-epochReplay20；CWD PROJECT/releases/cvs_F_epochReplay_E20_20261019_r01；dispatcher和row命令见experiment.json。输出 PROJECT/runs/20261019-phase1-F-epochReplay-e20-s2026092701-m1-r01 首次独占创建，恢复先核实进程与产物，不覆盖、不重复启动。原不可变rawAs runtime只提供代码依赖，随机初始化不继承其权重。

技术停止仅限角色/来源、非有限值、实际raw/ordinary/usedAs路径、epoch/cache顺序或输出接口等系统技术矛盾；只保留本run现场，不干预健康任务，不自动重试。最小流程采用一次实际代码/config Git commit、窄协议负测、launcher首步真实scratch checkpoint save/reload无query smoke（PASS直接训练）、一次P0/P1直接缺陷审查、单归档一次传输比较、远端编译/预检和启动后PID/CWD/argv/GPU/日志增长。没有设计签名/逐文件hash/额外审查门槛。

本地验证已PASS：12文件编译、23缓存协议负例和225合成apply、ordinary实际IQ/RNG不变、两个隐藏U拒绝、全128样本0.5/0.5 CE梯度、6固定config负例及B1原生参数仅run/candidate/output三项差异。尚未运行真实信道或模型训练，真实CP smoke留作launcher首步。

唯一一次独立P0/P1直接缺陷审查 PASS：缓存跨step/跨epoch、完整fresh shadow、真实ordinary/usedAs、原U/EMA/Sinc/4440预算、CP不含cache、eval无replay、共享GPU容量与all7独立评分路径均未见直接缺陷。预检VERIFIED：新run/release/archive不存在，原rawAs依赖commit3d4e188匹配，B0/B1源/资格/七视图预测评分均完整，GPU4空闲24124 MB。
