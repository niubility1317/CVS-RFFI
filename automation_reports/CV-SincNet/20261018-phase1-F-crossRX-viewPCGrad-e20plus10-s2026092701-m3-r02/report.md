# F＋跨 RX 同 TX：ordinary／完整 practical 监督梯度组织 E30

状态：PREREGISTERED；三行实现和本地检查完成，尚未启动。联合目标 ACTIVE。本轮属于 A_s 训练组织，新增 IQ A_g=0；不能替代 B00/B10/B01/B11 或跨种子证明。

唯一假设：保持原 IQ、ordinary／As 权重及 U 规则，处理两种合法有标签视图的反向梯度分量，可能比普通拼接更好地兼顾跨 RX clean 与 practical 身份识别。这是待证假设；梯度冲突、投影次数或辅助损失下降不构成识别收益。

## 固定矩阵与目标

run_id：20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02。同一 model seed2026092701、split392005，三行 ORIGINAL／MEASURE／PROJECT；各从同一原 fullsourceAs E20 student＋own EMA 独立加10至E30，重置原 AdamW／cos10。各2220新增／6660累计更新，父20轮共享一次；新增总30模型轮，唯一训练成本50模型轮，三个模型逻辑累计90轮。不是从零 E30，也不从失败变体续训。

ORIGINAL 保留原梯度；MEASURE 每步额外求 ordinary 和 As 梯度但不修改 native 累积梯度；PROJECT 做同样求导，再仅对冲突的 As 分量修正。MEASURE 控制求导成本和潜在副作用，不能省略。三行均全部完成，不依据中途目标或源微涨分排序／选择性重跑。

所有行保留 F 三 SIG_TIME＋统一 Sinc CENTER25K/CENTER_UNITS、joint160、原默认 cosine6×160／scale30 和原骨干联合训练。没有新分类器、部署分支、特征对齐损失、U 视图或教师。

数据为原全部五源 RX1/3/4/6/8、day1/2/3、L6300/U56700（TX始终隐藏）/V27000，实际物理角色与原 source_contract 精确相同。没有训练 RX 留出或 L5580/dev720。只用现有256点IQ，无长上下文依赖。NoExplicitCFO 与既有必要内部逐包补偿保持。

父权重为原预登记 fullsource concat E20、CONCAT-s2026092701/source/final_ssdg.pth（父 run 20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02），从零、同一完整源契约、固定 final20，源 V 只监测；本次不是依据其目标分数挑选。既有来源核验沿用，首步严格核对 source角色、实际args、final20/4440、ownEMA 和祖先；不加载其他权重或默默回退。

## 实际 IQ、损失与梯度

三行共同采用原顺序六批缓冲和相同隔离 L/U loader RNG：L20261018131、U20261018132；不重组样本，不改变曝光。普通增强沿用实际父 args 的 E13 固定日程；本对照不是完全无增强。每步 ordinary128 一次生成后两次 RDrop；完整 As128 从原 raw 输入产生一次后两次 RDrop，不在 ordinary 的 AWGN/multipath 后重复星地信道。原 practical 场景、传播、噪声、RX 硬件／补偿依赖及随机种子保持，全部 legal labeled As 受监督，不设 urban 高置信门。

原有标签目标始终为 .5 RD(ordinary)+.5 RD(As)+.1 SupCon，原 U 的 weak/strong、邻近批门、ownEMA 和损失规则保持；student变化可自然改变后续伪标签，不能称其轨迹完全相同。SupCon 仍只取共同 ordinary 对。总损失尺度、四次 L 前向和原始监督份额不变。

令 gO=∇[.5 RD(ordinary)]，gS=∇[.5 RD(As)]。在与原 global clip 相同的物理梯度单位中（两个 Sinc CENTER25K 梯度先除25000，其余原参数坐标不变），计算内积。若内积为负，PROJECT 用 gS−(gO·gS)/(gO·gO) gO；否则原样，gO 为零也原样。再把修正映回参数梯度单位，加到 native 总梯度，执行原 physical clip5／AdamW。全部且仅原可训练参数参与，默认头继续联合更新；原 SupCon／U 的已累积梯度不覆盖。

这是受 [PCGrad 原论文 §2.3](https://arxiv.org/html/2001.06782v4)启发的固定单向偏好变体，不冒称论文完整随机对称算法。参数坐标和缩放明确固定；它只去除当前两个监督目标的一阶反向分量，**不保证 AdamW、裁剪、动量、weight decay 后的实际位移降低 clean，也不保证有限步或未见 RX 收益**。所有结论仍由默认头识别验证。

旧 RX-update-projection 诊断按五个源 RX 的 CE 约束 AdamW 提议位移，且旧固定版本源否证失败；本次不修改其阈值或继承其产物。当前约束对象是 ordinary／实生成完整 As 的监督梯度，时机在原 AdamW 前、默认头也联合参与，无额外风险样本。关联对话的 U 教师星地训练也未被复制或改动。

## 发布、停止与自动评价

本地命令：python -m experiments.cvs_view_pcgrad30.publish。远端 CWD：PROJECT/releases/cvs_F_viewPCGrad_E30_20261018_r02；入口 CVS-RFFI/bin/python -u experiments/cvs_view_pcgrad30/runner.py。实际代码 commit 由该发布读取本地HEAD并写 release_commit.txt；登记的 PENDING 字段发布时替换，Git commit 是唯一代码身份。

不可变输出：PROJECT/runs/20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02。CPU2，普通账户；共享容量检查全 owner 每 GPU 最多3进程并预留12GB，具体 GPU 由实际启动记录给出。一次 N607 preflight、单 release archive 本地／远端 SHA 比较、远端编译和合成检查。dispatcher 首步一个真实匹配 checkpoint 的 no-query 合成 smoke2，PASS 即自动三行训练；没有额外 smoke 许可或 source-health 训练门。

固定 final10／累计30后，每行自动原 clean＋六 practical 各168000条逐样本默认头预测。本行七视图 prediction/logp 完整固定后单独 CPU truth-last；不改测试 SNR、信道、received IQ 或困难样本覆盖。逐行结果不得回流任何其他训练。实际训练、预测、评分、分析分别登记。

只对本 run 的来源／角色错误、越权、非有限值、输出碰撞、梯度生命周期不符、不可执行或预测／评分无法闭合等系统技术故障停止未启动行；保存产物，不自动重试，不停止健康已运行旧任务。低成绩不技术停止。

固定单种子反证：PROJECT 必须分别相对 ORIGINAL 和 MEASURE 同时 clean≥0、mid urban≥0、low urban≥＋.5pp，各 suburban≥−.25pp。任一失败关闭本固定组织，不改投影坐标／单向规则／seed／父／幅度／预算救跑。所有结果保留，target 不决定参数、候选排序或自动扩种子。完整目标仍是匹配 noAs B00 的 clean 正收益、low urban>60%、每 suburban 距自身 clean 探索目标≤2pp且不靠降低 clean；本三行不能独自证明 A_g 或交互。

## 验证、成本与预期产物

本地21个协议负例 PASS；闭式冲突／正向／零梯度、Sinc物理单位、共享辅助梯度保留和 RNG 检查 PASS。真实 F 结构三行合成hook1536包前向／3次合成更新，ORIGINAL 与 MEASURE 梯度最大差0；真实IQ/CP/训练更新/target均0。一次独立限定P0/P1审查PASS；r01发布后独立post-state发现PROJECT名称与路径占位符冲突，直接修复后一次同问题定向复核PASS。未读真实数据／权重或目标成绩，无重复全审。

三行每行新增 L NN1136640，student train预计1703640、原U teacher预计567000；实际源评估、初始smoke、预测、teacher、累计父成本及wall分别记录。MEASURE／PROJECT 每行额外4440次 VJP，按每VJP覆盖一对128包计1136640个额外 L 包导数；该计数不等于精确FLOPs或耗时。ORIGINAL额外VJP0。局部捕获仅记录host enqueue时间，总训练wall包含实际GPU计算。发布远端合成再1536NN／3合成更新，以及一次实际CP smoke2，均单列。

预期原source args/config/module、完整epoch/step/LU序列、原ordinary/channel审计、reference_steps/unit_steps、view_gradient_steps（每步内积／范数／系数／实际修正／额外VJP）、final默认头CP/source_frozen/completion、七视图prediction、独立全RX/TX/day指标及CM/配对错误、成本与同row报告。权重、IQ、预测、原始日志和私有runtime不入Git。

## r02 技术修复边界

替代20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r01；r01在创建source run root前退出，实训练/IQ/CP/NN均0，远端与本地各1536合成NN／3合成更新保留。r02仅将发布注册路径展开限制为PROJECT/前缀，不改候选名PROJECT、说明文字、科学机制、三行、父、数据、E30预算或判据。定向检查与定向独立P1复核PASS；不覆盖旧release，不继承旧合成状态，无自动重复提交。预算未延长。
