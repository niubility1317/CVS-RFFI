# 完整 practical 的阴影初值分层抽样：匹配四行 E30

状态：RUNNING。四行均已开始实际更新，未完成训练或取得识别增益。

## 唯一假设和依据

在保持原 practical 单包联合先验和物理接收处理的条件下，对每批 L 的独立 Loo 阴影初值作随机分层，能改善有限 E30 训练的 low/mid urban 身份识别，并保护 clean/suburban。仅研究 As 抽样组织，不称为新 IQ A_g；不增加对齐、门控、损失或部署分支。

实际调用为原 ResidualChannel：ChannelStream 用独立 shadow RNG 抽 z~N(0,1)，独立于 geometry/state/scatter 等流；LOS 按场景和状态的 mu/sigma 将 z 转幅度，然后完整 process 由传播核推导接收质量、锁定状态、补偿和 AGC。新实现先构造原 snapshot，再仅替换独立 z，随后调用原 process 一次。保留原场景频率、状态条件概率、原 15 个 RX/day 硬件 session、全部其他潜变量与随机流、噪声 N0、referenceSNR30、reflect 边界及 noEQ/residual/post_sync 路径。没有修改测试 IQ、SNR、信道或样本。

随机一维 Latin hypercube 在每个 [j/n,(j+1)/n) 概率区间放置一个独立抖动点，再随机分配给组内包；它保持每包 U(0,1) 边缘，用标准正态逆 CDF 得原 z 先验。仅拒绝机器端点，不截断正态尾部、居中、反向噪声配对或优化点集。[SciPy 官方算法说明](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats.qmc.LatinHypercube.html)支持该抽样构造；它不证明异质物理包下的梯度方差必降或识别必升。

按本批原场景×原传播状态分为最多 3×3 组；没有 TX/RX/day/接收质量/置信度分组。状态在替换 z 前产生，保持原条件分布；NLOS 中 z 不影响 LOS 的包仍完整保留和监督。所有源物理出现次数、标签、loss 权重和前向数不变。

此机制与已关闭的正负噪声、MC2、TX×RX 重组、共享传播、RX 硬件 population、dropout 耦合、carrier selector 等不同。旧失败路线不重开。分层覆盖检查是工程证据，不作为识别收益或联合成功。

## 固定矩阵、初始化和预算

|行|L 原监督对|第二监督对|阴影初值|
|---|---|---|---|
|NOAS|RD(同一 ordinary)|RD(同一 ordinary)|不生成 As|
|AS_NATIVE|RD(ordinary)|RD(完整 As(raw))|原 native RNG，不替换|
|AS_IID|RD(ordinary)|RD(完整 As(raw))|私有独立 U→正态，匹配重抽及逆 CDF 控制|
|AS_LHS|RD(ordinary)|RD(完整 As(raw))|条件组内随机分层 U→正态|

每包 .5×第一 RD＋.5×第二 RD，原跨 RX 同 TX SupCon 仅 ordinary 原对，原 U/oldgate/自身 EMA 不变。每个 RD 对的两次前向使用同一份已生成 IQ；4 L 前向/物理包，各行总监督尺度一致。全部合法有标签 As 参与，不以 urban 高置信筛除。没有新 U 增强。

四行均从唯一原 fullsource REPEAT/s2026092701 固定 final20 学生＋其自身 EMA 独立分叉，来自 20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02。该父从原契约 scratch 产生、固定预算冻结，无其他父/教师；不是按目标分数挑父。严格检查实际 source/initialization/frozen/args/role_ids 与 CP 身份。全部从零建立 AdamW、原物理 Sinc 单位及 cosine10，训练 10 轮/2220 更新，累计 E30/6660 更新；SupCon 按累计 E21 至 E30。As 仅新增 10 轮暴露，不与 scratch30 混称。

所有行保持 F 三 SIG_TIME、统一 Sinc CENTER_25K/CENTER_UNITS、joint160、默认 cosine6×160/scale30 和骨干联合训练，NoExplicitCFO 及必要内部逐包补偿保留。原 ordinary 实际 E13 配置和随机调用顺序保留，没有开启新 PA/DAC/class-signature/量化/剪裁等增强。

源契约：原 split392005、6 TX、RX1/3/4/6/8、day1/2/3、全部原 256 点、L6300/U56700 TXhidden/V27000。无训练 RX 留出，无 L5580/dev720，不扩权限。源及固定测试数据复用已有 VALIDATED_ONCE 载体，只核对匹配句柄/实际角色和继承权限，不另建哈希链或数据验证。

父 E20/4440 更新共享一次；本轮新增 4×10=40 model-epoch/8880 更新，各模型累计 E30。独立训练总成本 60 model-epoch/13320 更新，与逻辑四模型累计120分别披露。每行预计物理 L284160、L 学生1136640前向、含U学生训练1703640、教师 eval567000；源 eval/smoke/CP/cache、源资格23400前向、正式1176000预测、CPU抽样/排序/逆CDF/日志/wall 均按实际单列。无额外模型前向/VJP/教师或参数；CPU开销不当成0。

## 固定反证与自动评分

源 V 保持单一角色；固定 1800 原 V（每 TX/RX/day 单元按原 opaqueID 前20，共90单元）仅默认头无更新读出，clean＋每种 practical 两个固定 draw。与原 full V27000 health 分开报告，不称新独立盲测。

AS_LHS 相对 AS_NATIVE 和 AS_IID 两个控制均须满足：clean不降、每个 draw 的 low urban 至少+.5百分点、mid urban不降、各 suburban最多−.25百分点；同时 clean 相对 NOAS 不降。容差1e−10百分点。任一失败关闭固定 shadow-only/scene-state分组/128batch/父/seed/E30 候选；不更换变量、组、bin、seed、父或预算挽救，不自动扩种子。源通过也不是 A_g 或联合目标完成。

每行固定 final10/cumulative30，不挑轮次或候选。每行完成即自动 clean＋六种原 fixed practical、168000物理包/视图预测；该 row 全7视图预测固定后独立 CPU truth-last 评分。全部4行完成，源矩阵判定先于 primary 读取正式分数；正式分数不回流其他行、调参或重跑。源失败也保留预登记终态，不删困难行或样本。

最终报告每场景 accuracy/macroF1、各 RX/TX/day、修复/新错、与 NOAS clean 净差、与两个 As 控制的 urban 增量，以及各 suburban 绝对率和与自身 clean 的差。探索接近 clean 定为各差≤2百分点；同时要求 clean不降，不能降 clean 缩差。本矩阵不是 IQ A_g×A_s 四格，不补造交互项；完整联合目标仍需独立有效 A_g 及组合证据。

## 执行和最小流程

Run：20261018-phase1-F-crossRX-shadowLHS-e20plus10-s2026092701-m4-r01；owner codex/root/F-shadowLHS-E30。唯一不可覆盖根 PROJECT/runs/20261018-phase1-F-crossRX-shadowLHS-e20plus10-s2026092701-m4-r01；逐行 source/qualification/predictions/scores 互斥。CWD PROJECT/releases/cvs_F_shadowLHS_E30_20261018_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_shadow_lhs30/runner.py。代码身份为实际 Git commit，publisher 从 HEAD 写入 release_commit.txt/登记。实际配置见 [唯一实验登记](experiment.json) 与 experiments/cvs_shadow_lhs30/design.py。

既有 N607 普通账户与 CVS-RFFI Python/PyTorch，CPU2/进程；共享资源锁，每 GPU 全 owner 总进程≤3 且≥12GB空闲；无可用资源则等待，不停/热改旧健康任务。沿用已核实不可变 source runtime，只归档本新包，不更改旧发布。

聚焦检查本地 PASS：5项抽样输入负例、1024次边缘随机化、完整128包/15session/原处理链、19项预算/父/权限/truth-last反例。0真实IQ/CP/NN/更新。当前本机无Torch，真实Torch native接口逐位对照明确留给远端--checks，发布必须PASS；不把本地检查当完整模型验证。一次独立 P0/P1 PASS，覆盖物理依赖/分组/随机流/deepcopy/epoch偏移/共同父与EMA/发布依赖/truth-last。

一个普通 preflight、一个 Git 代码配置提交、一次归档传输 SHA 比较、一次远端编译；真实父 checkpoint/noquery 两包 smoke 是 dispatcher 首步，PASS立即继续。发布后核对 owned PID/CWD/argv/GPU/日志增长。系统技术失败（角色/父不匹配、非有限、输出冲突、非法或失败预测等）保留失败与健康行、清空待启行、不自动重试；科学失败只关闭固定候选，不擅停健康任务。

预期完整源/step/epoch/reference/unit/LU序列、实际普通增强、每批 shadow 分组/uniform/z/物理出现记录、默认头权重、源资格、全7预测/独立评分、同row完整分析/成本/报告/登记及 Git。原 IQ、ID、模型、预测、日志和私有 runtime 不入Git。REJECTED_EXTRA_GATE：不增加审批、seal、报告形式审查或重复机制评审。

## 发布后实际证据

执行提交 0a9e6541efc3a806cf3dc20e08b4c4ec9dc22ba3，远端提交一致性 VERIFIED；普通账户 preflight、一次 archive 比较、远端 compile、670 次合成物理处理（0真实IQ/CP/NN）及真实 Torch 原 native 接口逐位对照 PASS。dispatcher 首步真实父 CP/noquery smoke PASS（1CP/2NN/0真实IQ/0更新）；四行严格共同父和自身 EMA 加载已通过。

NOAS/AS_NATIVE/AS_IID/AS_LHS 分别在 GPU0/1/2/3，owned PID/CWD/argv/CUDA/CPU2 与 GPU进程独立核实。当前快照 NOAS 完成 E21/222更新，全部已读 step 为295；三个 As 已读完整 step 行155/171/170，阴影记录156/174/168，均仍在写入，EOFError 按 partial 而非失败处理。已读记录全128包，LHS每条件组概率区间完整，未见 technicalfailure。两秒窗口内 NOAS/AS_NATIVE训练日志增长；AS_IID/AS_LHS进程在GPU且已发生更新，当前窗口未增长，后续继续核对原 handle，不据此停止或重启。

这些是运行和组织证据，不是完整训练成本或身份准确率。四行按固定 E30 继续，逐行自动七视图预测/独立truth-last，不读取正式分数指导其他行。[执行快照](execution_evidence.json)。

已读取关联对话“开发 CVS Phase1 星地增强训练”的最新进展：其正在补完整源 B00/B01 的02/03 seed与初始化接线，本轮固定seed01的原先验阴影分层抽样为独立方向；不合并两条路线的增益、数据契约或成本，不干预其任务。

## 终态分析准备

已实现独立全量复算、报告及标准结果图程序，未改运行中的训练包。复算覆盖原生CSV/JSONL/observer全部epoch、2220step、L/U物理出现、原损失尺度/前向成本，以及各As全部2220×128阴影记录与原L序列、条件组/概率区间/逆CDF/native状态计数；正式分数必须等待已冻结源矩阵及完整七视图评分后读取。12项合成反例通过，能拒绝错包、错误z、缺记录与单个lowurban draw不足却宣称通过；0原IQ/CP/NN/更新/truth。程序已编译，真实终态分析尚未执行，不把合成验证称为科学结果。
