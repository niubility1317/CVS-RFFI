# 纯 F 从零 E20：同一受损 IQ 的非真类关系迁移

状态：RUNNING（2026-10-19 12:04 HKT 独立读回）。一个 seed 的四行固定矩阵。代码/config commit 为 a276b957ce85b939e75cac19f6cc9e5191c44398，发布commit为 1e7cc414f73f093dfbdccee0e4ba7cf90c7a5324；Git远端OID已独立比较相同。归档传输、远端compile、checks均VERIFIED；唯一P0/P1审查PASS。四行分别位于GPU3/4/6/7，PID/CWD/argv/GPU与逐步日志增长均VERIFIED。此时B0已E4、其余三行已E1；尚无终轮、正式预测或评分结果。不将本报告 SHA 作为门槛。

12:18 HKT 更新：B0 已完成自身从零 E20/4440 更新并固定终轮；源 V27000 clean=98.0556%、最差源 RX=94.8148%，不能冒充正式未见RX成绩。真实特征为2×160；变化参数项包括Sinc2、signature7、time17、frequency18、PA15以及默认分类权重1，说明并非只调整融合/头。B0共568320个训练L前向包。七视图各168000包的预测已固定，NN=1176002（含2个synthetic smoke），参数冻结、独立头拟合0，truth未读。其余三行此时已E10，源矩阵决策与正式评分均未发生；全run保持RUNNING。私有独立读回见 local_artifacts/F_nontrue_E20_20261019/source_artifacts.json。

本轮遵守用户最新目标附件：F 仅为三路 SIG_TIME_X3、CENTER_25K 和 CENTER_UNITS。默认 cosine 6×160、scale30，NoExplicitCFO，骨干和默认头联合训练。无 CROSS_RX、RDrop、附加部署分支或拟合诊断头。所有学生从零训练累计 E20；不会复用旧 FcrossRX E20/E30 学生权重。

唯一假设：完整后验蒸馏可能施加不必要的真类置信压力，本轮只测试删除该内容、保留其余五类相对关系是否带来识别增量。先前同物理 V1800、同两次完整信道观测中 CSIL 的源 mid/low 为 86.6667%/78.1944%，PAMF 为 72.8056%/59.0833%。这是迁移假设的依据，不是纯 F 的结果或公平上界。

上一轮完整后验 KD 已关闭：其 FcrossRX E30 formal clean 相对普通 As 下降 2.8583 pp，low 上升 0.8524 pp。不能据此确定置信压力就是失败原因。本轮不扫原 KD 的权重或温度，而是更换监督对象；辅助项对每包学生真类 logit 的导数严格为零。历史 self-subnetwork margin-envelope 保留真类置信并从自身随机子网络取竞争关系，与外部冻结 CSIL、排除真类的条件分布不同，旧负例不直接覆盖当前问题。不将算法名称当作创新或有效性证据。

矩阵和完整参数只维护于 experiment.json 与引用 config。四行：B0 原普通 CE；C0 为 .5 ordinary+.5 raw 的计算匹配 NoAs；B1 为 .5 ordinary+.5 complete practical(raw) CE；M 为 B1 加唯一非真类损失。C0、B1、M 每步均执行 256 个学生 L 包、128 个完整 physical As 和128个冻结CSIL 包；C0 的 As/teacher 仅 shadow。B0 只有原128学生L前向，保留原始锚点，明确不声称其成本相同。

非真类辅助项为 `.5*6*mean128[1(argmax teacher=y)*KL(softmax(teacher[-y]) || softmax(student[-y]/6))]`。教师原生 cosine scale5 不再额外加温；学生部署scale30，对辅助项除6。按全部128归一化，错误教师仅屏蔽辅助项，每一个合法 As 都保留温度1、smoothing .01 的真标签CE。主 CE 总尺度1；相比B0，普通输入份额由1变成.5，新增raw/As份额.5，新增辅助项额外改变梯度尺度，依靠B1和C0归因，不能声称一切同时不变。U56700 真值隐藏、原 ownEMA/伪标签资格和13L+7U规则均不改。

数据为 L6300/U56700/V27000，RX1/3/4/6/8、day1/2/3、split392005。通用 trainer 在构建数据时比较实际完整 role_ids，禁止 target loader。历史教师固定last200/9800updates，完整来源已在旧probe核实；首次前向再次核对 payload/同角色/无祖先/无target选模，冻结且置于学生state_dict外，不参与ownEMA或部署。其历史成本单列。

已只读核对实际运行依赖：fullsource runtime commit b94076b191648103daeb8a9cb1c17bdcc9c0e231，实际 train_ssdg 与 DataAugmentation 都来自原 evidence_responsibility release 的 experiments/adv3b02_xuc/code，不依赖同名根code实现。原普通增强包含循环移位、幅度、相位、CFO/相噪、AWGN20—36dB、轻多径/DC/taper；p_dac=0、class-signature关闭。现场 `aug_enable_pa_normal=True`，native调用显式传 `no_pa=False`，所以原 ordinary 还包含 p_pa=.14 的既有 PA 失真，不能只看 augmentor 默认 no_pa=True 就误报关闭。本轮遵守“保留原普通增强”，没有新增PA/DAC/IQ失真；As从raw独立生成，不串接该ordinary PA。实际开关继续随resolved产物核对。

完整 As 始终从 raw L 生成，不从 ordinary 或已受损 IQ 再叠加；只在原始256窗口内采用既有 residual/post_sync/noEQ，reference30dB、reflect边界、物理延迟/同步原实现。源输入单位RMS，As前原channel normalize。信道内部保留AGC（请求单位RMS增益，限幅±30dB），因此部分总幅度变化被AGC消除，触及限幅时不保证单位RMS；As输出后无额外normalize。ordinary的0.9—1.1幅度在增强后没有统一输入RMS层将其精确消除；参考响应分支具有平坦增益商去除，不能推断全模型严格增益不变。普通循环移位最多±32会移动原256包与固定[80,160)参考窗口的相对位置；此既有行为各行共同保留，As不额外叠加这次移位。硬件session按RX/day/eq，非TX；train/eval随机流及固定formal数组隔离。ordinary与As是不同输入、各承担CE；teacher只见同一份As。C0是raw额外槽位。所有现场resolved开关与实际模块路径随产物保存。

源端推进规则固定：M−B1 的两draw源 mid/low 平均至少 +.5 pp，且 M−B0 的全V clean 不低于 −.2 pp。只用这两个必要判断，不加几十个分层门槛。固定终轮、无best模型。源决策先落盘，之后独立truth评分；失败关闭当前非真类配方，不小扫系数、改seed或换名挽救。通过仅取得E50资格，追加前再固定E21—50的优化器/LR/EMA/U缓存/采样/信道衔接及匹配B0/B1控制；不把20+30冒称连续cosine50。

命令：`CVS-RFFI/python -u experiments/cvs_F_nontrue20/runner.py`。N607普通账户；CWD为本run专属release；实际路径来自private runtime，不入Git。全owner每GPU最多2进程且空闲≥12000MB；锁内分配，不改健康任务。每row独占root和owner，系统性数值、权限、数据、来源、接口、输出或预测失败保留产物、停止新队列，不按性能终止或自动重试。实际PID/CWD/argv/GPU和日志增长需独立读回。

必要验证为损失真类梯度零、teacher detach、类重排、错误教师mask、协议负例；一次独立P0/P1审查。launcher首步读回自身scratch checkpoint并执行真实模型smoke，PASS后直接训练。每row完成后自动源readout、固定clean及六practical预测，全部7视图固定后独立exec truth-side评分。完整源矩阵决策固定在正式评分之前。每view168000包；目标未见RX0/2/5/7/9/10/11和既有day集合。报告acc/macroF1/最差RX、修正与新增错误、自身每个suburban差距、真实joint160、Sinc/序列/主干/默认头参数更新、完整日志与成本。单seed只能探索，未达目标不得标记goal complete。
