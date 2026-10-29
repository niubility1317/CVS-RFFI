# G10 星地冻结 PAMF：相对距离与无标签注册

本 run 为 `20261029-phase3-star-relative-s2026092701-r01`。所有新消费 IQ 均先按 MILKR 的同一 G10 观测规则叠加星地。只加载原 Phase1 三视图 F0/E50 和冻结 DURA v1.0/R69，不继承 Phase3 clean-E51 或其辅助读出；原默认六类头保持。P0 为星地 received，P1 为同 received 的 DURA 输出，均不是 clean。FP32、TF32关闭，G使用reuse/graph、native关闭。

## 新统计机制与控制

[Relative Mahalanobis Distance 原论文](https://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-007.pdf)通过减去无类别背景距离，减少普通类距离中无区分力方向的影响。[Mahalanobis++](https://proceedings.mlr.press/v267/muller25a.html)研究了单位特征归一化。两者不证明本项目星地/RF性能；本轮是冻结PAMF多物理分支上的自定义变体，不使用论文数字作为目标证据。

source L 的每条原记录只有一份G10观测，分别计算P0/P1，各6300条。每path独立拟合既有8分支标准几何；这已经包含完整共享协方差，不称首次full whitening。新增metric仅5个分支：fused160/time160/frequency160/pa160/pa_moments48。先按geometry标准化并unit，拟合实际类均值（不再将均值unit）和全体均值；类内/背景协方差分别按N−6/N−1归一，用0.8协方差＋0.2 trace/d I，trace floor1e-8，Cholesky白化。

每分支MD knownness为负的最小类距离，RMD为背景距离减最小类距离。由source L的score均值/标准差（floor1e-6）归一后五分支等权平均。普通MD为匹配控制，原MSP/maxlogit/margin/energy/cosine也保留；RMD主方法在任何本轮target访问前固定，不按结果切换。全部是非梯度聚合统计，没有新神经训练、身份头或OE暴露。

source V27000每record同样只有一个G10观测；不输入源模型前的clean，且源V不拟合geometry/metric或joint margin的尺度。σ由source L的七score标准差得到，floor1e-6。每path各方法仅一个global τ，以overall、5RX、6真实source场景、30RX×scene共42组known标签保护FRR≤5%及同输入F0 closed到open损失≤2pp；whole ties，不以真实scene路由target阈值。缺必要类覆盖返回不可认证，不扫描unknown校准。闭集身份与baseline均是同路径原F0。source数据为33300个原物理记录、33300次星地叠加、66600次F前向和33300次G前向，不将两计算path当独立样本。

source原始角色由既有metadata及contract.role_ids精确恢复，仅L_s/V；用sig_i取packet，base_index不是原始packet index。原ManySig是整PKL加载，再仅索引L/V，不能宣称只读这些角色的物理字节。原U_s、OE、source novel和target不进入源阶段。实际checkpoint合成星地IQ smoke先于真实L/V模型前向，PASS立即继续，不使用inference_mode或Torch/NumPy不兼容桥接。

## 完整四态矩阵

target复用新G10 RX24-1 U384与独立future Q864，same1248原物理记录。主状态DA1_REG1，比较不适应/适应×不注册/注册四态；P0/P1状态独立。源bundle只部署几何、metric、τ、ε/r/σ及90组源锚点和分数均值，不读源IQ/逐行特征bank。

适应仅使用单模型高置信及跨分支原型一致core；不是两个独立F的一致性。保持MSP≥.95、fused原型cos≥.5、time/frequency/PA至少2票、每伪类≥8且至少3类。合格伪类等权的源与U单位特征均值差乘.5、逐分支bias范数≤.1、reference坐标27固定0；F0/defaulthead/G/metric/τ均冻结。core不足返回identity并如实判科学失败。

RMD拒绝的U以source ε做DBSCAN，min_samples8、每注册簇至少16个独立opaque ID，簇数自由推断（0合法）。F0单位特征簇均值形成匿名原型，source L同TX/RX/day的LOO距离.95分位作r；source聚合锚点确定new margin惩罚。源同组第7个other近邻距离.95分位作ε。old6和所有匿名类共同竞争，无真实角色路由。margin与上轮非参数注册规则一致，便于区分新的星地几何/拒识证据与未来原型局限。

四态Q全部固定后，独立scorer才连接U/Q真值。只用U truth做匿名ID置换匹配，不在Q拟合映射；未发现/未匹配到达设备仍在未来new macro分母，错误匿名接受不当拒识。逐六场景报告宏指标、分母/覆盖、N/A、旧类接收/正确open/同输入F0损失、新类未来macro、后续unknown、混簇及合并拆分。缺类组不认证成功，不靠重抽样补齐。

主目标仍为星地跨RXunknown宏拒识总体及预登记六景≥90%，已知接收尽量高且FRR≤5%、同输入原F0 open损失≤2pp；适应有旧类正收益，未来匿名new macro≥80%、注册旧类损失≤2pp、后续未知≥90%。本轮只有一个开发RX/seed和一个G10 episode，unknown设备已有source开发评测背景；正式RX24-18未模型访问，不是独立新TX总体盲测。完成本轮不等于完整Phase3目标达成。

## 执行边界

source→P0四态预测/独立score→P1四态预测/独立score；全部固定矩阵按顺序运行，不因科学失败改变后续状态。新不可覆盖run/release、唯一owner，默认N607，单GPUworker串行、每卡至多新增2且启动前free≥4096MB，不改其他owner。代码commit、实际命令/Python/CWD/paths和artifact记入最小预登记；一次聚焦negative＋真实checkpoint smoke、一次P0/P1、一次archive比较/compile及启动后真实PID/CWD/cmdline/GPU/log增长。技术失败保留partial，不为追分重试；零簇/core不足不触发重跑。阶段结果不会回流修改阈值或方法，总目标保持ACTIVE。
