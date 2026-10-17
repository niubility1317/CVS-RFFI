# 共享微弱加性 IQ 方向：源域可迁移敏感性问题

2026-10-17。状态：算子、固定问题与本地合成检查已实现；未登记远端实验、未读取真实 IQ／checkpoint、未启动源推断或 E20。上一原 SupCon 混合视图实验已按固定判据关闭，训练底座继续是 F＋跨 RX 同 TX、E20、原默认 cosine 头与联合训练。

## 当前唯一问题

当前 F 的默认头是否存在一个在其他源 RX/day 上构造、对全部 TX 使用同一向量的微弱加性方向，其身份判决影响可迁移到留出 RX/day，并明显超过同能量重排方向？这是后续 IQ 增强的源依据问题，当前并不训练。正结果只说明模型存在共享方向性敏感，不能证明纯 RX 噪声、标签保持、Ag 有效或 practical 收益。没有敏感性也不证明所有加性观测增强都无效；只关闭本固定构造。

## 与已有结果及另一对话的边界

- 本任务原 SupCon viewrouting 的 Q_mix 相对 Q_ord 七个视图均下降，完整结果见 [报告](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01/report.md)。该路线及旧 phase／gain／frequency／timing／noise／spectral 等固定失败方案保持关闭。
- 最新总索引中 [非 DC 源峰值](E:/type10-7/automation_reports/CV-SincNet/20261017-diagnostic-source-nonDC-spur-crossfit-manysig-m1-r01/report.md) 只在14/18折选出相同RX/bin，且峰接近短前导谐波，固定候选关闭。不将这些峰当作可抹除杂散，不另启窄带峰增强。
- 已读 [真实同TX独立观测配对](E:/type10-7/automation_reports/CV-SincNet/20261016-phase1-FcrossRX-realpair-source-RX1-e20-s2026092701-m2-r01/report.md) 与 [C4精确相位](E:/type10-7/automation_reports/CV-SincNet/20261016-phase1-FcrossRX-phasecycle-source-RX1-e20-s2026092701-m1-r01/report.md) 完整报告，二者按各自规则关闭；不把可逆或真实同TX自动当训练有效。其held source V5400与本任务held source L1260口径不同，不能拼接或作新匹配控制，不继承其模型。
- 已读旧 VAT r02 的真实配置：U上局部分布一致性、逐包方向、不同训练合同与预算。这里的预期问题是冻结默认模型上的一个共享观测方向，0训练／0新训练损失；不重跑其U一致性配方或读取其权重，也不按历史目标分数选半径。
- 已按要求读取“开发 CVS Phase1 星地增强训练”的最新任务内容；当前方向为接收量化残留可观测性。本任务研究共享加性方向，不进入量化、剪裁或ADC模型，不向该对话发消息或改其工作。

## 方法依据与限制

[Miyato等 VAT 原文](https://arxiv.org/abs/1704.03976)定义不使用标签的局部分布方向，提供概率参考与探测梯度的算法依据；[Moosavi-Dezfooli等](https://arxiv.org/abs/1610.08401)研究共享的微小扰动方向。两者均不能证明当前RFFI的跨RX clean增益、物理身份保持或星地适应，也不声称这里精确复现原算法。

[IEEE Jamming Echoes](https://doi.org/10.1109/INFOCOMWKSHPS65812.2025.11152806)描述带外干扰会损害RFFI，支持外来加性观测变化的动机；没有由此选择其干扰频率、强度或假定增强有效。本文不用该论文作为当前输入中存在杂散的证据。

接收到的IQ仍含原传播、噪声及TX/RX作用。新向量只表示微弱加性观测压力，不被命名为纯RX硬件分量，也不等于替换真实接收机。小范数不保证身份语义不变，冻结模型敏感也不保证增强训练受益；最终仍须匹配E20的跨RX身份识别收益。

## 已固定的算子

仅适用原native RMS后的FP32 [N,2,256] IQ，采样率25MHz、equalized1，保留NoExplicitCFO及全部必要内部逐包补偿。两路I/Q坐标不交换、不量化、不剪裁原信号、不加PA、不做CFO移频、时移、拼接或通道重复；不重跑外部RMS归一化。参考窗与整个原网络照常处理变动后的IQ，不要求内部参考量等于clean。

1. 一组合法参考IQ的每包经同一个当前冻结F/default head的eval前向得到无梯度概率参考；再叠加同一个私有Rademacher探针，探针复RMS为1e−4，时间均值为0。第二次eval前向用float64 logsoftmax/KL，仅对共同输入向量求梯度。概率参考与搜索均来自同一模型，不建立teacher模型，不读TX标签、预测正确性、质量或角色真值。
2. 所有参考样本的输入梯度按包数求均值，只产生一个[1,2,256]向量。去除两路各自的时间均值，限制复RMS≤1e−3、单分量峰值≤3e−3，FP32相对容差1e−6。这里缩放的是加性向量，原IQ不被幅值剪裁。梯度范数≤1e−24时记录不可辨识并输出0，不回退随机方向后宣称成功。
3. 同能量控制分别在I/Q通道内部重排这个向量，保持通道能量、总体复RMS、峰值与零均值。它重排的是新增向量而非原IQ，没有切片接缝。私有seed不推进原采样、dropout、U或全局随机流。
4. raw、x＋direction、x＋permuted三视图都使用完整原模型和默认头；不拟合分类器，不更新参数、EMA、BN或优化器。所有向量的RMS、峰值、参考包数、前向和输入反向成本分别记录。搜索中的KL只定义方向，不是训练时新增最小化损失。

单位功率native RMS条件下，新增向量的功率比至多1e−6，即signal-to-added-perturbation ratio至少60dB；它不是原录音或总接收SNR。此数值在新源读取前固定，不是从目标准确率选出。

## 最小源问题与模型来源

只分析原可见fit L5040：RX3/4/6/8、day1/2/3、6TX、每单元70包。U真值始终隐藏，RX1原L1260、U、V和target均不进入计算。继续复用原VALIDATED_ONCE角色，不重建数据或改变物理ID。

拟用自有satmix25源run的B00/seed2026092701固定final20默认头checkpoint：F/X3/unifiedSinc＋CROSS_RX、scratch E20、原13L＋7U、source-only/final-only、无祖先初始化或target选择。其来源与物理角色已有可复用核对；源执行器仍须核对实际checkpoint/config/合同一致，不能只信名称。本次仅冻结诊断，不把它当新训练父权重；不加载旧VAT、C4或其他对话权重。

12折以一个RX/day单元为留出：参考仅来自其他3RX及其他2day的全部6TX，2520包；检验是该RX/day全部6TX共420包。每包正好作为留出评估一次；构造方向的参考每折同时排除检验RX和day。方向对留出单元所有TX完全相同，不依据TX单独构造，不使用TX标签决定扰动。已训练模型本身见过fit源数据，因此这不是新的模型盲测，只检验方向构造在源条件之间的迁移。

已固定预期费用：参考forward60480包、输入梯度贡献30240包／240次调用；三视图source eval15120包；真实匹配checkpoint no-query合成smoke2包，合计75602新NN包前向。模型epoch、参数更新、新teacher、头拟合、As生成均0。输入反向和前向分别报告，不把0更新说成0成本。原模型E20历史成本单列，当前合成验证另列。

## 固定放弃条件

本问题只支持单独登记后续实验，不能自动晋级训练或正式target。需全部满足：

- raw fit准确率≥98%；三个视图和全部5040包、12折完整且有限，12个方向均可辨识，预算与数据权限准确。
- 同能量permuted准确率减direction准确率≥2pp，且至少3/4源RX各≥1pp。损失、梯度、KL或输入变化不能替代此真实身份判决差异。
- 方向视图在raw正确固定集合的保持率整体、每个RX及每个TX均≥95%；permuted整体保持率≥99%，每个RX≥98%。不删弱包或重选集合。

任何一项失败，关闭本固定共享方向问题；不改epsilon、xi、峰值、迭代数、模型、RX/day、阈值或seed救跑。通过也只说明共享方向脆弱性，不证明Ag有效；随后仍须另行固定同成本random/identity控制的L-only增强E20，原U不改，独立检验clean净收益，再考虑As和完整四因子。

## 本轮实际完成

- 算子／代数与状态检查：15个反例通过，64个toy合成NN包前向、32个输入梯度贡献。
- 真实F结构合成检查：20个NN包前向、4个输入梯度贡献，通过默认头输入梯度、RMS/峰值、同能量控制、原IQ/参数/已有梯度/随机流保持。没有实际模型权重或真实IQ。
- 源角色与科学条件检查：19个反例、5040条合成metadata、12折，0NN。
- 合计84个合成NN包前向、36个输入梯度贡献，0优化更新／真实IQ／checkpoint／target。首次真实F检查在原生参数解析前构造失败，0NN；按已验证SSDG导入顺序修正后通过。保留既有NumPy桥警告和日志，没有改环境或安装依赖。
- 源执行器、正式登记、候选唯一P0/P1、真实checkpoint smoke和远端发布尚未完成，当前不是LAUNCH_READY，不称源实验证据或增强成功。只保留必要最小流程；泛用brainstorming的额外设计审批属于REJECTED_EXTRA_GATE，不另设审批层。

完整目标保持ACTIVE：有效Ag、稳定As、同契约B00/B10/B01/B11及交互、跨seed与正式clean＋六practical、clean净收益、lowurban>60%、suburban接近本模型clean均未完成。


源执行器与单行登记已完成，尚未发布。使用实际opaque角色handle与独立RX/day元数据，明确冻结全部模型参数；直接CLI的标准库同名遮蔽已通过重命名消除。15＋19＋24项相关反例及冻结F合成检查通过，累计168合成NN／72输入梯度贡献。唯一P0/P1的旧child入口已修正并限定复核PASS；无其他直接P0/P1。真实CP smoke/源输入/识别结果待发布，不把本地检查当科学证据。


源实验20261017-diagnostic-F-crossRX-sharedDirection-fitL-s2026092701-m1-r01已完整ANALYZED／VERIFIED，执行361978e3449d254258b9b61141508a0ed4c945ab：raw5032/5040、direction5030/5040、permuted5032/5040，总差0.03968254pp，逐RX最大0.15873016pp，0/4达1pp；direction修复1破坏3。全部15120源判决/216单元/12折向量和75602NN／30240输入梯度贡献／240调用独立复算一致，0训练更新/target。FAIL_FIXED_SHARED_DIRECTION_CLOSED；不调epsilon/xi/峰值/迭代/模型/RX/day/阈值/seed，不进入E20。结果只关闭本固定构造，不证明所有加性增强无效；完整联合目标ACTIVE。
