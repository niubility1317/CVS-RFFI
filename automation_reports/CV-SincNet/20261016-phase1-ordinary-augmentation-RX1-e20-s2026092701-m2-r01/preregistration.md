# 旧普通 IQ 增强：匹配源 RX1 留出 E20

状态 REGISTERED_NOT_RUN。唯一问题：固定当前SIG_TIME＋RDrop时，旧普通L增强整体是否有真实跨源RX身份收益。两行N0/N1从零E20，相同初态/两前向/LS.01/RAW_HZ/defaultcos/joint160/NoExplicitCFO，N0关闭普通L增强，N1保留原配方。无As、新G、额外loss、结构或U机制。旧sourcefold的CENTER25K/LS0权重和fullsource权重均不加载。

真实数据契约为L5040/U45360，训练RX3/4/6/8、day1/2/3、6TX；RX1 L1260/U11340不用。原V保持一个物理角色，seenRX21600作原健康观察，RX1 V5400只在final20无更新评价，不是新增Vcal/Vselect角色。所有模型、ownEMA、训练/probe/calibration在构建前排除RX1。来源carrier与物理ID沿用已验证句柄，不重新验证数据。

当前原native更新预算222/epoch、4440/row保留；实际四RX U loader178batch，通过原sourcefold方式循环至222，7轮U重复观察396368/row。每行L物理曝光568320，两次RDrop共1136640个L前向；U/EMA/源健康/finalheld5400及checkpoint-smoke2另按实际计数，不把两个E20当单个E20成本。总2scratch/40model-epoch/8880更新；不扩预算。Native AdamW lr.0002/wd.0001/cos，13L+7U，全部实际args发布后留存。

普通增强一次生成IQ，RDrop第二次复用同一IQ。两组使用相同独立L/U loader采样流，并记录实际全部L/U物理样本序列逐条相等。N1普通增强独立Python/NumPy/CPU/currentCUDA随机流且恢复全局状态，避免打开增强改变dropout/采样随机流。分布和原增强参数保持，训练轨迹不冒称旧run的逐位复现。原U弱/强、旧门控、ownEMA及损失不改变；U真值始终隐藏。

N1包含原配方中已开启的PA/AWGN/multipath等，作为原有整体配方接受源证伪；不能因此预先称标签保持或有效Ag。没有新增/放大这些物理损伤，实际base cfg与每epoch概率、波形MSE/峰值落地。无训练星地视图，正式测试IQ/信道不触碰。

科学规则在训练前固定：N1−N0在heldRX1 V accuracy至少+1pp、macroF1不下降；seenRX V均值下降<=.5pp；全4440更新有限、TX×4RX×3day覆盖完整且真实L/U样本顺序匹配。任一失败则结束ordinary整体作为Ag，不换heldRX、强度或seed救援。PASS仅支持另登记其余4源折；完整五fold均值>=+1pp、>=4正折、最差fold下降<=1pp后才考虑正式组合。当前不自动扩展、测试target或启动joint。

每GPU所有owner与preCUDA进程合计<=3、free>=12000MiB、共享原容量锁，CPU2/workers0，ordinaryN607，单owner/不可覆盖root。两行只能有一个launchowner。技术异常保存本run产物并停止待启动行，健康已启动行继续，不自动重发或干预其他任务。精确命令、CWD、路径、seed和预期产物见experiment.json。

本地真实模型合成检查PASS：198个相同初始state张量、两行共512网络前向、128普通增强包；同IQ两次RDrop、默认W/SIG梯度、EMA方法绑定/状态、私有RNG与loader generator一致，5协议负例。第一次计数适配错用encoder.forward_features，实际接口为features，修复后通过；随后本地Torch2.1/NumPy2 ABI阻止RandomSampler.numpy路径，改为检验绑定generator的真实randperm流，不修改生产sampler。N607实际完整L/U序列将独立核对，不能将本地seed检查冒充完整真实序列证据。当前完整模型检查执行2轮共1024合成网络前向，无真实IQ/checkpoint/更新。

真实scratch checkpoint无query smoke是各worker首个模型步骤，PASS直接继续；无额外许可。目标仍为clean和星地双收益、lowurban>60和suburban保持，本源折不构成达标。

唯一独立下一运行P0/P1审查PASS，未发现直接缺陷；独立检查再执行512合成网络前向/128增强包/5负例。本轮本地已知总1536合成网络前向/384增强包，0真实IQ/CP/更新；只读N607普通账户/主机/路径/GPU preflight PASS。当前尚未发布运行。
