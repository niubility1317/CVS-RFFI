# 同传播 MMSE 接收处理：完整源检验

**状态：ANALYZED／VERIFIED，固定源接收视图条件通过；训练收益尚未证明。**当前训练预算为每模型累计 E30；本行新增训练和 optimizer 更新均为 0。全部 5 个源 RX（1、3、4、6、8）、源 day1/2/3、完整 L6300、U56700 真值隐藏和 V27000 合同保持，不留出训练 RX，不拆 L5580/L_dev720。

## 唯一假设与历史区别

唯一假设是：现有 practical 的 MMSE 接收处理在保留 H、噪声及质量补偿耦合的条件下，能形成可用于 A_s 的身份观测变化。先用固定普通星地 CONCAT source-final20 的冻结默认头判断同一源 IQ 的输入干预，不把源分数当作训练收益。A_g 与联合目标仍未证明。

实际普通 A_s 是 residual/post_sync、EQ 关闭 G=1、25 MHz／2.462 GHz、reference_snr_db=30、reflect、质量适配／CFO／phase／IQ 补偿／AGC 开启。旧 HQ／bridge 只将新增噪声降低 6 dB，EQ 关闭，且有共同对齐、前向或损失尺度差异；9 月旧 full/MMSE 与 NoEQ 的架构、预算、测试处理不匹配，不能证明本候选有效或无效，也不据其目标成绩定参数。已关闭的带限噪声、PDP 与其他路线不重开。另一个对话当前做周期顺序 IQ 操作，本行研究接收处理，未向该对话发送消息或借用权重。

[MMSE 官方说明](https://www.mathworks.com/help/comm/ref/ofdmequalize.html)将信道估计和噪声方差作为输入。本仓库的实现是 H＋误差代理，不是从当前录音导频获得的真实估计；这是工程接收处理变化，不能声称真实 RX 替换或在轨验证。

## 固定最小比较

同一 L 取 raw，一份命名物理随机流在原三种训练场景（low suburban、mid urban、low urban）各生成 NoEQ、MMSE24、MMSE64。三者共享同一个 H、几何、遮挡、散射、源 RX/day session 硬件及标定、基础噪声、接收质量、lock、CFO 和相位/IQ 补偿；只改变接收均衡 G，以及 G 引起的信号／噪声变化和重新计算的 AGC。噪声由 G 着色一次；不重新生成更低噪声、不强制锁定、不删困难包。unlocked 时原规则跳过 EQ，三者波形须逐值一致。

保留既有 MMSE FIR129、增益上限20 dB、channel-estimation NMSE −25 dB 及其接收质量 scale。唯一候选的目标总延迟固定24点，来源是原 NoEQ 的共同分数延迟滤波器数值延迟；默认64点只作坐标对照，不得根据准确率改选。24 是接收滤波目标，不保证有限误差估计的每包实际群延迟完全相同。没有 IQ 平移、crop、窗口挑选、输出重新 RMS 或反演未知 TX 硬件。每个视图都是对原 L 独立应用一次完整等效 GH，不将星地信道连续施加多次。

固定 seed2026092701，split392005，源传播流20261017931，原 RX session seed20261016423。6300 唯一 L，10 个默认头视图，共63000真实源 NN＋真实 checkpoint 合成 smoke2＝63002；18900 个唯一物理 latent 实现，三种处理独立计算共56700次完整处理。1 checkpoint／1源 cache，0教师、0头拟合、0新训练／EMA／归一化更新。父模型原 E20／4440 更新单列。V carrier 可反序列化但不用其特征统计或前向；U 仅不透明角色 ID 排除，不读其 IQ／TX。原模型及默认 cosine6×160、三 SIG_TIME、统一 Sinc、joint160 和 NoExplicitCFO 保持冻结 eval。

## 晋级与放弃条件

只检查固定 MMSE24 相对同实现 NoEQ：源 low urban 和 mid urban accuracy 均至少增加0.5个百分点，源 low suburban 下降不超过0.25个百分点，每个源 RX 的 low urban 下降不超过1个百分点；至少20%的原 NoEQ low urban 错误包具有实际 IQ 变化（相对范数差大于1e−4）。这只是检验接收处理是否触及并改善源身份错误，不把该比例当作训练泛化上限。全部条件同时满足才支持后续独立 E30 匹配训练设计；本 launcher 不自动训练。

任一条件失败关闭本固定 MMSE24 候选，不改选64，不调整 SNR、NMSE、增益上限、延迟、seed、阈值或 RX 子集。MMSE64 指标完整保留，但不参加候选选择。即使通过，也没有证明 clean DG、A_s 训练效果、low urban>60% 或 suburban 接近 clean；后续训练须有同数据、同父模型与预算、同默认头、同 U 规则、同视图前向的控制，并完成原七视图独立 truth-last。

## 验证和执行边界

本地已实际通过10个输入反例、12组耦合合成三视图、基础噪声经 G 一次着色的最大误差0、输入／全局 RNG 不变、物理 ID 批顺序重排一致，以及已知单径 H 的24／64点目标延迟和增益上限检查。0真实 IQ／checkpoint／NN。合成检查不是身份结果。发布前唯一独立 P0/P1 审查、单份 Git 归档传输比较、普通 N607 预检和远端编译按已有最小流程执行；真实默认头 checkpoint no-query smoke PASS 后直接继续源检验。

精确命令、环境／CWD、不可变 run/release、owner、CPU2/CUDA隐藏、输入来源及技术 stop rule 见 [experiment.json](experiment.json)。独立复算只读取固化源 prediction 和标量元数据，重新计算全部 RX／TX／day／state／lock 混淆矩阵与修复／新增错误，无模型或 IQ 前向。正式 target IQ 与 truth 不访问；技术失败保留输出，不自动重发，不修改其他健康任务。

唯一独立 P0/P1 审查 PASS：9 个文件与登记已只读核对，语法及63002前向预算复算通过；审查者未读取真实 IQ／CP／target，未执行 NN 或远端操作。真实源运行已完成。

实际发布代码97e2631c8d500e015bd50f6eef9ae590284c569a已独立核对远端 OID；一次普通账号 preflight、archive传输比较、remote compile VERIFIED。源 PID561505 独立观察存活，CWD／argv／CUDA隐藏／CPU2匹配；真实 checkpoint smoke PASS，进度已从128推进至1792个首场景源包。尚未完成判决，不报告识别收益。初次观察脚本误沿用旧PID547702，已改为本行实际PID并重新独立核对，没有据此重启或误判本行终止。

## 实际完成与固定判决

|源场景|NoEQ Acc／F1（%）|MMSE24 Acc／F1（%）|MMSE24 Δ（百分点）|MMSE64 Acc（%）|修复／新增错误|
|---|---:|---:|---:|---:|---:|
|practical_low_suburban|91.047619／91.003666|92.984127／92.959363|+1.936508|39.492063|164／42|
|practical_mid_urban|73.904762／73.804723|77.952381／77.866436|+4.047619|36.492063|386／131|
|practical_low_urban|61.015873／60.783034|64.111111／63.908368|+3.095238|35.857143|293／98|

raw源准确率98.412698%，属于普通星地CONCAT source-final20的训练源读出，不是正式clean。MMSE24五项固定条件全部通过。独立复算63000个源判决、144份混淆矩阵和60份TX recall，最大指标误差0；完整RX/TX/day/state/lock见[分组结果](stratified_results.md)。low urban原2456个错误中24.470684%具有实际MMSE24变化，超过固定20%要求；全部源RX的low保护条件通过。

三场景各6300条，EQ实际生效数为5987、5030、3543；low urban的2757个unlocked包全部保留，三种处理波形逐值一致，因此没有直接恢复失锁样本。MMSE64的明显下降说明数值参考坐标必须控制，但不能将全部差异仅归因于延迟，有限FIR截断与估计误差仍保留。候选24点在结果前已由原共同延迟固定，没有根据准确率改选。

实际63002 NN、6300唯一L、1CP／1源cache、18900个唯一latent、56700次独立接收处理计算；0新增model epoch／optimizer／teacher／headfit／target，模型全部state_dict不变。worker142.267270s，独立分析另0IQ／CP／NN／更新。全量基础噪声经G一次滤波误差0，全部H、质量和补偿依赖检查通过。原PID561505自然退出，完整completion和日志48→86→121→196字节、无failure／traceback独立核实。

本结果只支持下一项训练假设：保留NoEQ观测，等先验混合现有MMSE24接收处理，能否使同一个F在原固定NoEQ practical上获益。源输入更容易分类不保证训练后泛化更好；61.016%→64.111%不是正式low urban提升。最小训练需与原NoEQ As共用同一source-final20／ownEMA，新增10轮、累计E30／6660更新、同4L前向及U规则，固定终轮自动七视图预测／独立truth-last。本源launcher未自动训练，该新训练run尚未发布。A_g、A_s训练收益与联合目标均未证明。
