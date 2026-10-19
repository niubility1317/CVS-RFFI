# 末端时域与频域有界残差滤波

状态：RUNNING / VERIFIED。目标仍为完整目标测试集 low urban ≥70%，其余五 practical 接近原 F clean 85.113889%（Agent 暂采用≤2个百分点口径）。分别报告同物理 clean 的波形差距和接入三套冻结原 F/defaultheads 后的真实 TX 识别、修复/损坏/净修复。当前已测最好 low urban 为65.309127%，不把新结构、源域分数或波形代替目标成绩。

源域依据：对已有1800条source V的固定预测做事后误差分解，没有新增模型前向、训练更新或目标访问。low urban原R03 NMSE0.068199，posterior03与两种时间注意力约0.07486至0.07490；误差分布在多个有效频带，且精修对各频带普遍增加误差，不能断言某一个频带是原因。原coarse已有频谱校正，不能把本轮称为首次加入FFT。完整聚合分解在source_error_profile.json，不含原IQ或逐样本数据。本轮只检验新增末端滤波器的基底，配置不根据目标成绩调整。

两行time_basis/frequency_basis各自从原合法R03（coarse20/correction20，method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）严格继承全部旧state，均不加载失败精修。末端加入同Core4→4、宽16/32/48/64的有界广义线性残差滤波器，新增210628参数，总2351468，张量及初值匹配。输出投影为0；先计算相对identity的delta，再应用并加回原恢复IQ，使频域臂初始化也逐点等于原R03，无FFT往返舍入改变初始输出。

time臂从当前received和R03输出的I/Q预测逐时刻系数，delta-main乘原输出、image乘其共轭；frequency臂从两者完整256点orthoFFT并fftshift的I/Q预测逐频点系数，ifftshift后delta-main乘原输出频谱、image乘共轭镜像频谱，最后ifft得到残差。主通道幅度限定0.5至2、相位±π/2，镜像实/虚各±0.25（复数幅度≤sqrt(2)/4）。仅改变基底，不能把两种算子称为数学等价。该形式限制自由度，不保证物理信道逆或身份保真；频域每条IQ多3次256点变换，不把同参数说成同计算量。推断只读当前IQ，一次coarse/correction/filter，无运行F条件、信道oracle、RX/场景、clean、其他帧、迭代或状态更新。

匹配原63000 L/U、R02缓存epochs1..10、.15clean/.60residual/.25full路由、observed＋clean等权、batch256、modelseed2026092701、dataseed2026101951。固定joint E10/2470，每臂新增L63000/U567000物理优化曝光、1260000IQ，原coarse/correction各累计30轮、60原组件epoch、串行继承50pass，新filter单独10轮。重置AdamW，旧网络LR3e-5、新filter2e-4，warm247/cos至各自0.1、WD1e-4、clip1、FP32/TF32关闭。记录两LR、滤波调用/IQ/系数位置/FFT次数、输出权重范数、encoder delta及真实非零梯度步数；零输出初始化首步encoder梯度0是预期，后续梯度与AdamW decay分开验证。

损失保持posterior03：paired IQ NMSE＋0.1logpower＋三套原F各自final160 cosine＋0.3×T²KL（T=2）的均值，无新增CE/prototype/focus/style/FiLM。原F50 CROSS_RX三seed2026092701/02/03与原默认头（method4df7b94d0fd8ce29056bd8236364e45afa92e07f）参数和统计冻结，0新头拟合、0L/U TX真值。称冻结F引导自监督恢复，不称整个系统全无标签；复用已验证同数据契约和父来源，不重验未变化数据或增加哈希链。

source V原1800条、原F1、七视图仅诊断，无早停、资格/NMSE gate、候选排序或状态更新。两行固定最终E10都自动完整测试原168000物理目标clean＋六practical/三原默认头，各自argmax，不投票或按环境拼接。每行七视图IQ/ID/预测全部固定后，独立无CUDA CPU才连接truth与配对clean；source/target scorer共享score.lock。目标成绩不回流任何行配置、预算、权重、顺序或选择性重跑。报告逐F/TX/RX/day accuracy/MacroF1、净纠错、距原clean差距，以及不做oracle增益/相位/时延对齐的full256/prefix232/tail24 NMSE与logFFTpower均值/中位数/P90、耗时/显存。一个恢复器seed和三原读出不冒充三个恢复器seed、新盲测、Phase2或在轨验证。

46项聚焦检查PASS；单次独立P0/P1审查PASS，无确定P0/P1；审查者独立复跑46项聚焦检查通过。launcher首步实际原R03/三F/source Ls四条（RX1/3各2条）no-query smoke：初始完整IQ逐点等于R03、初值匹配、coarse/correction/filter.out及原头到IQ梯度有效、F无梯度、新checkpoint往返；0V/0TX/0target/0optimizer，PASS立即训练，无smoke令牌。一次N607普通账号预检VERIFIED，原R03合法、前序完整、新run/release不存在、GPU4/5各24124MB空闲无PID、空间≥8GB，其他owner保持。GPU4 time/GPU5 frequency，每卡一条本run worker且计入allowner<2，source≥18GB/target≥12GB；CPU2线程/scorer最多1。实际source进程退出后同卡接续target，不等待另一行。

停止规则：非法来源/权限/代码、冲突、非有限/OOM、预算/新模块未执行或预测失败时保留产物，技术失败该行、无自动重试，健康行继续；不停止/重启/热改其他任务。一个代码配置commit、不可覆盖root、一次归档本地/远端SHA比较、远端compile/import，随后核实PID/CWD/命令/GPU/日志增长。命令、环境/CWD、路径、矩阵、资源和预期产物见experiment.json；原IQ、权重、逐样本预测、日志和私有运行状态不进Git。另一用户对话只读确认当前为解析观测量损失/展开结构路线，未发消息或改动其任务。


## 实际发布与首轮核验

方法commit `c68588016e6960e48ef3b320c8e5daeefc8f16ac`已推送并独立核实远端OID。一次发布归档本地/远端比较、远端compile/import及真实原R03/三F/四Ls no-query smoke均PASS；初始完整IQ逐点等于R03、两臂初值相同、coarse/correction/filter.out及原头到IQ梯度有效、F无梯度、新checkpoint往返一致，0V/0TX/0target/0optimizer更新。两臂实际2351468参数，新filter210628；烟测8条IQ的系数位置2048，time无额外FFT/frequency共24次256点变换。PASS立即训练。

两条实际source worker的PID/CWD/完整命令/GPU4/5及持续日志增长已核实，三F参考缓存各63000条。新filter.encoder已有实际非零梯度，独立于AdamW decay参数变化：

- time_basis：1/10轮、247/2470步，filter调用247次、IQ126000条、系数位置32256000、额外FFT0次；encoder非零梯度246步、输出权重范数0.00205451、encoder delta0.04005143。
- frequency_basis：1/10轮、247/2470步，filter调用247次、IQ126000条、系数位置32256000、额外FFT378000次；encoder非零梯度246步、输出权重范数0.00431857、encoder delta0.02758108。

首轮核验时尚无新源V或目标准确率。新模块实际参与不代表性能有效；两行固定E10后各自自动完整目标测试，source V不作资格筛选，用户完整目标仍未达成。后续最终状态与结果以末尾完整结果节为准。
