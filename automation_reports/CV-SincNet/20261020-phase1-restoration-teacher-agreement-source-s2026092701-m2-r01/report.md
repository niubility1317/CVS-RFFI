# 冻结三 F 的帧级与复采样点级梯度一致约束

状态：PLANNED。用户目标为完整目标 low urban≥70%，其余五 practical 接近原 F clean85.113889%。“接近”暂按≤2个百分点，是 Agent 的操作解释。登记时最好 low urban66.384127%，目标未达到。

源代码依据：已有三F引导先平均语义loss再求IQ梯度，旧日志未保存逐F梯度冲突；此前teacher mean/worst是标量聚合比较，不是IQ方向一致性比较。本轮检验是否可以减少教师特有的相互冲突方向，不声称已从旧日志观察到冲突，也不用当前局部敏感度候选的目标结果制定方案。

|固定行|反向语义梯度|执行|
|---|---|---|
|既有per_frame_cap|直接mean3F，再逐帧cap|只复用完整历史结果|
|frame_agreement|整帧512维三对dot均≥0才保留mean3F|新GPU4|
|time_agreement|每时刻I/Q二维三对dot均≥0才保留mean3F|新GPU5|

三个F逐个计算相同final160 cosine＋0.3T²KL（T2）对恢复IQ的sum-loss梯度。每教师保留独立图元组，避免从loss stack切片时对其他图作零梯度遍历；未归一化教师梯度、未拟合权重。三对dot阈值固定0：全非负保留未加权mean3F，否则identity梯度置0。time行始终把I/Q当二维复采样向量，禁止逐I/Q轴signmask。agreement operator对教师置换、正交I/Q坐标变换不变，不将此性质扩大为整个恢复模型的相位等变性。

零dot接受；零norm教师对的cosine不进入诊断均值，不改变mask。记录总/agreed/activeagreed单位、三对负dot计数/有效cosine数及和、raw/agreed norm²、各教师对保留方向的负dot数及最小值。帧行1260000单位，复采样点行322560000单位；零梯度位置不能当有效一致引导，另列activeagreed。数学上保留identity方向与各教师局部梯度的dot非负；加入wave、AdamW/clip/有限步后不保证各损失或最终目标改善。此为自定义算子，不称AND-mask/PCGrad复现。零一致或零cap不作为资格失败。

随后复用原per_frame_cap：gw＋s×agreed，s=min(1,sqrt(norm²gw/norm²agreed))，i2≤1e-20取1，wave0且identity非零取0；停止scale梯度，除实际observed+clean数量一次，再完整网络反传/clip1/AdamW。标量日志仍为IQNMSE＋0.1logpower＋mean3F(finalcos＋0.3T²KL)，自定义反向不冒充其普通标量梯度。两新行只改变一致性的粒度。

均从原合法R03（methodb01b534f52daf9cae9e940ed6ec6f9243bccbdbe，coarse20/corr20）完整相同state独立开始，2140840参数，无新增NN/头，不继承失败精修、control、局部J或另一对话权重。三原F50 CROSS_RX/defaulthead（2026092701/02/03，method4df7b94d0fd8ce29056bd8236364e45afa92e07f）参数/统计冻结，0L/U TX真值、0F更新、0新头拟合。

原R02 source63000 L6300/U56700、缓存epochs1..10、.15clean/.60residual/.25full、observed+clean、batch256、modelseed2026092701/dataseed2026101951保持。jointE10/2470，L63000/U567000优化曝光、1260000 IQ；coarse/corr各累计30轮、60组件epoch、串行继承50pass。AdamW3e-5/warm247/cos3e-6、WD1e-4/clip1/FP32/noTF32。推理仍原网络单帧received IQ，无运行F/梯度/一致性、clean、真实信道/TX/RX/场景、其他帧或更新。

复用控制按operator结构固定为gradient_balance/per_frame_cap（method9866f24dab7097294d585069bfcfe529d3346ade），不按其目标分数选择，不从它初始化，不重训/预测。数据/父权重/参数/预算/原loss/逐帧cap一致；逐教师autograd调用与梯度存储工作不同，如实报告耗时/显存，不预设三倍FLOP或速度。

source V1800/原F1/七视图只诊断，无源分数/NMSE gate、早停或排序。两新最终E10都自动完整目标168000、clean＋六practical、三原默认头各自argmax；各行七视图全部IQ/ID/预测固定后，独立无CUDA CPU才连接truth/pairedclean。分别报告逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净纠错、距clean差距和同物理full256/prefix232/tail24 NMSE/logpower。无按环境拼接/投票，单恢复器seed不冒充三次恢复训练、新盲测、Phase2或在轨性能。目标不回流参数/预算/顺序/选择性重跑。

54项聚焦检查PASS，唯一一次独立P0/P1审查PASS并独立复跑54项。审查后独立图tuple调用优化本地再验证54项，不增加重复通用审查。launcher首步实际R03/三F/四Ls no-query smoke，验证完整初始IQ/state、真实逐F/合并梯度、native posterior到IQ梯度、F无梯度及checkpoint往返，0TX/V/target/optimizer，PASS立即训练。

一次N607普通账号预检VERIFIED：原R03合法、新root/release不存在、GPU4/5各24124MB空闲无PID；局部敏感度source在GPU6/7实际PID/CWD/cmd LIVE并保留。每卡一个本owner、allowner<2，重叠最多4个本owner GPUworker，source≥18GB/target≥12GB，CPU2线程/scorer最多1。新source评分等局部敏感度run complete/failure后，再与本run target共用score.lock；source实际退出才同卡target。非法输入/权限/代码、冲突、非有限/OOM、预算/梯度执行或预测失败则保留并失败该行，无自动重试，健康行继续。

一个实际代码配置commit、不可覆盖root、一次归档本地/远端SHA比较及remotecompile、真实smoke、PID/CWD/cmd/GPU/日志增长随后核实。精确命令、环境/CWD、路径、来源/数据契约、seed、矩阵及期望产物见experiment.json。IQ/权重/逐样本预测/原始日志和私有机器状态不进Git，没有给另一用户对话发送指令。
