# R19：性能优先的物理上下文保持与prior低秩压缩

状态：VERIFIED / QUEUED，真实smoke已通过，实际CPU控制器等待R18逐卡交接；未开始GPU训练，无新版正式结论。

本组从用户指定R46 prune_refine350521继续，性能优先。主对话进行GN4通道裁剪，本对话R18进行全链低秩与条件先验匹配；R19保持所有通道/GN/skip，进一步保留原237927参数的coarse/action完整函数并冻结，只对prior的saving kernel>1卷积做SVD分解。两个比例固定为.625/.75，不从新目标结果选择父或rank。推理仍自带完整context/三步prior/六次CG，训练教师不参与部署。

|固定行|完整参数|可训练prior|冻结context|较350521减参|GPU|
|---|---:|---:|---:|---:|---:|
|prior625_cond|329404|91477|237927|6.024461%|2|
|prior750_cond|343974|106047|237927|1.867791%|3|

研究依据是R18截至13/12轮全部25条源训练记录与当时完整stdout：teacherstageIQ误差由首轮约.052降至约.004，仍有函数拟合缺口，初期coarse/action梯度敏感性较大。该证据只支持“保留物理context再检验prior压缩”的假设，不证明context导致TX识别损伤。R18当时0正式预测，R19矩阵在其目标访问前固定；主对话新目标分数不用于本组配置。全部源证据保存于私有source_basis.json，未读额外IQ、F、TX真值或目标成绩。

两版独立从原R46 source-final E20/4940初始化及作IQ教师，方法bbb8daf817de050b958d217b70d4c33f4aefe341；不继承R18或主对话新学生。原7/63/30物理源角色、6300Ls/56700Us、R07 received01..20、READY/EXACT_MATCH及完整上游R44/R38/R17/R07/R04来源核对复用。350521父由用户在旧目标结果揭示后指定，属于已暴露工程基准，不称新的独立盲测或干净泛化。恢复器训练0TX标签/F权重输出/V/target；实际使用合法模拟pairedclean及IQ教师，应称TX无标签源自监督，不称无clean盲恢复。

统一目标：最终pairedclean IQ NMSE +.05三尺度相对功率log1p STFT误差 +恒定1的三stage教师IQ NMSE +.05条件prior相对NMSE。条件teacher在实际studentprior输入detach后前向，输出detach，以每例teacherprior能量clamp1e-4归一化、三stage平均。原teachercontext每例只计算一次并供两条prior轨迹共享，studentcontext额外前向0；独立推理改为自带完全相同的context，测试已核对两种路径逐位一致。teacher和context参数不更新，只有完整共享prior有梯度；保留stage独立bias/norm和共享factor左右权重。

每版固定新增20轮/4940更新，global256/micro256、每轮247、尾批24实际归一化，L126000/U1134000/源1260000曝光，额外条件teacherprior3780000例。prior-only ResetAdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/TF32关闭，复用已验证原始float32 buffer。与R18的B64/19700更新不同，不能称相同优化成本；源曝光和20轮预算不缩减。coarse80/action80/prior100，组件轮次和260，祖先串行120+新增20=140；F50训练另计。

每版固定最终E20/4940，自动source1800×7、同卡完整成本、独立源评分、formal168000×7及独立CPU truth-last；每行全部七视图预测固定后再连接truth。无源门槛/早停/跨行筛选，负结果也全部测试。原F50 seed2026092701及默认头冻结，不拟合新头。逐景accuracy/MacroF1、TX/RX/day、净纠错、clean损伤、波形/FFT及完整训练曲线全保留。父350521逐景0.8pp/最多损失1344正确是上限，性能提升优先；原R17历史1pp、lowurban≥70、其他五景距rawclean84.6761904762≤5pp分别报告，不用减参宣称成功。

同卡64sourceV，完整G和G+原冻结F，B1 warm16/64次×3轮、B32 warm4/16次×3轮，候选顺序正反交替，报告median/p90/p95/各轮median/吞吐/共享suite驻留/峰值/增量内存/完整参数/权重字节/ConvLinearMAC。预加载GPU输入，不含传输/预处理/loader；MAC不含FFT/复数CG/norm/激活/访存/F。三G加一F共享驻留不称独立部署内存；并存负载影响计时，参数减少不预设推理加速。

GPU2/3分别接续R18对应实际owned PID/argv退出且原行终态后运行；R18+R19总最多2本任务GPU workers、每卡1，free≥18GB/每卡全owner<2再次检查，其他owner保持。R06共享CPU score.lock、CPU2、max1own scorer。新不可覆盖run/release/output，普通账户，一次preflight/归档两端SHA/compile；launcher首步真实父/F50/2Ls noquery smoke PASS即继续。技术失败保留影响行，不自动重试、热改、回退或误杀；健康行继续。25项聚焦检查及唯一独立P0/P1审查PASS。精确commit、命令、环境/CWD、路径、矩阵和停止规则见experiment.json。


真实发布核实：method `3ac37ca1138f266f5fd91f74e7feca0eb11911cc`，归档两端一次SHA一致、compile通过；原R46/F50/2Ls真实noquery smoke PASS，coarse/action实际梯度0、三prior梯度有限非零、父不变及独立权重重载一致。CPU控制器PID/CWD/argv/commit/日志增长VERIFIED，当前0新GPUworker，原R18继续。25项执行检查和唯一独立P0/P1审查、13项结果计算检查PASS。
