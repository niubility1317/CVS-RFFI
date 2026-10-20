# R19：性能优先的物理上下文保持与prior低秩压缩

状态：VERIFIED / FORMAL_TEST_RUNNING，两版20轮训练、全部源诊断与完整成本测量完成；正式七场景测试进行中，尚无最终正式结论。

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


## 实际GPU训练核实（2026-10-21）

普通账户实际控制器与两条worker的PID/CWD/完整命令/方法commit已匹配，worker分别在物理GPU2和GPU3，日志均继续增长，未见failure或Traceback。prior625_cond已完成12/20轮，prior750_cond已完成6/20轮；这是一次训练进度快照，不是最终成绩。R18的对应实际owned GPU句柄已退出后才接续，两个组同时最多2条本任务GPU worker。两版继续固定最终轮次后完整七场景预测及独立truth-last，健康训练保持。此前QUEUE记录为发布时历史快照。


## 源训练与实测成本完成快照（2026-10-21）

两版20轮、4940更新与各1260000源IQ曝光已逐条核对，共40条训练记录；coarse/action实际更新0、prior更新4940，F/TX/V/target训练读取均0。两版source1800×7预测及独立源评分完成，正式测试仍在进行，本节不含目标成绩，不作识别性能结论。

|版本|完整参数|较350521减参|ConvLinear MAC/IQ|checkpoint字节|完整G+F B1中位ms（父→新版）|B32吞吐条/s（父→新版）|
|---|---:|---:|---:|---:|---|---|
|prior625_cond|329404|6.024461%|33858988|1630564|56.583522→60.931619|466.074572→484.438950|
|prior750_cond|343974|1.867791%|37364908|1672898|53.945618→56.918391|474.955965→468.891229|

父版ConvLinear MAC为39081004、checkpoint1686812字节。每版均以同卡64sourceV、三轮交替顺序测量；完整G包含原context/三prior/六次CG，G+F包含原默认头。输入预加载GPU，不含传输/预处理/loader；MAC不含FFT/复数CG/norm/激活/访存/F；共享suite驻留不称独立部署显存。

prior625_cond：总训练915.298846s，每轮中位45.977131s。全部loss首→末：iq_nmse 0.0545602837→0.0250384813；teacher_stage_nmse 0.0246153532→0.00244777892；spectral_error 0.0132604995→0.00657971356；conditional_prior_nmse 0.748820508→0.0848619108；loss 0.117279688→0.0320583416。完整stdout异常类别：[]。

prior750_cond：总训练897.982139s，每轮中位44.868146s。全部loss首→末：iq_nmse 0.03540548→0.0227818606；teacher_stage_nmse 0.00912655639→0.000771190823；spectral_error 0.00885817831→0.00601673632；conditional_prior_nmse 0.313594523→0.0289053987；loss 0.0606546715→0.0252991581。完整stdout异常类别：[]。

32.94万版G+F B1中位延迟增加7.6844%、B32吞吐增加约3.94%；34.40万版分别增加5.5107%、下降约1.28%。减参未带来单条加速；两卡绝对时间不可用于比较rank机制。训练时间较R18降低同时涉及冻结context、B256及4940更新，不能单归因于网络变小。最终识别结论必须等待逐版全部七景固定预测及独立truth-last。
