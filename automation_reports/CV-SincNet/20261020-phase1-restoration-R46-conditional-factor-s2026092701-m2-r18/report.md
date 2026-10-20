# R18：35.05万基底的保通道低秩压缩与条件先验匹配

状态：VERIFIED / RUNNING，实际两版首轮训练及日志增长核实，尚无正式性能结论。

用户明确从R46 `prune_refine`完整350521参数继续优化压缩，并要求与另一个对话不同。另一个对话R48保留350521优化对照并裁剪通道到273765；本轮保持全部通道数、GN4统计集合、skip、独立bias/norm、三步prior与六次CG，只将saving kernel>1 Conv1d做输出通道SVD625分解。两行均完整278180参数，其中context186703、prior91477，比指定父减参20.6381%。不以结构更小宣布识别或速度优化成功。

|固定行|完整参数|条件prior NMSE权重|GPU|
|---|---:|---:|---:|
|factor625_stage|278180|0（不执行额外teacher条件前向）|2|
|factor625_cond|278180|0.05|3|

两版结构、初值、源顺序及训练预算完全相同，唯一方法差异是条件函数约束。基础目标为pairedclean最终IQ NMSE +0.05三尺度相对功率log1p STFT误差 +1→0.1的三stageIQ教师误差；conditional版额外将实际student各stage先验输入detach后送入同一阶段冻结parentprior，匹配studentprior输出，以teacherprior每例能量clamp1e-4归一化，三阶段平均。Teacher输入/输出均停止梯度，student原路径保留梯度；对照跳过额外条件前向，额外计算不隐去。推理不携带teacher或条件损失。

唯一直接初始化及IQ教师为R46 prune_refine固定source E20/4940，method bbb8daf817de050b958d217b70d4c33f4aefe341；不从其他新目标分数挑权重。复用其完整R44/R38/R17/R07/R04原7/63/30源物理ID、READY/EXACT_MATCH数据契约；核对实际launch、plan、source训练完成和checkpoint元数据。用户在其七景结果揭示后指定父，属于已暴露工程基准研发，不声称新的独立盲测或干净泛化。F-free、TX无标签不等于无clean：合法源模拟配对clean参与目标，0TX标签/F权重或输出/V/target用于训练。

每版固定新增E20、B64/micro64、每轮985/共19700更新，received01..20；6300Ls/56700Us每轮，共L126000/U1134000曝光；全部coarse/action/prior联合优化。ResetAdamW2e-4、warm985、cosine2e-5、WD1e-4、clip1、FP32/TF32关闭，尾批24实际归一化。组件继承80→累计100、和300；串行pass120→140。与其他对话B256/4940不同，优化更新成本更多，不称同计算预算。

每版固定finalE20/19700，自动source1800×7诊断、matched成本、独立源评分、target168000×7预测与独立CPU truth-last；所有七视图预测固定后才连接truth。原F50 seed2026092701/defaulthead冻结，无新头。任何源负收益也不取消正式测试；无早停、跨行候选筛选、目标反馈调参或选择性重跑。逐TX/RX/day、clean损伤、净纠错、波形/FFT和完整20轮曲线保留。当前指定350521逐场景0.8pp（最多少1344正确，500×损失≤4×168000），原R17历史1pp、lowurban≥70和其他五景距rawclean≤5pp分别报告；不是性能优化目标的替代。

同卡sourceV64条，G及G+冻结原F，B1 warm16/64次×3轮、B32 warm4/16次×3轮；候选正反顺序交替。报告median/p90/p95、逐轮中位、吞吐、驻留/峰值/增量显存、完整参数/文件字节/ConvLinearMAC。输入已在GPU，时间不含传输/预处理/loader；驻留是三G加一F共享suite，增量峰值对应当前前向。MAC不含FFT/复数CG/norm/激活/访存/F；实际时延包括全部context/3prior/6CG，不预设加速。

GPU2/3每卡一本任务，总最多2，先确认本对话R17旧actual owned PID退出，再按free≥18GB和每卡全owner<2占用。CPU2/R06锁max1own scorer。另一对话GPU0/1和所有健康任务不停止、不热改。普通账户、全新不可覆盖run/release、一次预检、一次归档两端SHA及compile、launcher首步真实R46/F50/2Ls无query smoke PASS后直接训练。技术失败保留影响行，健康行继续，不自动重试或回退。唯一P0/P1审查限下次真实run正确性；精确命令/环境/CWD/路径/停止规则见experiment.json。


真实发布已核实：method `aca99499d3f0170df71380f6429ac9a01ef412f8`，30项聚焦检查和唯一独立P0/P1审查PASS，归档两端一次SHA一致、compile通过。真实父/F50/2Ls无query smoke PASS，两版GPU2/3实际PID/CWD/argv/commit、GPU对应及首轮日志增长逐项VERIFIED。另一对话0/1不变。
