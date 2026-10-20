# R28：冻结 prior 低秩卷积合并

状态：VERIFIED_QUEUED_AFTER_REAL_SMOKE。新R28已发布并通过真实2Ls检查，两份完整299297权重导出/严格重载；等待R26实际进程完成，0新GPUworker，无新版正式分数。

用户要求压缩同时提升性能。本轮固定 BOTH R22 source-final E20/19700（complex625、complex625_identity，方法 e190cf5367b03da8aafced51276a8730ad8274fa），仅作冻结推理的代数变换，无新训练/选模/教师。它们原始训练仍来自用户指定原 R46 prune_refine350521 E20/4940、方法 bbb8daf817de050b958d217b70d4c33f4aefe341，源7/63/30物理角色及完整祖先契约保持。未来训练唯一初始化/教师仍为原350521，本次不把新学生变成训练父版。历史目标已暴露，属于工程执行比较。

|固定版本|源完整参数|合并后完整部署参数|coarse+action / prior|GPU|
|---|---:|---:|---|---:|
|R22_complex625_priorcollapsed|278180|299297|186703 / 112594|2|
|R22_complex625_identity_priorcollapsed|278180|299297|186703 / 112594|3|

只将三阶段 prior 内的 spatial Conv→1x1 Conv 线性对合并，核为 right[:,:,0] @ left.flatten(1)，保留左层stride/padding/dilation与右层bias。28组共享权重合并后仍按真实对象别名共享；阶段bias、GN4、skip、coarse/action、3stage/6CG保持。共84个prior FactorConv替换为单Conv，数据自由完整前向实际减少84次Conv1d调用。参数从278180增到299297，仍较原350521少约14.61%；ConvLinear MAC增加，不能把调用减少写作FLOPs减少或已测加速。完整部署权重/参数和存储均按合并后模型统计，不仅报告源压缩文件。

线性代数相同不等于FP32逐位相同。CPU控制器首步读取原2条L_s的clean/received01，与原F50/defaulthead冻结执行原源模型、合并模型、严格重载模型，检查全部输出有限/形状/正方差、IQ与方差保持原rtol2e-5/atol2e-6接近度、严格重载全部输出逐位一致、原源与F状态不变；完整记录F特征/logits接近度和argmax差异但不据源等价或识别收益筛掉版本。每版先导出独立299297权重，0优化器更新/0TX训练标签/0F训练知识，无sourceV或target用于参数更新。

两版均按冻结矩阵自动完成源1800×七景、同卡完整成本、独立源评分，再正式168000×clean/三郊区/三urban。批量128与原R22预测相同；每行全部新预测固定后，先只读复用旧R22的IQ/features/logits和argmax作差异比较，再独立CPU truth-last。旧native目标不重跑，不假定argmax保持，不从结果调参/选择性重测。逐TX/RX/day、confusion、repair/damage、clean损伤、波形/F几何保留；原父逐景0.8pp/1344条、原R17历史1pp与lowurban70%/其他五景距rawclean5pp/clean保持和强TX真实性分开报告。

同卡64sourceV同时测原R17、原350521父、原R22低秩278180与本次299297，覆盖G与G+原F、B1/B32、三轮正反顺序，记录median/p90/p95/吞吐及每轮值、resident/peak/incremental、完整ConvLinearMAC及权重字节。计时含全部物理解算，排除传输/loader/前处理。四G一F是共享suite驻留，不冒充独立部署占用；MAC不含FFT/CG/norm/访存/F。CPU转换开销和实际GPU成本分别记录，无CUDA Graph或native fallback。

N607普通账户，GPU2/3最多2own/每卡1，严格排在R26同卡所属worker终态及实际退出之后。前驱workers为空时等待；再以共享容量锁free≥18GB/allowner<2分配，不停止/修改任何健康owner。CPU评分2线程、R06score.lock、最多1ownscorer。一个方法commit、新不可覆盖run/release/output、一次preflight、一次归档两端SHA、一次compile，发布后以实际PID/CWD/完整命令/log增长核实。技术失败保留影响行，无自动重启/热改/回退，健康行继续。

25项聚焦检查覆盖卷积合并数学、不同stride/pad/dilation/bias、共享核和阶段偏置、全模型参数与调用计数、保存/严格重载/别名冲突、F/来源/训练权限负例、两版七景链、truth-last与原输出保护；唯一审查由/root/review_r14_prior_compact独立复跑25项PASS，未访问真实IQ/F/TX/远端、未编辑文件。主对话缓存归一化斜率，本组改变prior线性执行，独立固定方案。当前无新增正式结果，完整目标ACTIVE。

R27原run在CPU首步F近零分量等价断言失败，已核实实际退出且0GPU/0target；诊断IQ原阈值均通过，源/F状态不变。新R28只修复F诊断边界，不改变变换/数据/源模型/矩阵/GPU/预算，不覆盖旧失败root；详见R27原报告。


## 实际发布、排队与独立交付

方法commit 7ca5af4f6f5d9f862e07e13b9da15902379044e4已推送并独立匹配远端OID。一次N607预检核实原350521父及两份R22 source-final20/19700、原数据READY/EXACT_MATCH与七景测试句柄；一次归档两端SHA及compile通过。实际release/CWD为r28，未复用旧R27失败root。

真实2Ls clean/received01检查PASS：两版合并后完整299297参数，28组共享核、84个prior factor模块已转换；IQ/方差原阈值通过，全部输出有限、正方差、严格重载四输出逐位一致，源模型与F状态不变。完整保留F features/logits数值差异，当前两条源样本argmax均一致；这不是正式分类保持结论。实际CPU控制器PID/CWD/完整命令/commit/日志增长VERIFIED，两版都等待R26同卡终态及实际退出，0新GPUworker，不改动其他健康任务。

结果工具新增22项检查PASS，覆盖两版全七景、原预测一致性与混淆矩阵的相互核对、源/F权限边界、真实进程退出、完整299297部署参数、四模型成本及缺失/错误证据拒绝、公开逐TX/RX/day CSV和报告输出。仅以聚合artifact独立复算，不读新的IQ/truth或调参。

单一本地完成owner只读等待实际controller/worker退出，之后收集完整source7/formal7/原预测差异/独立truth-last，取回两份合并后完整权重并CPU严格加载，检查组件参数/MAC/有限输出，再报告、提交推送并独立核对远端OID。没有重试、热改、目标重测或复制完成owner；原R23/R24/R26完成owner保持。完整70%/5pp/clean/TX真实性目标仍ACTIVE。
