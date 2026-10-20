# R27：冻结 prior 低秩卷积合并

状态：FAILED / VERIFIED。CPU 首步检查失败，实际控制器退出，0 GPU worker、0 正式预测；旧产物完整保留，无性能结果。

用户要求压缩同时提升性能。本轮固定 BOTH R22 source-final E20/19700（complex625、complex625_identity，方法 e190cf5367b03da8aafced51276a8730ad8274fa），仅作冻结推理的代数变换，无新训练/选模/教师。它们原始训练仍来自用户指定原 R46 prune_refine350521 E20/4940、方法 bbb8daf817de050b958d217b70d4c33f4aefe341，源7/63/30物理角色及完整祖先契约保持。未来训练唯一初始化/教师仍为原350521，本次不把新学生变成训练父版。历史目标已暴露，属于工程执行比较。

|固定版本|源完整参数|合并后完整部署参数|coarse+action / prior|GPU|
|---|---:|---:|---|---:|
|R22_complex625_priorcollapsed|278180|299297|186703 / 112594|2|
|R22_complex625_identity_priorcollapsed|278180|299297|186703 / 112594|3|

只将三阶段 prior 内的 spatial Conv→1x1 Conv 线性对合并，核为 right[:,:,0] @ left.flatten(1)，保留左层stride/padding/dilation与右层bias。28组共享权重合并后仍按真实对象别名共享；阶段bias、GN4、skip、coarse/action、3stage/6CG保持。共84个prior FactorConv替换为单Conv，数据自由完整前向实际减少84次Conv1d调用。参数从278180增到299297，仍较原350521少约14.61%；ConvLinear MAC增加，不能把调用减少写作FLOPs减少或已测加速。完整部署权重/参数和存储均按合并后模型统计，不仅报告源压缩文件。

线性代数相同不等于FP32逐位相同。CPU控制器首步读取原2条L_s的clean/received01，与原F50/defaulthead冻结执行原源模型、合并模型、严格重载模型，检查有限输出、rtol2e-5/atol2e-6接近度、严格重载输出逐位一致、原源与F状态不变；记录argmax差异但不据源识别收益筛掉版本。每版先导出独立299297权重，0优化器更新/0TX训练标签/0F训练知识，无sourceV或target用于参数更新。

两版均按冻结矩阵自动完成源1800×七景、同卡完整成本、独立源评分，再正式168000×clean/三郊区/三urban。批量128与原R22预测相同；每行全部新预测固定后，先只读复用旧R22的IQ/features/logits和argmax作差异比较，再独立CPU truth-last。旧native目标不重跑，不假定argmax保持，不从结果调参/选择性重测。逐TX/RX/day、confusion、repair/damage、clean损伤、波形/F几何保留；原父逐景0.8pp/1344条、原R17历史1pp与lowurban70%/其他五景距rawclean5pp/clean保持和强TX真实性分开报告。

同卡64sourceV同时测原R17、原350521父、原R22低秩278180与本次299297，覆盖G与G+原F、B1/B32、三轮正反顺序，记录median/p90/p95/吞吐及每轮值、resident/peak/incremental、完整ConvLinearMAC及权重字节。计时含全部物理解算，排除传输/loader/前处理。四G一F是共享suite驻留，不冒充独立部署占用；MAC不含FFT/CG/norm/访存/F。CPU转换开销和实际GPU成本分别记录，无CUDA Graph或native fallback。

N607普通账户，GPU2/3最多2own/每卡1，严格排在R26同卡所属worker终态及实际退出之后。前驱workers为空时等待；再以共享容量锁free≥18GB/allowner<2分配，不停止/修改任何健康owner。CPU评分2线程、R06score.lock、最多1ownscorer。一个方法commit、新不可覆盖run/release/output、一次preflight、一次归档两端SHA、一次compile，发布后以实际PID/CWD/完整命令/log增长核实。技术失败保留影响行，无自动重启/热改/回退，健康行继续。

25项聚焦检查覆盖卷积合并数学、不同stride/pad/dilation/bias、共享核和阶段偏置、全模型参数与调用计数、保存/严格重载/别名冲突、F/来源/训练权限负例、两版七景链、truth-last与原输出保护；唯一审查由/root/review_r14_prior_compact独立复跑25项PASS，未访问真实IQ/F/TX/远端、未编辑文件。主对话缓存归一化斜率，本组改变prior线性执行，独立固定方案。当前无新增正式结果，完整目标ACTIVE。


## 真实 smoke 技术失败与只读诊断

方法 commit 0686ed878fe16c1eb53995749fa9fb41cd1e3371 已推送并独立核对远端 OID，一次预检/归档两端 SHA/compile 通过。真实 CPU 检查将 IQ、方差与冻结 F 的非线性输出共用 rtol2e-5/atol2e-6 逐元素断言，在 F 特征一个近零分量失败（差3.1106174e-6）。实际控制器已退出，尚未提交 GPU worker，无 sourceV/formal 预测、无目标 IQ/truth 访问；不是识别性能负结果。

对同两条合法 L_s 的 clean/received01、两版模型做只读完整诊断：IQ 均通过原阈值，相对 L2 最大7.0742e-7；方差逐位相同。F 特征/默认头输出全部有限，特征最大绝对差6.5566e-6、logits最大绝对差4.8280e-5，两条样本的argmax差异均0。原源模型及F状态不变，0优化器/标签/V/target访问，远端写入0。该少量样本不证明正式分类保持。

本地已复现“近零 F 分量触发额外等价断言”的边界，修复保持 IQ/方差原阈值、全部输出有限/形状/正方差以及严格重载四输出逐位一致；F 数值与argmax差异完整记录，不作源筛选或等价门槛，由每版完整七景正式测试评价。唯一一次原问题定向复审 PASS，新增10项检查独立复跑PASS；没有重复全审查或放宽IQ阈值。

替代 run 为 20261021-phase1-restoration-R46-prior-collapse-s2026092701-m2-r28，使用新 release/output；本R27不重启、不覆盖，原失败与导出权重保留。完整研发目标保持ACTIVE。
