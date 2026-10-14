# 软身份关系源诊断：固定方案未通过

XR_GRAPH未通过预定源梯度兼容性条件，不扩02/03，不发布完整训练。源teacher软边真实同TX质量为98.697450%，超过预定90%；但t1/t2五批平均关系导数与同strong图默认头CE的余弦为−0.262721／−0.330844，未达到0。机制实际生效、源边质量较高，与身份CE兼容是两项不同条件。本次没有新目标性能结果，也不能由final50训练L代理断言所有关系学习方法无效。

运行`20261015-phase1-source-soft-relation-manysig-s2026092701-r01`，实现`5ab6d390d5306a787c0efbf4a1532df861f0b877`，执行`e2242a9201afe4bf896305cc3d364f069f677a65`。普通N607的固定DRO首SIG_TIME模型2026092701 naturalfinal50和同checkpoint自身EMA，source_proof核对原scratch50／33+17／11100及源契约。部署网络195041参数、原joint160及6×160默认cosine头保持。没有模型优化或专门训练分类头，没有新网络、投影、教师或头。

## 同批源结果

固定五个L180共900唯一物理ID，源RX1/3/4/6/8及days1/2/3；两个模式共享同一teacherweak和studentstrong图，学生沿用原噪声0.015／train dropout。原teacher默认头正确899/900，学生strong默认头正确898/900。教师q只定义detached软边，纯关系函数不接收TX标签；L真值在边固定后用于诊断身份质量和默认CE。未访问U IQ或U真值，V返回即丢弃、0前向0梯度，0query或target前向。

XR每cohort25920条允许的有向跨RX边，ALL为32220条非自边；900行均有效。XR真实身份软边质量各cohort98.498302／98.583370／98.552737／98.818251／99.034590%。这是训练L上的教师边质量，不是U伪标签准确率。平均strong CE=0.097514684；未加权XR KL=1.621025276，有效系数按预登记固定0.16。

| 实际参数 | ALL有效导数范数 / CE范数 | ALL对CE余弦 | XR有效导数范数 / CE范数 | XR对CE余弦 | XR非零cohort |
|---|---:|---:|---:|---:|---:|
| t1.pw.weight | 3.435226 | -0.263262 | 3.394885 | -0.262721 | 5 / 5 |
| t2.pw.weight | 3.981038 | -0.331290 | 3.936461 | -0.330844 | 5 / 5 |
| 0.local_projection.weight | 0.515288 | 0.001669 | 0.511427 | 0.003583 | 5 / 5 |
| id_backbone.cls_head.weight | 0.000000 | 未定义 | 0.000000 | 未定义 | 0 / 5 |

表中导数先在五个等大小cohort求向量均值，再计算范数／余弦；不是逐cohort比值的均值，不是优化器位移或累计贡献。t1／t2／SIG的XR梯度各5/5批非零；默认W的关系梯度为None，按零向量保存，其CE梯度非零。W没有被隐藏冻结来制造该独立性检查。学生/教师参数与buffers逐元素不变、无累积.grad；私有strong随机状态完整恢复。

## 复核、成本和后续

单归档传输比较、远端编译、真实checkpoint无query smoke通过；启动时PID/CWD/argv与提交记录匹配，日志从0增长至938字节，之后自然完成、技术failure为空。5个cohort的图、softmax目标、KL、同图CE、900唯一ID、参数向量坐标、边数量与预定资格在第二个独立进程中复算VERIFIED。完整结果在 [results.json](results.json)，复算在 [independent_artifact_recount.json](independent_artifact_recount.json)，发布与结束状态在 [launch_verified.json](launch_verified.json)。

总计1800源包×模型前向／10批，5梯度批／20参数层／60保存向量，0训练轮次、0优化／headfit。原生诊断计时2.492975秒，不含完整发布、预检、数据传输和后续独立复算；原PT／源ID／checkpoint／日志留原不可变run，不入Git。无GPU实验分配，未停止或修改其他健康任务。

保留FAIL和全部对照，不扫描温度、系数或换父，不因sourceFAIL创建另一个近似配方。本结论局限于预定final50源L代理；不证明训练早期或U上的梯度，也没有DG、84%、新TX、卫星或Phase2声明。当前RSC固定02/03六新仍按原50轮运行，首种子负结果不用于改动它们。
