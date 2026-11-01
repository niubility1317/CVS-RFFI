# 冻结源模型：等价信道损失的前向与反向计时

**该固定源批次测区的中位wall减少7.496589%，区域峰值allocated显存减少4.008942%。** 10对计时中factor有7对更快、3对更慢。范围是同步前向/反向/有限性与求解检查/clip，不含optimizer、全量数据IO和epoch指标日志；没有完整30轮同质量提速或识别收益结论。

![计时与区域显存](E:/type10-7/automation_reports/CV-SincNet/20261102-diagnostic-Restore-factorized-source-cost-s2026092701-r01/source_cost.png)

| 路径 | timed次数 | wall中位数 ms | wall p90 ms | CUDA event中位数 ms | peak allocated MiB | 增量peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| dense | 10 | 382.360001 | 402.015083 | 382.293991 | 983.348633 | 948.890137 |
| factor | 10 | 353.696042 | 367.774891 | 353.629745 | 943.926758 | 909.468262 |

两路交错次序、各5次先测；全部24条预热/计时记录见step_times.csv，10对比较见paired_times.csv。当前单GPU、单固定批次、短窗口测量可能受到共享负载与频率变化影响，不能只挑更快的7对，也不从中位数线性外推完整训练墙钟。显存是batch预驻留的工作集，没有原63000缓存和全量F0参考，不与真实训练峰值混称。

## 数值和权重状态

固定CovarianceFeature final30/7410、原5359代码生成的1230341参数，与同一原3ff8地面Tri-SIG-TIME SSL/defaulthead；实际载入本次新计算实现，严格父config/state。两路loss均0.002348790643737，所有G裁剪前参数梯度最大绝对差1.86264515e-09、相对L2差1.3145415e-07，在预登记范围内。G和F0全部state最终逐值不变、F0无参数梯度，optimizer对象/更新与新训练轮次均0。

factor路径按反射输入列先合并碰撞FIR项，保留main/image正确共轭与2倍实矩阵范数；原源矩阵相对MSE、噪声logMSE和实际后验F0特征损失权重不变。首微批仍执行完整FP64后验与冻结F0，其他微批实际省去完整矩阵构造。数学损失与当前梯度一致，不保证新训练的浮点优化轨迹逐位相同。当前已完成训练/Main保持原5359实现，默认推理仍dense=True；没有据本次短计时热切换或新训候选。

## 输入、访问和成本范围

只用父已保存E30首logical256/draw10，源L_s 27、U_s 229；不按TX或质量选样。每路1次完整梯度一致性检查、2预热、10计时，共26次逻辑反向、6656次G访问/832条后验F0输出；另clean F0参考256和真实Ls smoke2次G/后验/F0/defaulthead。重复访问不是新独立记录或训练曝光。clean参考缓存0.045968秒，整个worker11.532557秒。

源G历史scratch30、F0 E50、原63000IQ先验与源特征方差知识仍计成本，本诊断不是F-free。0TX/V/OE/novel/目标IQ、预测、评分、新统计拟合或身份头；不把Fclean、IQ误差或本次耗时当作识别性能或恢复上界。

## 执行与独立核验

一次真实source2最终G/F0、三项聚焦检查和唯一独立P0/P1通过；旧等价损失组件的12项检查与独立审查沿用。父最后GPU阶段cross_both在GPU4时于GPU6准入，free 24110MiB、原compute 0、新增1；分卡/至多两条本任务NN，父和其他owner保持。单archive SHA一致、远端compile，实际运行中PID/CWD/commit/CUDA/GPU与source smoke/log增长快照已保存。最终worker/controller均自然退出，完整结果exit0。

辅助CPU观察器的启动后存活回执检查返回失败，随后独立读回其已保存的运行中快照，并核实真实GPU worker、source smoke、日志增长和终态；没有重启或重复跑计时。外层helper未输出具体stderr，短观察进程结束竞态只是可能原因，不写成已证实事实。NN任务状态依据独立产物为VERIFIED，观察器启动回执单独记FAILED。

独立CPU复算24记录、20timed/10配对、访问计数、时序、wall/CUDAevent分位数与显存差，最大统计误差0；复算新增神经前向/IQ/真值读取0。当前性能主线继续同原Main双总体七景，source-only计时不改变该矩阵或主验收。
