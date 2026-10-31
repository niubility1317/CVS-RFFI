# 先验内部相位坐标统一：从零单阶段30轮

- run_id：`20261031-phase1-DURA-phasePrior-E30-s2026092701-r01`
- group_id：`phase1-dura-phase-canonical-prior-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原266951参数/两prior四CG，仅在prior前按coarse最强点转入统一相位坐标、prior后转回；不改IQ幅度/顺序/标量，不增参数或Core调用。scratchjointE30/完整七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果见下方、analysis.md与results.json；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定结构与预算

从零单阶段30轮/7410步，原266951参数G及两prior/四CG/27FFT保持。每个prior内部以coarse base最强点为相位锚，全部4组IQ同时转入该坐标，Core计算后输出转回；标量logvar保持。只有base近零才回退到四组IQ最强点，全IQ零输出0。该约束只作用于prior，不声称整个G等变或F相位不变；不旋转训练目标、不四路平均，不增加参数/Core调用。新增argmax/gather/旋转成本不包含在ConvLinear MAC内，正式实测时延与训练成本。

原纯G8cf对应的初始参数逐项一致，权重共享和CPU/CUDA探针随机流保持；只改prior输入输出坐标。无F/教师/TX/V/target训练，物理四项finalIQ1/prior0.1/STFT0.05/action0.05、AdamW与源01..20后01..10保持，固定最终30轮。研发使用过源F诊断，不能将整条研发路线称完全F-free。3个聚焦检查、实际factory/预算/入口检查与唯一P0P1通过。首步真实2Ls三模块梯度、严格重载、native/optimized一致、prior等变及原F默认头smoke后立即继续。

自动profile、源1800×7、正式168000×7；每role全七景预测固定后独立CPU truth-last。即使源分数较差也完成新G正式矩阵。与同预算纯G配对，默认DURA保持；逐六景≥137216正确、clean≥142256及实际同质量效率仍为验收，R17每景1pp和旧低城区70%单列。无额外设计审批gate，方法与矩阵只在本最小登记维护。


## 实际启动证据

VERIFIED / RUNNING。执行12c7421528c51ee6a0fac4b5bffdbe5057432ff0已提交推送并独立核对远端OID；一次N607 preflight/单归档传输SHA/remotecompile通过。实际controller1643978/worker1644071/GPU4的CWD、完整命令、CUDA、nvidia PID与日志增长已核实；旧own句柄均消失，其他owner保持。启动前同卡1个计算进程、free22883MiB，新增本任务1个。3项聚焦检查、实际factory/预算/入口及唯一P0P1通过。真实2Ls三G模块梯度/strictreload/native-optimized一致/两个prior共同相位对应关系/原F默认头通过；smokeG8、prior-only8、F2。至少完成1轮/247步，完整前缀loss/曝光/源视图/无F-TX-teacher-init计数核对通过，首轮34.008975秒。新G仍从零单阶段30/7410，266951参数与初始权重匹配pureG；prior内部坐标操作不增加参数或Core调用，但实际训练/推理成本待测，不预称加速或泛化收益。不读取F/教师/TX/V/target训练，研发曾用源F诊断明确披露。固定最终30后自动profile/source1800×7/target168000×7，逐role全预测固定后独立truth-last；源表现较差也不取消。默认DURA保持、完整目标ACTIVE。


## 完整训练与正式结果

VERIFIED / ANALYZED。G从零单阶段30轮/7410步，L189000/U1701000；完整30条loss/曝光/源视图/clip/F-TX-teacher-init计数核对通过。仅包裹两个prior的相位坐标，不增参数/Core调用、保留共享别名与初值；训练F/教师/TX/V/target0。研发曾使用源F诊断披露，先验相位性质不冒称整体G或F不变性。七景×168000全部预测固定后truth-last，126正式、105源V、126pureG配对分组与14个源/正式IQ误差独立复算。clean 84.792857%，low urban 65.272024%；低城区较纯G -2.100595、默认DURA -3.617857个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。训练及保存1013.246842秒、控制器至自动全量分析1195.450898秒；266951参数/25.068072m ConvLinear MAC/两prior四CG27FFT，坐标操作另有非卷积成本，实测时延完整保留。默认DURA保持，同质量实际效率仍未证明，完整目标ACTIVE。

详见[完整分析](analysis.md)、[曲线与七景](training_and_formal.png)、results.json与transfer_diagnostics.json。controller/worker已自然结束；没有根据目标结果修改锚点、选epoch或选择性重跑。
