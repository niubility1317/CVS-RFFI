# 物理恢复与原时序签名保持：从零单阶段30轮

- run_id：`20261031-phase1-DURA-signatureGuide-E30-s2026092701-r01`
- group_id：`phase1-dura-actual-signature-guidance-E30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原266951参数G/物理四项/scratchE30，新增原F50实际加权TripleSIG源约束，精确前缀一次输入VJP，逐条上限0.5倍IQ梯度。新cleanSIG缓存40.32MB，完整七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整结果见analysis.md/results.json；完整分组与成本保留。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 独立实际时序签名指导恢复器

G保持原266951参数，从零单阶段30轮/7410步，完整coarse/action/prior联合优化。原source63000及20份缓存后重复01至10、随机流、学习率、四项物理损失与纯G8cf对照匹配；不复用G权重，不再串联模块预训练。训练损失新增实际原F50加权TripleSIG相对MSE，alpha1固定，精确t2前缀一次输入VJP后注入detach梯度；逐条上限为IQ梯度范数0.5倍，近clean时同步减弱。该上限只约束IQ输出空间，不是G参数/整体风险或clean准确率保证。

原F50/defaulthead保持冻结，不使用classloss/硬伪TX/直接TX/V/target训练。新source clean签名缓存63000×160 FP32/40.32MB，实际63000前缀前向；训练1890000前缀前向和输入VJP，不跑后续完整F路径。F加载、缓存和额外F路径隔离CPU/CUDA随机流，F50/11100上游监督与新增成本全部披露，属于F-guided。

此前PA/固定步幅/损失缩步源诊断FAIL全部保留：最后一项仍有RX8净−1，未宣称资格PASS。本次是用户既有源F指导和30轮研发授权内另登记的可学习G假设，不是把pairedclean源优化器直接用于部署或以总体净增覆盖保护条件。它保留物理约束，只有冻结后的真实TX测试能判断收益。

5项新聚焦检查、已有精确前缀检查和实际入口通过，唯一P0/P1通过。首步真实2Ls核对完整F-vs-prefix的实际SIG和IQ输入VJP、真实G梯度cap、三个G模块反向、epoch0严格重载及原头state；PASS立即继续。smoke完整F4/前缀4/VJP6/G6、直接TX/V/target0。满30轮固定final后自动同卡profile/source1800×7/target168000×7，每role全七景预测固定后才独立truth-last，源成绩差也不取消正式测试。

默认DURA保持，六景各≥137216正确/81.6761904762%、clean≥142256，原R17逐景1pp和低城区70%另列。完整参数/存储/延迟/显存/训练缓存和全流程时间如实报告，不凭前缀更短宣布同质量提速。其他owner不受影响。


## 实际启动核实

VERIFIED RUNNING。执行47591cdfe2e6e6dd7534cdea846badd942321b8f，controller1690298/worker1690393、GPU4；实际PID/CWD/完整argv/空CUDA控制器与workerCUDA4、GPU计算进程和日志增长均已核实。启动前该卡剩余22883MiB、已有1个计算PID，新增1条符合总数≤3；历史本goal训练/源诊断PID均不存在，其他owner保持。首步真实2Ls全部检查PASS并立即继续。新source SIG缓存40.32MB/0.847077秒、63000前缀输入已完成；当前完成源训练轮数2，正式性能结果尚无，不宣称优化成功。5项新检查及1项checkpoint权限负测试通过，唯一P0P1通过。launch_evidence.json只作聚合状态证据，实际源数据/权重/日志私有保存。


## 完整训练与正式结果

VERIFIED / ANALYZED。G从零单阶段30轮/7410步，L189000/U1701000；完整30条物理+SIGloss/曝光/源视图/clip/F-TX-init计数与实际逐batchcap0.5核对通过。新增实际加权TripleSIG源缓存40.32MB/0.847077秒/63000前缀输入；训练1890000前缀+1890000输入VJP、完整F后续路径0，F全state不变、加载和路径RNG隔离。F50源监督知识继承、属于F-guided；原源oracle资格FAIL保持，不称其晋级或物理上限。七景×168000全部预测固定后truth-last，126正式、105源V、126pureG配对分组与14个源/正式IQ误差独立复算；512缓存签名与完整F误差8.047e-7。clean 84.769048%，low urban 68.835119%；低城区较纯G +1.462500、默认DURA -0.054762个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=True，旧低城区70%=False。G训练及保存1112.173977秒，含worker准备1113.944462秒，控制器至自动全量分析1298.002890秒；266951参数/25.068072m ConvLinear MAC/两prior四CG27FFT。默认DURA保持，同质量实际效率须结合正式质量判断，完整目标ACTIVE。

详见[完整分析](analysis.md)、[曲线与七景](training_and_formal.png)、results.json与transfer_diagnostics.json。controller/worker已自然结束；无目标反馈、换轮次、调cap或选择性重跑。
