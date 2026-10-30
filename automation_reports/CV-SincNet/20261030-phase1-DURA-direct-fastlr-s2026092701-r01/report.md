# DURA从零单阶段E20：源拟合学习率匹配

- run_id：`20261030-phase1-DURA-direct-fastlr-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

MatchedscratchE20sameinit/data/loss/budget;onlypeakLR2e-4to1e-3;sourcefull20epochunderfitdiagnosis;noancestorweights/teacher;all7formal

本候选的直接依据是完整source训练日志：从零E20的最终IQNMSE0.04014097，常规续训0.02096943；峰值LR2e-4沿用的是旧权重微调设置，从零20轮末仍有源拟合缺口。它是优化器日程假设，不认定全部误差必然来自学习率。本run在Agent读取该pilot正式分数之前已登记。

唯一科学变更为峰值LR从2e-4到1e-3；固定warmup247、最终2e-5、E20/4940、batch256、AdamW/WD/clip、随机初始化/modelseed、source shuffle/probes/目标权重及所有数据/七景矩阵。仍无祖先checkpoint、学习教师或TX/F/V/target训练。三个子模式run/worker/score均进入同一fast_lr模块，customschema严格加载同一新配置；新不可覆盖root。测试已验证子进程传播与LR端点/预算，一次独立P0/P1审查PASS。此前两个恢复队列已全部结束，不会重新启动；当前任务GPUworker仍合计最多2。

每版均完整测试；不按source表现取消正式预测，不按目标结果调参或换父。成功仍要求六个星地场景逐个距同一rawclean≤3pp并保持clean性能，同时核算累计训练成本。只完成更快的source拟合不视为目标完成。

## 已发布与实际运行

VERIFIED：新release `20261030-DURA-direct-fastlr-r01`、执行commit `a0acfbe0c237e38d0460a9f1ea29c46b22cf84a6`，单archiveSHA一致/远端compile通过，CPUcontroller PID3580693、worker PID3580791、物理GPU6、CWD/cmdline/log增长已读回。实际2Ls梯度/随机checkpoint严格存取/原F50无query smoke PASS后已自动继续。

首个完整epoch为E1/247、39.260秒，Ls6300/Us56700曝光，IQNMSE0.4660237，F_inputs/TX_labels/learned_teacher_inputs/initialization_checkpoint_loads均0。该记录只证明实际source训练健康进行，不代表正式性能或最终速度收益。后续按固定E20自动七景truth-last；不重新启动已结束的旧恢复队列。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 完整E20与七景正式结果

VERIFIED/ANALYZED。20条完整epoch记录、4940更新、L126000/U1134000曝光，0祖先checkpoint/学习教师/TX/F/V/target训练。训练771.222秒（12.854分钟），参数266951，训练峰值allocated 1034.42MiB。完整stdout无traceback/OOM，source1800×7和target168000×7全部完成，CPUcontroller和GPUworker自然退出。

每组全七prediction完成后独立CPUtruth-last；再次从不可变logits/truth独立复算126个总体/TX/RX/day分组，与正式结果一致。原F50/defaulthead、单seed、原168000物理记录；rawclean84.676190%，不扩展为当前Tri-SIG-TIME SSL或Phase2/3成绩。

|场景|准确率%|相对原学习率pp|相对默认DURApp|距rawclean pp|
|---|---:|---:|---:|---:|
|clean|84.665476|-0.736905|-0.012500|0.010714|
|practical_high|79.994643|+0.674405|-2.830357|4.681548|
|practical_mid|78.910119|+1.695833|-2.650595|5.766071|
|practical_low_suburban|77.457738|+2.492857|-2.600595|7.218452|
|practical_high_urban|78.752976|+1.896429|-2.992857|5.923214|
|practical_mid_urban|72.305952|+6.613690|-3.180357|12.370238|
|practical_low_urban|65.641667|+9.600595|-3.248214|19.034524|

六个星地场景均高于原学习率从零E20，但仍全部低于默认DURA；lowurban提高9.600595pp至65.641667%，距rawclean仍19.034524pp，六景逐个3pp目标=False。clean相对raw少18条正确（−0.010714pp），与默认DURA少21条正确；不能只凭增益对照或近似clean保持宣布完整目标达成。默认保持。

完整source曲线显示同E20 IQNMSE从0.04014097降为0.02702048，降低32.69%。描述性训练误差≤0.05首次时点从E10提前至E5，各epoch模拟观测不同，此数不是预登记选模/早停门槛，也不是同识别精度2倍加速。最终固定E20才做正式测试，所有负结果保留。

匹配同卡G/G+F B1/B128三轮交替时延、P95、吞吐见results.json。共享负载下训练墙钟与此前不同卡不作因果速度比；CPU步骤收益也不替代GPU质量成本。初始恒等的下一独立候选已在本run正式结果前固定，其余数据/预算/损失/LR保持，不根据本结果回调。

权重与全部原始日志保留local_artifacts/pamf_restoration/<run_id>/scratch_joint/及remote原root；未提交模型/IQ/日志。执行commit a0acfbe0c237e38d0460a9f1ea29c46b22cf84a6，目标ACTIVE。

原R17逐景最多下降1pp容忍=False；具体delta_R17_pp见results.json。通过或未通过该旧容忍均不替代六景rawclean3pp目标。
