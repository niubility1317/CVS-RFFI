# DURA从零单阶段E20：固定二倍插值执行优化

- run_id：`20261030-phase1-DURA-direct-fastinterp-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

SamefastLRscratchE20/4940/data/seeds/objective;onlyfourCore12of15linear2x replaced;sourcegradientclose andmatchedstep14.44% faster;all7formal

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

## 启动后实际状态

VERIFIED/RUNNING。执行commit 34b5a5cbbf495fad2871beb555e33cd80fa6fe1d，N607一次只读preflight通过，唯一release归档本地/远端SHA匹配、远端compile通过；CWD为`releases/20261030-DURA-direct-fastinterp-r01`。首步两Ls真实输入、实际冻结F50 checkpoint无query smoke PASS，随后直接继续。CPUcontroller PID3707960，GPU4 worker PID3708661；独立/proc核实CWD/完整命令/CUDA可见设备，并在nvidia-smi中匹配实际worker。训练日志从0增长到468bytes、至少完成E1。其他owner后续启动的健康进程保持。

当前尚无正式结果；E20固定冻结后自动source/target各七景、独立truth-last和成本报告，不因源成绩改变测试矩阵。目标继续ACTIVE。

## 完整训练与七景结果

VERIFIED/ANALYZED。完整20条epoch/4940更新、L126000/U1134000曝光、0祖先/教师/TX/F/V/target训练。训练819.150秒（13.652分钟），266951参数，训练峰值allocated 1031.92MiB。完整stdout无traceback/OOM/nonfinite，source1800×7和target168000×7完整，controller/worker自然结束。执行commit34b5a5cbbf495fad2871beb555e33cd80fa6fe1d。

全七prediction固定后独立truth-last；再次从全部不可变logits/truth独立复算126个总体/TX/RX/day分组一致。原F50/defaulthead、同168000物理记录、单seed，不挪用为当前Tri-SIG-TIME SSL/Phase2/3成绩。

|场景|准确率%|相对原生fastLR pp|相对默认DURA pp|距rawclean pp|
|---|---:|---:|---:|---:|
|clean|84.632738|-0.032738|-0.045238|0.043452|
|practical_high|81.076786|+1.082143|-1.748214|3.599405|
|practical_mid|79.466667|+0.556548|-2.094048|5.209524|
|practical_low_suburban|77.889881|+0.432143|-2.168452|6.786310|
|practical_high_urban|79.725000|+0.972024|-2.020833|4.951190|
|practical_mid_urban|72.682738|+0.376786|-2.803571|11.993452|
|practical_low_urban|65.819048|+0.177381|-3.070833|18.857143|

六星地均略高于原生fastLR对照，但全部低于默认DURA，逐景3pp目标=False。lowurban65.819048%，距rawclean18.857143pp；clean较raw少73条正确（−0.043452pp），相对原生fastLR少55条。原R17逐景最多损失1pp容忍=False。默认不变。

此前相同源B256步骤交替测量的14.437%收益仅为该源checkpoint和窗口的步骤证据。本完整训练819.008秒，原生控制771.222秒；不同GPU/时间/共享负载，不能给出严格因果速度比，更未证明整段训练加速。完整质量虽在本单seed六景略高于原生控制，也不解释成slice的普遍识别收益。保留为已验证执行分支，不自动替换默认或声称效率总目标完成。

最终源IQNMSE0.02719969，原生控制0.02702048；算子probe误差0.07944527，原生0.08031073。尽管冻结面板前向/全部裁剪梯度在小容差内，浮点插值/反向归约差异与CUDA非确定性不能保证4940步AdamW轨迹逐位相同；这些数值差异不能单独归因于某个因素。未据正式分数重跑或挑epoch。

参数/权重字节/ConvLinear MAC、两prior/四CG/27FFT预算保持；12/15 Core插值采用slice，action另3次native，冻结推理和实测profile均用同一路径。完整G/G+F B1/B128同卡三轮时延/P95/吞吐、共享驻留显存见results.json。所有权重/原始日志留在local_artifacts/remote原root，仅提交公开汇总。下一action020的源物理门槛、系数与单变量方案已在读取本正式分数前固定，保持原生插值，不据本结果换父或改预算；目标ACTIVE。
