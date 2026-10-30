# DURA从零单阶段E20：移除方差辅助校准

- run_id：`20261030-phase1-DURA-direct-noVarAux-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Sourcegradientmatchedablation:onlyremove.001logvarianceauxloss;sameordinaryscratchinit/E20/4940/LR1e-3;nativevariancelearnsthroughIQ;all7formal

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

## 预登记匹配对照

仅移除0.001×logvariance辅助校准，保留最终IQ、内部prior、谱和source action四项。依据固定fastLR源checkpoint的32批源梯度；不因恒等版正式成绩切换初始化或选择父权重。实际完全从普通随机初始化训练，累计E20/4940，source数据/seed/探针/学习率/FP32均匹配fastLR。方差仍通过恢复路径学习，只声明正值一致性参数，不声明物理噪声校准。

一个独立P0/P1审查PASS，两个聚焦测试PASS；首步真实checkpoint无query smoke后立即训练。完成后自动source1800×7、target168000×7、同卡时延及独立truth-last，所有负结果保留。无性能早停或候选重排，默认DURA不变。具体命令/不可覆盖root/GPU容量与技术失败停止规则见experiment.json。

## 启动后实际状态

VERIFIED/RUNNING。执行commit cf40edea90667d9645a724b1e8f333027cdf7869，N607一次只读preflight通过，唯一release归档本地/远端SHA匹配、远端compile通过；CWD为`releases/20261030-DURA-direct-noVarAux-r01`。首步两Ls真实输入、实际冻结F50 checkpoint无query smoke PASS，随后直接继续。CPUcontroller PID3641689，GPU7 worker PID3642320；独立/proc核实CWD/完整命令/CUDA可见设备，并在nvidia-smi中匹配实际worker。训练日志从0增长到467bytes、至少完成E1。其他owner后续启动的健康进程保持。

当前尚无正式结果；E20固定冻结后自动source/target各七景、独立truth-last和成本报告，不因源成绩改变测试矩阵。目标继续ACTIVE。

## 完整20轮与七景正式结果

VERIFIED/ANALYZED。完整20条epoch/4940更新、L126000/U1134000曝光；0祖先checkpoint/学习教师/TX/F/V/target训练。训练851.976秒（14.200分钟），266951参数，训练峰值allocated 1034.42MiB。完整stdout无traceback/OOM/nonfinite错误，source1800×7和target168000×7完成；控制器/worker自然结束。执行commit cf40edea90667d9645a724b1e8f333027cdf7869。

全七prediction固定后独立CPU truth-last，再从全部不可变logits/truth独立复算126个总体/TX/RX/day分组一致。原F50 seed2026092701/defaulthead、168000物理记录、单seed，不扩展为当前Tri-SIG-TIME SSL或Phase2/3成绩。

|场景|准确率%|相对fastLR pp|相对默认DURA pp|距raw clean pp|
|---|---:|---:|---:|---:|
|clean|84.742262|+0.076786|+0.064286|-0.066071|
|practical_high|80.904167|+0.909524|-1.920833|3.772024|
|practical_mid|79.406548|+0.496429|-2.154167|5.269643|
|practical_low_suburban|77.869048|+0.411310|-2.189286|6.807143|
|practical_high_urban|79.998810|+1.245833|-1.747024|4.677381|
|practical_mid_urban|74.392857|+2.086905|-1.093452|10.283333|
|practical_low_urban|68.470833|+2.829167|-0.419048|16.205357|

六星地均高于同初始化/数据/训练预算fastLR对照，lowurban+2.829167pp到68.470833%，距默认DURA仍低0.419048pp。其他五景亦低于默认；六星地各距rawclean3.772024至16.205357pp，逐景3pp目标=False。clean较raw净增加111条正确，419纠错/308损伤，相对默认多108条正确；不能用总体clean增益证明每个TX细节均恢复。原R17逐景最多损失1pp容忍=True，亦非本次验收目标。默认保持。

源域IQNMSE较匹配fastLR最终0.02702048降到0.02503956，降低7.331%；保持固定finalE20，未选中间轮次。完整曲线与全部分量有限，logvariance误差仅日志、不用于训练；因此其约20.93不能解释成物理噪声校准。该单seed匹配消融支持移除辅助项在本配方的收益，不证明一般因果规律。实际14.198分钟受共享GPU负载影响，不能与不同时间/卡12.85分钟做训练执行因果比；仍需全质量成本验收。

同卡G/G+F、B1/B128三轮交替median/P95/吞吐和共享驻留显存见results.json。所有原始日志、权重和完整分组保留local_artifacts/remote原root；Git仅公开汇总。下一fastinterp配方已在本正式结果前由源端零更新速度阈值固定，仍保留原fastLR全部五项损失，不据此目标成绩混入noVar或恒等初始化。目标继续ACTIVE。
