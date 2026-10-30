# DURA从零单阶段E20：增强源算子监督

- run_id：`20261030-phase1-DURA-direct-action020-s2026092701-r01`
- group_id：`phase1-dura-single-stage-efficiency`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

MatchedfastLRordinaryscratch/native/E20;onlyactionprobeweight.05to.2;sourceoracleAthresholdPASS;0ancestor/teacher;automatic7formal

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

VERIFIED/RUNNING。执行commit 36cc47e1c140089ae013288e8e4a615c85563e05，N607一次只读preflight通过，唯一release归档本地/远端SHA匹配、远端compile通过；CWD为`releases/20261030-DURA-direct-action020-r01`。首步两Ls真实输入、实际冻结F50 checkpoint无query smoke PASS，随后直接继续。CPUcontroller PID3775702，GPU4 worker PID3775805；独立/proc核实CWD/完整命令/CUDA可见设备，并在nvidia-smi中匹配实际worker。训练日志从0增长到465bytes、至少完成E1。其他owner后续启动的健康进程保持。

当前尚无正式结果；E20固定冻结后自动source/target各七景、独立truth-last和成本报告，不因源成绩改变测试矩阵。目标继续ACTIVE。

## 完整训练与正式结果

VERIFIED/ANALYZED。完整20条epoch/4940更新，L126000/U1134000曝光，0祖先/教师/TX/F/V/target训练。训练1091.922秒（18.199分钟），266951参数，峰值allocated 1034.42MiB；完整stdout无traceback/OOM，source1800×7与target168000×7完成，controller/worker自然结束。执行commit36cc47e1c140089ae013288e8e4a615c85563e05。

全七prediction固定后独立CPU truth-last；再次从不可变logits/truth独立复算126个总体/TX/RX/day分组一致。原F50/defaulthead、同168000物理记录、单seed，不挪用为当前Tri-SIG-TIME SSL/Phase2/3成绩。

|场景|准确率%|相对原fastLR pp|相对默认DURA pp|距rawclean pp|
|---|---:|---:|---:|---:|
|clean|85.237500|+0.572024|+0.559524|-0.561310|
|practical_high|80.426190|+0.431548|-2.398810|4.250000|
|practical_mid|78.931548|+0.021429|-2.629167|5.744643|
|practical_low_suburban|77.422024|-0.035714|-2.636310|7.254167|
|practical_high_urban|78.947024|+0.194048|-2.798810|5.729167|
|practical_mid_urban|71.064881|-1.241071|-4.421429|13.611310|
|practical_low_urban|63.557738|-2.083929|-5.332143|21.118452|

增强action监督未实现星地性能提升：较匹配fastLR六星地3升3降，lowurban下降2.083929pp至63.557738%，相对默认低5.332143pp、距rawclean21.118452pp。clean85.237500%较raw提高0.561310pp/943条净正确，不能替代星地效果。六景逐个3pp=False，R17逐景1pp容忍=False，默认不变。

完整源轨迹显示E20 IQNMSE0.035239高于原fastLR0.027020，probe NMSE0.233236也未降低，末轮98.7854%批次裁剪。源oracle固定prior可改善不保证该损失系数在有限AdamW训练中更好；冻结E20局部梯度范数比例不代表整个训练过程。新的初始化诊断也没有达到预定减半门槛，不据此加入组合或放宽标准。没有选择中间epoch或按目标结果重跑。

实际18.198分钟处于共享GPU负载，不能把跨时/卡耗时差直接归因于action系数。全G/G+F B1/B128同卡交替median/P95/吞吐与共享显存见results.json；完整参数/存储/ConvLinear MAC、两prior/四CG/27FFT均保持。所有权重/原始日志/完整分组留原root与local_artifacts，仅提交公开汇总。

用户另行明确授权独立源F指导分支；其既定noVar基础和beta0.1在读取本正式分数前已于可选方案固定，不据本结果选父或调整超参。该新分支继承F源有标签知识，必须单独说明，原F-free实验与本负结果完整保留；目标继续ACTIVE。
