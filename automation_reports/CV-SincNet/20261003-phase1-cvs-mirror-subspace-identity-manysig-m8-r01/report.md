# CVS镜像频对关系：能量与子空间投影×四seed

**本轮实验与测试收尾完成；新结构未取代固定源控制，整体性能与论文目标仍未达成。** 8份从零训练的固定E200模型、80000步日志与720个全V关系分支单元已核验。mirror_energy源分数97.0981%，相对Shallow为-0.0236±0.0801个百分点，2/4个seed正提升。mirror_subspace源分数97.1171%，相对Shallow为-0.0046±0.1423个百分点，3/4个seed正提升。固定源规则保留relation_frequency_energy；四seed配对SD描述固定划分下模型随机性，不是置信区间。

- run_id：`20261003-phase1-cvs-mirror-subspace-identity-manysig-m8-r01`
- group_id：`cvs-clean-mirror-subspace-identity-ce`；类别：`cvs`；阶段：`Phase1-CVS-source-research`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留完整Shallow主干，将接收机IQ混合及理想逐频信道写为镜像频对左作用；同参数比较能量关系与有界子空间投影。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整源结果与所选历史控制的测试复核已完成；新结构未获源选择，不新增query。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 实现与本地证据

两种镜像频对结构各247731参数，其中新增26744。保留完整Shallow主干；输入用实窗STFT构成镜像频对，在同一可学习复时间投影后分别计算能量关系和有界子空间投影。初始整网函数等于各自scratch主干。原单TX交叉熵、数据、优化器和E200预算保持不变。

108项本地检查通过；8模型共24次公开信号CE更新通过，所有新增参数得到更新。独立审查发现秩亏区伴随矩阵相消导致微负trace，已改为等价外积Gram/Cauchy–Binet实现，未截断trace，保留修复前证据。该P1的独立定点复核通过，13项聚焦测试及8项原始复现全部通过。

登记8个scratch训练和12条既有源元数据，共20条固定源记录。固定源评分选中后，若为新候选，仅其4个模型参加clean测试，与48份不可变控制合成52行、416条ALL/RX评分；旧候选获胜则核验既有冻结测试。公开性质诊断、源机制分析和测试结果均不改变选择规则。

[设计](../../../docs/CVS_MIRROR_SUBSPACE_20261003.md) · [本地验证](evidence/local_validation.json) · [公开算子与CE证据](evidence/public_cpu_smoke.json)。启动时正式源训练已发布并独立核验；不能宣称识别性能提升或论文目标完成。

## 启动读回

实际发布commit为`bb0ffae4263419a992a44de8975ec7eb83c4596f`，dispatcher PID2413989。全部8个worker的PID/CWD/argv、父进程、CUDA可见设备及nvidia-smi物理GPU映射一致，两次日志读回增长。实际配置匹配两个候选各247731参数、26744谱关系参数、原单CE、FP32及固定源数据角色。启动时不按早期分数选模，未接触目标。

[启动核验](evidence/launch_validation.json) · [完整实际配置与进程](evidence/launch_readback.json) · [SCP前路径与资源](evidence/pretransfer_readback.json)。

## 冻结训练期间补齐的收尾工具

完整源collector/analyzer已通过69项收集与15项分析测试，覆盖全部80000步、1600轮和每模型27000个V样本的31频对标量；大文件保留在local_artifacts。分析独立重算20源记录，不重选epoch。

条件clean工具已通过74项契约和93项结果分析检查；67项旧spectral契约兼容检查通过。独立P0/P1审查102项聚焦合成测试通过，不额外计入上述独立用例。只有新源赢家的4个模型会增加query访问，48控制预测保留原路径，全部52预测固定后独立truth-last评分。正式clean配置最终未创建：源规则保留原控制，下文记录已有测试证据的复核。

[工具验证](evidence/offline_tooling_validation.json) · [条件clean独立审查](evidence/conditional_clean_independent_review.json) · [最近训练读回](evidence/continuation_readback.json)。源训练release保持`bb0ffae4263419a992a44de8975ec7eb83c4596f`，没有热修改、停止或重启。

## 源产物终态核验

8份固定E200训练、80000步日志审计及完整曲线分析完成，全部训练进程和dispatcher已终止。源码版本`bb0ffae4263419a992a44de8975ec7eb83c4596f`，完整20源记录选择与实际远端冻结结果一致；源赢家为`relation_frequency_energy`。冻结时测试收尾尚未完成；其后复核结果见下文，不把源验证指标当作泛化结果。

[完整源分析](source_analysis.md) · [源终态](evidence/source_terminal_readback.json) · [冻结核验](evidence/source_freeze_validation.json) · [架构与科学边界](../../../docs/CVS_MIRROR_SUBSPACE_20261003.md)。

## 冻结后的测试收尾

两个新候选未入选，不新增query访问；条件52行clean实验未激活。独立读取既有48行预测状态、配置及checkpoint来源，384条评分与原记录完全一致，所选4份checkpoint与本次源选择逐seed匹配。复用relation_frequency_energy既有clean准确率70.6926%±1.2835%；这是历史控制的复核，不是新结构的测试成绩，也不新增独立测试证据。

[复用核验](evidence/retained_control_test_verification.json) · [已有clean报告](../20261003-phase1-cvs-spectral-relation-clean-manysig-m48-r01/report.md) · [本轮结论](evidence/completion_verdict.json)。保持单CE与原训练策略；后续研究只能依据源证据，不能用这次测试收尾决定新结构。

## 对当前最优源控制的结论

两种新结构相对`relation_frequency_energy`均4/4个seed负差；镜像子空间的平均源分数差为−0.2389个百分点，最差源RX差为−0.3704个百分点。保留原控制，未启动条件52行clean实验。局部数学不变性通过不代表识别优化成功，Phase1论文与整体性能目标尚未完成。详细证据与机制限制见[源分析](source_analysis.md)。
