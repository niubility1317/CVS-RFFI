# CVS镜像频对关系：能量与子空间投影×四seed

- run_id：`20261003-phase1-cvs-mirror-subspace-identity-manysig-m8-r01`
- group_id：`cvs-clean-mirror-subspace-identity-ce`；类别：`cvs`；阶段：`Phase1-CVS-source-research`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留完整Shallow主干，将接收机IQ混合及理想逐频信道写为镜像频对左作用；同参数比较能量关系与有界子空间投影。

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

## 实现与本地证据

两种镜像频对结构各247731参数，其中新增26744。保留完整Shallow主干；输入用实窗STFT构成镜像频对，在同一可学习复时间投影后分别计算能量关系和有界子空间投影。初始整网函数等于各自scratch主干。原单TX交叉熵、数据、优化器和E200预算保持不变。

108项本地检查通过；8模型共24次公开信号CE更新通过，所有新增参数得到更新。独立审查发现秩亏区伴随矩阵相消导致微负trace，已改为等价外积Gram/Cauchy–Binet实现，未截断trace，保留修复前证据。该P1的独立定点复核通过，13项聚焦测试及8项原始复现全部通过。

登记8个scratch训练和12条既有源元数据，共20条固定源记录。固定源评分选中后，若为新候选，仅其4个模型参加clean测试，与48份不可变控制合成52行、416条ALL/RX评分；旧候选获胜则核验既有冻结测试。公开性质诊断、源机制分析和测试结果均不改变选择规则。

[设计](../../../docs/CVS_MIRROR_SUBSPACE_20261003.md) · [本地验证](evidence/local_validation.json) · [公开算子与CE证据](evidence/public_cpu_smoke.json)。正式训练尚未发布，不能宣称识别性能提升或论文目标完成。
