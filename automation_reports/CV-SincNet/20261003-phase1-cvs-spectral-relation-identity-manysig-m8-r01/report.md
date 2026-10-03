# CVS频点内可学习时间关系：整包与逐频能量归一化×四seed

- run_id：`20261003-phase1-cvs-spectral-relation-identity-manysig-m8-r01`
- group_id：`cvs-clean-spectral-relation-identity-ce`；类别：`cvs`；阶段：`Phase1-CVS-source-research`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留完整Shallow主干，在频率融合前加入共享复时间混合及Hermitian关系编码；同参数比较逐频与整包能量分母。

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

## 结构与本地证据

[设计及先例边界](../../../docs/CVS_SPECTRAL_TEMPORAL_RELATION_20261003.md)。当前两个候选各四seed，保留完整scratch Shallow绝对路径；逐频与整包分母为唯一结构对照差异，各新增26744参数、总247731参数。原单CE、源数据角色、训练预算和精度均固定。实现依据为已完成的源域动态滤波归因，不使用目标测试评分。

8个公开模型完成24次原CE更新；全部新增tensor发生变化、全部参数具有有限CE梯度。理想频点增益性质和有限窗FIR敏感性分别报告。公开检查不是实际识别成绩；不依其数值筛选候选。[公开完整证据](evidence/public_cpu_smoke.json) · [汇总](evidence/public_validation_summary.json)。

本次模型从零训练，无历史checkpoint祖先；Shallow与Anchor仅作为固定源元数据控制。真实scratch checkpoint读回检查在每个source worker读取源IQ前执行。正式训练已发布；实际源代码commit和进程证据见下方启动读回。

## 公开反例驱动的训练前修正

首轮仅绝对floor的原始证据和模型源码均保留；加入固定相对floor=1/64，不扫描参数，也不使用真实源或目标样本。模型36项和源协议/全V产物55项，共91项检查通过。新版8个模型完成24步公开CE；性质测试按实际strong-bin资格报告，触及floor的频点不声称任意幅度不变。首次新版测试错误地要求全部随机频点均不触及floor，已修正测试资格域；未为通过该检查再改模型或阈值。

[当前公开验证](evidence/relative_floor_validation.json) · [完整记录](evidence/public_cpu_smoke_relative_floor.json) · [旧结构源码](evidence/pre_relative_floor_model.py)。有限FIR敏感性只是公开机制检查，不是训练增强或选模分数。

## 发布准备

状态RUNNING；发布前91项聚焦测试、修正后8模型24步公开CE验证及独立P0/P1审查通过。尚无正式训练结果。[本地验证](evidence/local_validation.json) · [独立审查](evidence/source_p0_p1_review.json)。原公开反例保留，不把理论性质等同泛化收益。

## 启动读回

实际发布commit为`d0847cd5ac9251541dbd93d31d0d7a3d933baa47`，dispatcher PID2372927。全部8个worker的PID/CWD/argv、父进程、CUDA可见设备及nvidia-smi物理GPU映射一致，两次日志读回增长。实际配置匹配两个候选各247731参数、26744谱关系参数、原单CE、FP32及固定源数据角色。当前不按早期分数选模，尚未接触目标。

[启动核验](evidence/launch_validation.json) · [完整实际配置与进程](evidence/launch_readback.json) · [SCP前路径与资源](evidence/pretransfer_readback.json)。

## 训练中补齐的离线收尾工具

源collector覆盖8模型完整80000步及1600轮日志，独立复算每模型27000个V样本的关系标量与90个分组，并重算16条源记录的固定选择。大型JSON与NPZ只保留在local_artifacts，Git收录紧凑审计及引用。分析按同seed计算差异，保留完整曲线和固定后段窗口，不改变E200选择。

条件clean实现完成：仅新source winner的4个模型加入44个既有冻结预测，48份预测固定后独立连接truth；未选中新模型不测试。新contract67项、分析93项及旧兼容13项测试通过；另一次独立P0/P1审查通过34项聚焦检查，不重复计入173项。尚未创建正式clean配置或访问目标。源release与模型仍固定为d0847cd5ac9251541dbd93d31d0d7a3d933baa47。

[条件clean独立审查](evidence/conditional_clean_independent_review.json) · [最近训练读回](evidence/continuation_readback.json)。完整工具验证清单见[evidence/offline_tooling_validation.json](evidence/offline_tooling_validation.json)。

## 源产物终态核验

8份固定E200训练、80000步日志审计及完整曲线分析完成，全部训练进程和dispatcher已终止。源码版本`d0847cd5ac9251541dbd93d31d0d7a3d933baa47`，完整16源记录选择与实际远端冻结结果一致；源赢家为`relation_frequency_energy`。当前测试收尾尚未完成，不把源验证指标当作泛化结果。

[完整源分析](source_analysis.md) · [源终态](evidence/source_terminal_readback.json) · [冻结核验](evidence/source_freeze_validation.json) · [架构与科学边界](../../../docs/CVS_SPECTRAL_TEMPORAL_RELATION_20261003.md)。
