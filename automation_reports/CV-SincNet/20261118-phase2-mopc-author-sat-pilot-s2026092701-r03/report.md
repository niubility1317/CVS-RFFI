# MoPC 作者仓库行为版星地小样本验证

- run_id：`20261118-phase2-mopc-author-sat-pilot-s2026092701-r03`
- group_id：`phase2-mopc-author-satellite`；类别：`comparison`；阶段：`Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

六条原生及 B→MoPC 路径，同一星地支持集 K5/N5；作者 drop-last 行为，无梯度裁剪，预测固定后逐任务独立评分。

用户允许与其他任务共享显卡，并指定每卡 MoPC 最多 4 个进程。共享模式保守地把每卡所有已观察 GPU/预分配进程计入 4 的上限，派发时空闲显存至少 6000 MiB、每个尚未充分分配的进程预留至 3000 MiB；其他 owner 不停止、不修改。资源计数是时点快照，其他 owner 并发变化时需持续核实，不以它保证跨 owner 原子互斥。

参考作者仓库 `ae6554316ad1a2175920e330133a2f103408bf78`；IQ、骨干和随机数流是项目适配。固定 A 的源类原型与注册训练、注册后两次原型提取均采用打乱、batch16、丢尾批；缺类及真实非有限隔离，有限梯度统计不应触发失败。混合 B→C 的旧原型使用全部目标旧类 support 均值，是项目移植而非作者原生 B；混合 C 损失逐步曲线尚不持久化。

按用户授权，MRIOR 与 Two-stage 的 B 阶段使用 L_s 源域回放；Tweak checkpoint 含源注册特征；ProtoNet 与 POSTER 的 B 使用目标 support；原生 MoPC 使用冻结源原型。真实源模型检查先于目标 worker，query 不参与拟合，预测固定后逐项独立 truth-last 评分。各方法实际源缓存用量和目标标签数应在最终报告说明。

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
