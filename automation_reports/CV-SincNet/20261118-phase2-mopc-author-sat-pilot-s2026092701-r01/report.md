# MoPC 作者仓库行为版星地小样本验证

- run_id：`20261118-phase2-mopc-author-sat-pilot-s2026092701-r01`
- group_id：`phase2-mopc-author-satellite`；类别：`comparison`；阶段：`Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

六条原生及 B→MoPC 路径，同一星地支持集 K5/N5；作者 drop-last 行为，无梯度裁剪，预测固定后逐任务独立评分。

作者仓库固定提交 `ae6554316ad1a2175920e330133a2f103408bf78`。这是适配项目 IQ、少样本类别阶段和外部骨干的仓库行为实现，项目随机数流与作者原脚本不保证逐步一致；不会把目标 query 用于作者脚本原有的逐 epoch 测试。每个训练 epoch 的不完整尾批丢弃，两次新类原型提取分别随机打乱并丢弃尾批；任一轮原型中缺类时保留现场并隔离该任务，不补样本或静默更改方法。目标全类预测完成后立即对该行独立 truth-last 评分；低分不触发技术停止。

用户于 2026-11-18 明确允许对比方法使用源域训练缓存。MRIOR 和 Two-stage 的 B 阶段使用 `L_s` 源域回放；Tweak 的 checkpoint 带有源域注册特征；ProtoNet 和 POSTER 的 B 阶段使用目标旧类 support；原生 MoPC 使用源模型已保存的原型和当前新类 support。六种方法在派发前另做真实源 checkpoint 无 query smoke。`U_s` 身份标签仍不开放，query 不参与拟合。每个方法实际源回放数量及目标标签数以运行时记录为准。

本 pilot 预登记的可证伪条件为：六条路径均能在不读取 query 真值及不以 query 更新模型的条件下产生 K5/N5 全注册类固定预测，随后由独立 scorer 完成评分；真实非有限值和作者采样丢失某类均单独记为缺项，不制造替代值。任何准确率高低都只如实报告，不据此修改方法或重新选用目标 checkpoint。完整矩阵是否具备可执行条件依据合法产物和失败原因判断，不依据目标分数挑选。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

截至 2026-11-18 预备状态：N607 的代码归档 SHA-256 与本地一致，远端编译通过；六种方法的 seed 2026092701 真实源 checkpoint 均通过 CPU 仅源检查，未读取 target/query。独立读回的调度配置为六方法、同一 capsule、K5/N5、无梯度裁剪及作者 drop-last 采样。原生与混合方法每行预测完成后由独立进程评分。当前尚未完成针对本候选的独立 P0/P1 复核，也未启动正式预测，不存在评分结果；科学可行性和 CUDA 数值表现仍待实际运行。

旧 clean 作者目标函数结果使用保留尾批/全量原型采样，不能在缺少等价证据时与本次 `author_drop_last_v1` 注册结果合并；早先删除的论文目标函数版和裁剪稳定版不复用。三 seed、三个数据池、七接收机、MIX01–03、四 K/N 的完整星地矩阵预计 1,134 个方法任务，仍需按本 pilot 的实际结果与已固定条件安全接续。clean 侧对应新版的逐项清点与自动接续尚未完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
