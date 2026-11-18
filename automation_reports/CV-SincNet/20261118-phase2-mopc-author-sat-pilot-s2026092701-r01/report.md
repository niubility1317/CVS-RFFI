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

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
