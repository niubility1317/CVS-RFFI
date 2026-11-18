# MoPC 作者仓库行为版星地小样本验证

- run_id：`20261118-phase2-mopc-author-sat-pilot-s2026092701-r02`
- group_id：`phase2-mopc-author-satellite`；类别：`comparison`；阶段：`Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

六条原生及 B→MoPC 路径，同一星地支持集 K5/N5；作者 drop-last 行为，无梯度裁剪，预测固定后逐任务独立评分。

参考作者仓库提交 `ae6554316ad1a2175920e330133a2f103408bf78`，本实现适配 IQ、少样本类别阶段和其他方法骨干，项目随机数流不保证与作者脚本逐步一致，也不使用作者脚本对测试集逐 epoch 访问。C 训练与两次独立的原型提取按 batch16 打乱并丢弃不足整批的尾部；原生 MoPC 源原型也从冻结 A 和合法 L_s 以相同采样规则重新计算。若某次提取丢失某类，该任务保留并隔离，不补样本。源原型使用模型 seed 固定采样；A 网络参数不变，原 checkpoint 保留。

五种 B→MoPC 为项目跨骨干移植：B 后旧类原型取全部当前目标旧类 support 的类别均值，作者原方法没有该 B 阶段；不能声称它们与作者原生完整流程等同。混合注册的逐步 C 损失当前不持久化，无法据此声称已核查其完整下降曲线。各已完成预测立即独立 truth-last 评分；低准确率作为实验结果记录，不触发技术停止。

用户允许对比方法使用源域训练缓存：MRIOR 和 Two-stage 的 B 阶段使用 L_s 回放；Tweak checkpoint 带有源注册特征；ProtoNet 和 POSTER 在 B 使用当前目标 support；原生 MoPC 使用重新计算的源旧类原型。六种方法派发前做真实源 checkpoint 无 query smoke。`U_s` 身份标签隐藏，目标 query 不参与拟合或 checkpoint 选择；逐行保留实际用量。

单 seed、小数据单元的可证伪条件是六条路径产生合法 K5/N5 全注册类固定预测、独立评分及准确的失败说明；真实非有限值和采样缺类保留并隔离。评分不用于调整已固定的方法、模型、种子、任务顺序或后续训练参数。完整三 seed 星地矩阵需在小规模结果核实后另行无重复接续，不能把此 pilot 当作全量已发布。

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
