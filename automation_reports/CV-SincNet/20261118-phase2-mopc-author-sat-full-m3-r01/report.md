# MoPC 作者仓库行为版星地三种子完整对比

- run_id：`20261118-phase2-mopc-author-sat-full-m3-r01`
- group_id：`phase2-mopc-author-satellite`；类别：`comparison`；阶段：`Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

六方法三池七接收机 MIX01–03 三种子，K5/10 × N5/10；先等待六项 pilot 独立评分，精确复用 pilot K5/N5 与混合 B 状态。

完整矩阵为 6 × 3 池 × 3 场景 × 7 接收机 × 3 seed = 1,134 个方法任务。每任务固定四个 K/N，但其中六个 pilot 的 K5/N5 由已完成且已评分的 pilot 保留，因此本 run 只生成 4,530 个新预测格，最终汇总需显式连接 pilot 六格。五个混合方法的 pilot K5/N10 沿用其 pilot 的同一个 K5 B 模型及 anchor，K10 独立适应；其他任务的每个 K 分别适应。已复用 B 的实际成本回查 pilot，不能把新行记录中的空 B 耗时当作零成本。

指定作者仓库提交 `ae6554316ad1a2175920e330133a2f103408bf78`。MoPC 不裁剪梯度，保留作者目标函数、点积原型纠正、batch16 打乱丢尾和两次独立原型提取；IQ、骨干和项目随机数流为适配。原生三 seed 使用分别从固定 A 与合法 L_s 派生的源原型；混合方法的旧类原型为全部当前目标旧类 support 的类别均值，是项目 B→C 移植而非作者原生 B。混合 C 的逐步损失目前未持久化，不能宣称完整损失曲线已检查。

用户允许对比方法使用源域训练缓存。MRIOR 和 Two-stage 在 B 使用 L_s 回放；Tweak checkpoint 含源注册特征；ProtoNet 与 POSTER 的 B 使用目标 support；原生 MoPC 使用冻结源原型。记录各方法实际源缓存和目标标签用量；U_s 身份标签隐藏，query 不参与拟合。每任务预测固定后自动独立 truth-last 评分，低性能不触发技术停止。

调度器只在 pilot 六项均已固定预测、全部独立评分且无隔离时自动派发；pilot 失败或隔离则保留现场并停止后继派发。正常数值保护不会因有限梯度的 FP32 范数统计溢出停机；真实非有限或作者采样缺类的任务隔离并保留部分输出。其他系统性技术失败停止派发并排空。源、支持、query 和任务身份均沿用已验证数据，本研究不是此前未暴露的独立盲测。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

2026-11-18：三个 seed 的源原型均由各自冻结 A 与 L_s 按作者规则产生，各使用 6,300 条中的 6,288 条；独立检查三份派生模型参数均与各自原模型一致。完整接续候选的独立 P0/P1 复核通过，代码 commit `066851074f763f84da8b7d864e01702ac2e2d61d`，归档本地与 N607 一次 SHA-256 一致，远端编译通过。

完整控制器 PID 688418 已独立核实 CWD/cmdline；六方法 × 三 seed 共 18 个真实源 checkpoint 无 query smoke PASS。首次状态 `WAITING_PREDECESSOR`：1,134 项待派发，0 worker、0 完成、0 评分、0 失败、0 隔离。它等待 pilot 六项全部评分且无隔离后自动运行；pilot 当时也因其他 owner 占用显卡而无 worker。不得把排队或源 smoke 称作目标实验已完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
