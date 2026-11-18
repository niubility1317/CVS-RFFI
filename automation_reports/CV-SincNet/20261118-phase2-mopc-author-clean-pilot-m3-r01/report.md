# MoPC 作者仓库版 clean 对照 pilot

作者仓库 `ae6554316ad1a2175920e330133a2f103408bf78` 的行为移植到本项目 IQ 与骨干，使用乱序和 drop-last，无梯度裁剪。clean 支持集与 clean 查询集为同一物理样本的地面配对视图；只在固定预测后逐任务独立 truth-last 评分。六个方法在 P01/MIX01/1-1、seed 2026092701 下各运行 K5/N5、K5/N10、K10/N5、K10/N10 四格，先等待星地完整队列停止派发且排空。旧 MoPC 注册结果的采样口径不同，不复用或混入本版。

2026-11-18 06:18 UTC 独立读回：调度器 PID 791268，真实 CWD 为预登记 release；六种真实源 checkpoint 的无 query smoke 均通过。状态 `WAITING_PREDECESSOR`，pending 6、active 0、completed 0、scored 0；此时 clean 预测尚未启动。

源模型使用已核实的冻结源 checkpoint，MRIOR 和 Two-stage 的 B 阶段使用获用户授权的 L_s 源缓存，Tweak 的源注册特征包含于 checkpoint。B 到 C 的旧类原型来自目标旧类 support，属于项目移植；报告须披露这些权限与差异。资源上每 GPU MoPC 最多 4 个进程，保守按全部可见进程计数，启动前空余至少 6000 MiB。真实非有限或作者采样遗漏类别的任务保留并隔离；系统性技术故障停止派发并排空。

- run_id：`20261118-phase2-mopc-author-clean-pilot-m3-r01`
- group_id：`phase2-mopc-author-clean`；类别：`comparison`；阶段：`Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

三池、MIX01–MIX08、七接收机、三模型种子；K5/10 × N5/10；clean support 和 clean query。

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
