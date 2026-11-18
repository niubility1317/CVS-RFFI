# MoPC 作者仓库版 clean 对照 MIX01–MIX03

- run_id：`20261118-phase2-mopc-author-clean-full-mix3-m3-r02`
- group_id：`phase2-mopc-author-clean`；类别：`comparison`；阶段：`Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

三池、MIX01–MIX03、七接收机、三模型种子；K5/10 × N5/10；clean support 和 clean query。

用户于 2026-11-18 将 clean 与星地对照均限定为 MIX01–MIX03。该队列共 1128 项，继承现有 clean pilot 的 6 项，合计 1134 项、4536 个 K/N 格；pilot 尚在等待星地队列并未生成预测。原 clean full r01 停止时无 worker、无预测及无评分，不重新派发其 MIX04–MIX08。原星地队列已经限定 MIX01–MIX03，继续运行。

2026-11-18 06:30 UTC 读回：新调度器 PID 809208，实际 CWD 为预登记 release；六种方法、三个模型种子共 18 项真实源 checkpoint 无 query smoke 全部通过。状态 `WAITING_PREDECESSOR`，pending 1128、active 0、completed 0、scored 0，等待 clean pilot 完成独立评分；当前尚未产生 clean 预测。

结束后核实 `COMPLETED_WITH_QUARANTINE`：1128 项完整队列中 1062 项独立评分、66 项因作者 drop-last 原型提取遗漏类别隔离；clean pilot 另 6 项均已评分。合计 1068/1134 个任务、4272/4536 个 K/N 格评分，缺项保留且不按零分计算。评分汇总与 PAMF 同 split 配对结果见本机 `E:/type10-7/local_artifacts/phase2_author_comparison_summary_20261118.json`。

MoPC 固定作者仓库 `ae6554316ad1a2175920e330133a2f103408bf78` 的项目 IQ/骨干与固定 RNG 移植版本，使用 shuffle/drop-last、无梯度裁剪。复用合法的源 checkpoint；MRIOR、Two-stage 的 B 阶段使用用户授权的 L_s 源缓存，Tweak 源注册特征包含于 checkpoint。每项预测固定后自动独立 truth-last 评分；每卡 MoPC 进程最多 4 个，派发前空余至少 6000 MiB。数值非有限或采样缺类的任务隔离并保留产物。

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
