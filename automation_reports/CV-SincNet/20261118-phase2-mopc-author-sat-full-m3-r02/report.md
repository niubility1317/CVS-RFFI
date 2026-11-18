# MoPC 作者仓库行为版星地三种子完整对比

- run_id：`20261118-phase2-mopc-author-sat-full-m3-r02`
- group_id：`phase2-mopc-author-satellite`；类别：`comparison`；阶段：`Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

六方法三池七接收机 MIX01–03 三种子，K5/10 × N5/10；先等待六项 pilot 独立评分，精确复用 pilot K5/N5 与混合 B 状态。

本 run 共 1,134 个方法任务，新增 4,530 个 K/N 格；另需复用 pilot 的六个 K5/N5 格形成完整 4,536 格。五个混合方法的 pilot K5 B 精确加载以计算同任务 K5/N10；K10 分别适应。每项预测固定后独立评分，复用 B 的耗时需回查 pilot，不能记为零。完整调度器只在 pilot 六项全部评分、无隔离后派发；pilot 失败时停止本后继派发。

用户授权共享所有显卡，MoPC 每卡最多 4 个进程。实现保守计数所有可见 GPU/预分配进程，最多总计 4 个；派发前 >=6000 MiB，低/未知占用按每进程 3000 MiB 预留，进程间资源状态为快照，其他 owner 不停止、不修改。作者仓库 `ae6554316ad1a2175920e330133a2f103408bf78` 的行为接入项目 IQ/骨干及固定 RNG，不使用裁剪；真实非有限或采样缺类隔离。

用户允许源缓存。MRIOR 和 Two-stage 的 B 使用 L_s 回放，Tweak checkpoint 含源注册特征；其余方法按各自源模型和目标 support 运行。B→C 旧原型为全部目标旧类 support 均值，是项目移植；最终报告需注明权限、实际源缓存和目标标签用量、随机数与骨干适配、混合 C 损失曲线缺项。

2026-11-18 06:18 UTC 期间独立读回：调度器持续运行，534 项预测完成、183 项评分完成、543 项待派发、22 项运行、35 项隔离，另有评分队列；该快照不包含已独立完成的 pilot 六项。已核实部分隔离是作者 drop-last 采样遗漏类别；Tweak M1→MoPC 的 P01/MIX01/2-1、seed 2026092701 则出现真实非有限 loss、cross entropy 及梯度，保留原始部分产物，不将它等同于有限梯度范数波动。隔离项不是完成的预测，未声明根因或修复；其他合法任务继续、评分逐项接续。

结束后核实 `COMPLETED_WITH_QUARANTINE`：1134 个完整队列任务中 1022 项独立评分、112 项隔离；前置 pilot 六个 K5/N5 格另已评分，与对应完整队列任务的其余 K/N 格互补。112 项包括 65 个作者 drop-last 原型提取遗漏类别和 47 个真实非有限，后者全部为 Tweak M1→MoPC。pilot 与完整队列合计 4088/4536 个 K/N 格已评分；不将隔离或 partial 计为完成。逐方法/任务、干净及星地配对汇总见本机 `E:/type10-7/local_artifacts/phase2_author_comparison_summary_20261118.json`；结果尚不具完整矩阵证据。

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
