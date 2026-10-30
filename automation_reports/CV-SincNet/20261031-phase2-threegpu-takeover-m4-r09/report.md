# Phase2 对照队列限制三张显卡

- run_id：`20261031-phase2-threegpu-takeover-m4-r09`
- group_id：`phase2-group1-native-AB-AC-concat-midPL`；类别：`comparison`；阶段：`Phase2-A-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原Phase2第一组只调度GPU1、5、7；接管健康GPU/评分进程，未选卡在跑任务自然结束；保留全部源模型、数据、方法、任务、失败及阻塞状态。

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

## 三卡接管计划

2026-10-31 用户指定Phase2对比队列只分配3张卡。选GPU1、5、7，依实际free显存发布，每卡最多2个本任务worker。安全接管r08已有预测及CPU评分进程，不停止、不重启、不修改科学配置；其他卡上的已运行worker自然完成，后续绝不派发至其他卡。原42570工作单元的路径、版本、依赖、已完成结果与16个既有ISSL失败及其blocked后继保持；不恢复这些失败任务，也不更改其他owner工作。

先做真实源checkpoint CPU无query smoke；确认旧CPU controller身份并暂停至T状态、核对原workers、重建完成/失败/阻塞、排除不明部分输出，然后仅停止旧CPU controller并启动新控制器；接管失败则恢复旧controller。独立P0/P1审查发现停止确认竞态，已修复并完成一次定点复核。自动运行与评分、同一方法矩阵、truth-last和原系统故障处理不变。非目标卡仍有在途worker期间，实际占用可能暂超过三张，直至这些worker完成。
