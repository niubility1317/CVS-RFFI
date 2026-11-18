# MoPC 作者仓库版 clean 对照 full

**范围变更，已停止并由 `20261118-phase2-mopc-author-clean-full-mix3-m3-r02` 替代。** 用户指定 clean 对照仅保留 MIX01–MIX03；本队列停止时 completed、scored 和 active 均为 0。下方八组 MIX 为历史预登记范围，并非当前待执行矩阵。

该队列覆盖三个数据池、MIX01–MIX08、七个接收机、三个模型种子、六个 MoPC 对照方法和 K5/10 × N5/10。pilot 六项各含四格；本队列 3018 项与 pilot 不重叠，合计 3024 项、12096 格。全部使用 clean support 和 clean query；逐任务固定预测后独立 truth-last 评分。只有 pilot 六项均评分且无隔离时接续。

2026-11-18 06:18 UTC 独立读回：调度器 PID 791975，真实 CWD 为预登记 release；六方法、三个种子共 18 项真实源 checkpoint 无 query smoke 均通过。状态 `WAITING_PREDECESSOR`，pending 3018、active 0、completed 0、scored 0；此时 clean 预测尚未启动。

固定作者仓库 `ae6554316ad1a2175920e330133a2f103408bf78` 行为，项目 IQ/骨干与固定 RNG 适配，乱序、drop-last、无梯度裁剪。历史 MoPC clean 注册结果的采样口径不同，不混用；源 checkpoint 可复用，MRIOR 和 Two-stage 使用获授权 L_s 源缓存，Tweak 的源注册特征存于 checkpoint。每 GPU 的 MoPC 进程最多 4 个，保守按全部可见进程计数；每项派发前空余至少 6000 MiB，不干预其他 owner。缺类/非有限逐项隔离并保留产物，系统技术故障停止派发。

- run_id：`20261118-phase2-mopc-author-clean-full-m3-r01`
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
