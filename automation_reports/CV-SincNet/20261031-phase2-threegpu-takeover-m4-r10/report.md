# Phase2 对照队列限制三张显卡

- run_id：`20261031-phase2-threegpu-takeover-m4-r10`
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

## 三卡接管与直接故障修复

用户指定Phase2对照队列只分配3张卡，选择GPU1/5/7；每卡最多2个本任务worker。r08其他卡在途任务自然结束，保留原预测/评分路径、方法矩阵、全部已完成结果、16个已有ISSL失败及blocked后继；不恢复失败任务、不影响其他owner。

r09因暂停旧controller后短暂出现未识别child而安全中止，旧controller已恢复；该child随后被原controller记为正常完成。r10仅对这一具体竞态做有界重扫：每次重新枚举并核对全部子进程，持续未知仍中止并恢复旧controller，不接受未知进程。新root和原root独立，第一次真实源checkpoint无query smoke后立即进行接管。修复及失败/依赖阻塞重建已完成聚焦测试和独立定点复核。
