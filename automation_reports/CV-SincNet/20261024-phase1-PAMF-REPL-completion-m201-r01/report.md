# PAMF-REPL 全部当前论文实验与匹配对照

- run_id：`20261024-phase1-PAMF-REPL-completion-m201-r01`
- group_id：`phase1-pamf-repl-complete-20261024`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

以选定 PAMF-REPL 替换当前半监督方法，重做匹配主比较、三核心、四内部机制、相关 L 细项、两层结构及效率。189 新 F、12 精确复用；DURA 不训练，仅补已有 G 的同 F 读出。

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

## 发布实际状态（2026-10-24）

VERIFIED：实际执行版本 `dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f`，修复发布 `20261024-phase1-PAMF-REPL-control-r02`。源码已push并独立核对remoteOID；单发布包传输SHA一致、远端编译和隔离导入PASS。

真实同契约E20 checkpoint源侧smoke PASS：768 L前向、256隐藏标签U、6学生视图、全部REPL机制有效梯度；target读取0/U真值读取0，无正式更新。

独立核查八卡共24个GPU worker，各卡不超过3，计入全机其他进程；16个同PID日志跨137.1秒增长。实际快照见launch_verification.json，不把快照当固定当前数。

所有当前F/读出/统计/时延及论文图表交付仍在执行；不能把发布或smoke当性能完成。DURA新增训练0，IQ恢复缓存新增0。189新F通过84唯一E20前段共享减少重复，目标分数不回流训练、参数、队列或选模。
