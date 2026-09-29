# D92-BranchInteraction冻结方法rx1完整重复基准

- run_id：`20260929-phase2-d92-branch-interaction-repeat-rx1-m4-r01`
- group_id：`d92-fixed-phase1-branch-interaction-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1和三臂support诊断前预设的交互核，复用BranchRidge原始received单view五块特征；当前row support一次kernel ridge拟合，全部K同式。

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

PLANNED: same frozen interaction candidate, full existing matrix, CPU-only immutable BranchRidge cache reuse and two registered baselines. Read-only cache/output/resource preflight VERIFIED; root is sole launch owner. No query scores read and no experiment launched yet.
