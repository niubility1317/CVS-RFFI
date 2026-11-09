# 四阶与记录排序模型源域配对诊断

- run_id：`20261109-phase1-cumulant-rank-paired-source-s2026092701-r01`
- group_id：`phase1-cumulant-rank-paired-source`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

两条预登记的固定源训练模型均冻结后，在同一来源合法 pilot 上比较波形误差和同记录特征。

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

## 固定源域比较

两条四阶统计源训练均固定最终第 20 轮后，本独立队列读取其冻结权重，在此前已探索的 672 条源 pilot 物理记录上比较原 DURA、四阶统计模型与记录排序模型。两模型的 32768 条训练 ID 排除这些 pilot；固定 DURA 可能见过部分 pilot。记录按 RX4/6、RX8 各七视图各 48 条组织，使用 512 条合法源训练 clean IQ 标准化功率与四阶特征。

指标为源配对 IQ 归一化误差、四阶统计误差，以及每组 48 条配对 clean 参考的同记录 top-1/top-5。clean 参考仅为源域诊断，记录匹配可能反映信号内容，不是 TX 身份准确率，也不是部署方案。诊断不读取 F、TX 标签、V、目标或目标真值，不更新模型；两个模型原定七景预测与评分不等待此诊断，也不得据此取消或选模。

队列等待两个已登记的 final20 源权重后选择 GPU4：至少 8192 MiB 空闲、总 GPU 进程少于 3，并检查 GPU4 没有其他本项目 worker；现有其他 owner 进程保持。真实两条源记录与固定权重 smoke 后执行 672 条。代码 55dedb9958addafe97e4c75a07d67f326d075253，完整命令、证据和停止规则见 experiment.json。聚焦数值／资源测试及独立 P0/P1 审查（资源修复定点复查）通过；队列尚待启动。

## 队列启动（2026-11-09）

发布归档本地与服务器 SHA-256 均为 a14a6035fe1f824e9c8a3528131858093615274fc6e70d5808309d615ed95938，服务器编译及导入通过。后台队列 PID 2427998 的工作目录、命令和等待日志已核实；当前两份最终源训练权重均未完成，队列处于等待状态且没有启动诊断 GPU worker。两个原训练与七景测试控制器保持独立。真实权重 smoke、源域结果及实际 GPU 占用尚待后续核验。
