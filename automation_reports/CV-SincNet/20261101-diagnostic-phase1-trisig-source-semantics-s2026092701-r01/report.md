# Phase1 源训练样本的基座语义稳定性诊断

- run_id：`20261101-diagnostic-phase1-trisig-source-semantics-s2026092701-r01`
- group_id：`phase1-trisigtime-source-semantics`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只使用 Phase1 L_s/U_s 配对 clean 与模拟接收 IQ，分四档检查冻结 Tri-SIG-TIME SSL 的分类一致率、特征余弦和 KL；不读真实标签或测试数据。

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

## 源域诊断结果与执行边界

已核实 N607 `complete.json` 含四组各 128 条 Phase1 源训练样本，F 更新、TX 标签读取和测试 IQ 读取均为零。四组按该源视图中的模拟噪声方差四分位划分，固定种子抽样；无输出反馈选样。

| 噪声方差分组 | clean／received top-1 判断一致率 | 特征余弦均值 | clean→received KL 均值 |
|---|---:|---:|---:|
| 最低四分位 | 98.44% | 0.9838 | 0.0362 |
| 第二四分位 | 96.88% | 0.9599 | 0.0960 |
| 第三四分位 | 96.88% | 0.9549 | 0.0659 |
| 最高四分位 | 60.94% | 0.8521 | 1.0190 |

这些数值只是源训练样本的冻结基座判断一致率，不是真实分类准确率，也不代表测试集或其他信道。结果提示应研究直接保持分类分布，但不能据此声称恢复改善。

执行状态 **PARTIAL**：完整 JSON 已写入，字段、各组数量和值域独立读回核验；之后打印日志时把 `flush` 错传给 `json.dumps`，进程异常退出。原日志与产物保留，故不把该进程标记为正常完成。问题已本地复现，并在代码版本 `0ab6cc9e7532380474c0dd319d7b294053e22f4c` 修复、验证；没有重跑或覆盖该诊断。
