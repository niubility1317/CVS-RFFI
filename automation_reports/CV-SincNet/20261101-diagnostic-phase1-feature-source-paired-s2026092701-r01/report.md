# Phase1 特征条件模型源域配对诊断

- run_id：`20261101-diagnostic-phase1-feature-source-paired-s2026092701-r01`
- group_id：`phase1-feature-source-paired-diagnostic`；类别：`cvs`；阶段：`Phase1-source-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

比较固定 DURA 与已完成的特征条件网络，在同一源训练样本上的干净基座一致性和波形误差。

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

## 源训练逐条配对结果

固定 512 条源训练物理记录（L_s 128、U_s 384）与两个源视图，用同一冻结 Tri-SIG-TIME SSL 比较原 DURA 和已完成 E30 的特征条件模型。逐条修复/损伤以及场景、状态、角色组求和经独立核对。

| 源视图 | 样本 | DURA 与 clean 教师一致 | 特征模型一致 | 修复 | 损伤 |
|---|---:|---:|---:|---:|---:|
| 01 全部 | 512 | 491 | 483 | 14 | 22 |
| 01 BLOCKED | 83 | 65 | 59 | 11 | 17 |
| 20 全部 | 512 | 492 | 492 | 13 | 13 |
| 20 BLOCKED | 62 | 48 | 46 | 8 | 10 |

全样本 IQ NMSE 在源视图 01 从 0.020213 增至 0.022439，视图 20 从 0.019440 增至 0.021307。这表明至少在这一小批训练源记录上，当前目标未稳定保住原 DURA 判断；不能单独归因于测试域泛化。clean 教师一致性不是 TX 准确率，样本来自训练集且两个视图共享物理记录。诊断未读取 TX 标签或测试数据。
