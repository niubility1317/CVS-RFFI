# 独立时频恢复：源信道一致性七景测试

- run_id：`20261108-phase1-direct-timefreq-source-physics-ground-test-s2026092701-r01`
- group_id：`phase1-direct-timefreq-source-physics`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

独立时频网络在源训练使用已知模拟信道构造辅助损失；部署只读取单条 received IQ。

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

## 预登记补充

固定最终权重后，接冻结 Tri-SIG-TIME SSL Main split 默认头；clean 和六种 practical 每景 168000 条，全部预测完成才独立 truth-last。报告相对 DURA 的逐景差异、clean 损伤与相对原始 clean 的 3 个百分点及相对 3% 两种门槛。地面模拟测试此前已探索，不是独立盲测；结果不回流训练或选模。

## 七景最终评分

七景各 168000 条预测固定后，已完成独立 truth-last 与 DURA 配对评分。六个受损场景均低于 DURA，均未达到相对下降 3% 门槛。逐场景、配对纠错及 TX/RX/day 分组见 [结果分析](analysis.md)。测试为已曝光的 Phase1 地面模拟结果。
