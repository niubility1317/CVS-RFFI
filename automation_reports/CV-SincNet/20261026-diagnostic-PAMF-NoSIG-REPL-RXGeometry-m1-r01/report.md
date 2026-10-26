# REPL-v2身份保护RX外推方向的完整源几何证伪

- run_id：`20261026-diagnostic-PAMF-NoSIG-REPL-RXGeometry-m1-r01`
- group_id：`pamf-repl-v2-20261026`；类别：`analysis`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只读既有NoSIG E30的L6300/V27000缓存特征，L-only类别平衡共享RX方向、身份子空间投影、跨类留一与V方向迁移；零训练/模型/IQ/U/target。

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


2026-10-26：CANCELLED_BEFORE_EXECUTION。用户改为完整 PAMF 保留 SIG_TIME 单视图；本 NoSIG 几何诊断未读取输入、未执行，模型/IQ/U/target/更新均 0。没有科学 PASS/FAIL，不转用其旧特征作为新架构依据。草稿算法与测试保留在私有目录。
