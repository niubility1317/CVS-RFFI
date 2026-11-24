# 导师补实验：优化自适应 Tri 与普通 PL

- run_id：`20261124-phase1-pamf-advisor-adaptive-m48-r01`
- group_id：`phase1-pamf-advisor-adaptive-versus-pl`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

24 个数据设置逐一配对，固定模型种子 2026092701。总标签预算为 630、3150、6300 条（L 与 V 合计）。源接收机组保持等量 L/V/U，覆盖 1 至 6 源、16 个组合。时间组为四折轮换留日及最后一天的向前测试。每个数据设置使用相同测试样本，依次计算 clean 与六种星地场景识别指标，预测完成后独立评分。

新矩阵含 48 个方法结果。25 行由本 run 执行训练或续训，其中 9 行启用优化配方，16 行补充普通 PL。其余 23 行通过同种子、同数据、同配方和同测试身份核对后，复用或等待原 run 的对应结果。优化策略来源为 `e479c466a6a0b1862a17407f6ad5f4d2ec6aedd2`。单源沿用混合对比分支，3150 条沿用接收机均衡分支。根据源接收机、采集日和类别的标签覆盖，630 条及五、六源设置触发 E20 标签锚定。六源分支将该锚定与其既有同日配对配方同时启用，这是新数据设置上的组合。其它设置按照优化版的回退规则沿用原版 Tri。原队列保留以完成匹配对照及回退结果。

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
