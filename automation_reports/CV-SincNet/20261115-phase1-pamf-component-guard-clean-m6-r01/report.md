# Tri-SIG-TIME 逐损失梯度保护 clean 探索

## 固定方法和矩阵

基于 E21–E30 源梯度抽样中浅中层对比及外推差异相对 L 多次反向的观测，在十类和六源各登记三项：只保护浅层对比、只保护外推 CE 差异、同时保护两项，共六行；均使用原始 Tri 权重 6 和其余原损失。每个参数组内，先把选中 U 分量沿当前完整 L 梯度的负向分量投影掉，然后对合成的完整 U 梯度应用原分组投影和 U/L 范数上限。重排、模型、数据、E20 来源、后续 E21–E50 预算和自动 raw clean 预测／独立 truth-last 评分保持。

每步单独记录原始完整 U、分量保护后总 U、最终保护后 U 的范数或点积，分量冲突和修正量，以及标准冲突限幅、L CE、伪标签接受及类别与置信度分布；E30/E50 源 V ≥0.95 正确率只是代理，训练 U 真值不读取。已曝光 target 用于选择共同种子，首批目标结果也已可见；本新矩阵只根据先前记录的源梯度现象设计，目标结果不回流选择，全部结论仍属已曝光目标上的探索。

本批预定 GPU0、1，与另一个健康 owner 的 GPU2、6 分离；实启动前重新核实每卡进程与剩余显存，每卡总进程最多五个，已有任务不停止或热改。校验、发布、启动状态和最终指标以 events、实际运行产物为准。

# Tri-SIG-TIME 逐损失分量梯度保护 clean 对照

- run_id：`20261115-phase1-pamf-component-guard-clean-m6-r01`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

十类、六源各3个逐损失梯度保护候选，E20各自续训30轮，训练后自动clean测试。

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

## 发布核验

**VERIFIED / RUNNING**：执行 commit `1f2d13239f1cd7f43d1a71b6ea3314579f18557c`；一次归档传输 SHA 对比和远端编译通过，6 个真实 E20 checkpoint 无 query 冒烟全部 PASS。独立核实六个训练 worker 的 PID、CWD、命令行、GPU 0/1 和增长的逐步日志；每行首步已看到所注册的分量冲突字段和原始 U／分量处理后 U／最终 U 范数。所有 clean 预测及独立评分等待本行冻结。
