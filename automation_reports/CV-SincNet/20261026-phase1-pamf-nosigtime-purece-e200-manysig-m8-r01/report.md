# 新整体PAMF仅去除SIG-TIME：纯CE E200八种子匹配消融

- run_id：`20261026-phase1-pamf-nosigtime-purece-e200-manysig-m8-r01`
- group_id：`phase1-pamf-isolated-sigtime-ablation`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Scratch matched whole-network ablation: ReferenceCVS/NoExplicitCFO/compensation/default head retained, only SIG-TIME sequence path removed.

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

## 固定实现和验证

只去除SIG-TIME的6维投影、四窗路径增量/反对称面积与残差融合。ReferenceCVS参考响应、NoExplicitCFO（坐标27常数零）、内部补偿及原默认头保持。完整模型196001参数，消融177025参数。共享参数初始化逐项完全一致，移除后的features精确等于完整模型的ReferenceCVS基线特征。两项聚焦测试（含5种非法训练配置）通过；原训练器与严格预测检查复用。

本次只测clean；同时保留全168000与D0+未见RX42000子集。已有12种常见网络和完整SIG_TIME不重训、不重新预测。全部报告按实际8seed均值/样本SD，退化种子不剔除。新整体模型指此历史纯CE SIG_TIME，不混入后来T1/EMA/X3训练配方。唯一P0/P1审查结果及真实CP smoke随发布记录。

唯一独立P0/P1审查PASS。D0索引和实际日期已从原ManySig metadata核实；原104历史预测与新整体8预测均存在。新主表12常见网络+SIG_TIME完整网络+严格无SIG_TIME，共14×8=112行；先生成已有13×8的双跨结果，8行消融完整后补齐，未完成不冒充112行。原residual_fusion与native不作为当前PAMF主表行。

发布前资源调整：GPU0/2/3/4已被其他健康任务使用。首次预检在任何SCP/远端写入前拒绝，尚未启动本run。实际改用空闲GPU5/6/7，一卡一条本run worker，8seed队列，科学配方不变。
