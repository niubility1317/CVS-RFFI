# PAMF第三阶段：源域类别与接收机留出开发

- run_id：`20261029-phase3-source-holdout-s2026092701-r01`
- group_id：`pamf-phase3-source-holdout`；类别：`cvs`；阶段：`Phase3-source-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从零四类PAMF，留出source TX0/1及RX8；L-only cleanCE20完整遍历/540步，完整V冻结导出后比较8种源拒识分数，零target访问。

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

## 划分seed登记更正（2026-10-29）

原登记将模型seed2026092701误记为source split seed。已依据实际source_contract.json更正为392005；模型seed保持2026092701。实际数据、物理ID、checkpoint和已完成预测/评分不变，不重训或重测。该更正只修复元数据，不将历史结果重新解释为新实验。

## 冻结设计与本地验证

实际code/config commit：`1d9f4bd57665641d65cb6d33c9183d5d44c03277`。完整设计见`docs/PAMF_PHASE3_SOURCE_DEVELOPMENT_20261028.md`，实体配置见`configs/pamf_phase3_source_dev_20261028.json`。全部73项Phase3聚焦测试PASS，本source候选一次独立P0/P1审查PASS；真实四类CPU构造/反向已验证，不将其称真实源性能。

数据兼容性独立VERIFIED：实际source contract的split_seed392005、rx_day15域和source RX/day一致；真实L/V metadata分别6300/27000、eq_i=1、90个TX×RX×day格全为70/300，角色ID逐行对应contract。该核对未打开IQ/U/target，不生成额外数据hash或重复capsule验证。

训练严格留出source labels0/1和RX8，从原L_s只取3360条，原V27000完整仅最后eval导出。无checkpoint继承或teacher；固定20完整遍历/540调用、539次非零LR，不早停、不用V训练/更新BN/原型。训练固定后L拟合4类中心与共享收缩协方差，V known校准各分数99%保留阈值，proxy unknown只作源规则评价。完整V包含RX8 known，因此RX8只对参数训练留出，源规则结果不是全未见RX阈值确认。

首fold必须按预登记95%/5%/2pp条件通过才扩展其余两个source fold；失败写NO_PROMOTION，不根据首轮target表改阈值/选规则。当前无target访问、无目标无标签适应/发现/注册结论。首步合法source L scratch smoke通过后自动训练/导出/源规则评分；输出独占，不停止或改动其他用户进程。
