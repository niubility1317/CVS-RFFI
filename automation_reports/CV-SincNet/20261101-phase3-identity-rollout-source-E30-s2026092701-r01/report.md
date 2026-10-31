# 完整生成轨迹监督的 30 轮星地恢复网络

- run_id：`20261101-phase3-identity-rollout-source-E30-s2026092701-r01`
- group_id：`phase3-identity-rollout-g10-source`；类别：`cvs`；阶段：`Phase3-restoration-source`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从 received 与一对相反噪声出发完整反传两条四步生成轨迹，以源配对 IQ 和冻结 F0 特征监督；从零 E30，完成全部固定评测。

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

## 启动与首轮：VERIFIED

此前完整固定系统计时已完成后，固定代码 `54e74d7f7f8c01e5755909f3a417a2279cc4c28b` 在 N607 独立启动。真实 F0/G10 的 2 条源 L_s smoke 通过、checkpoint 严格重载；轨迹 IQ 和 F0 身份梯度范数分别为 4.0995 和 8.9804。

控制器 PID 2142900、worker PID 2142992、训练 PID 2142996，GPU 4；已从零完成第 1/30 轮，493 次更新，约 58.2 秒。训练与评测尚未完成，不据训练损失推断星地性能。
