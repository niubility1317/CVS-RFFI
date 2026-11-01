# Phase1 类别条件候选与冻结基座核验恢复网络

- run_id：`20261101-phase1-class-hypothesis-E30-s2026092701-r01`
- group_id：`phase1-class-conditional-restoration`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源域训练类别后验与类别条件受限波形候选，冻结 Tri-SIG-TIME SSL 核验并保守回退。

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

## 固定训练完成

类别条件模型已完成预先固定的 30 轮、14790 次源域更新。使用真实冻结 Tri-SIG-TIME SSL 和 DURA 独立重载最终模型，在两条源记录核对输出确定性和 5% 校正限制；最终检查点 SHA-256 为 `670711b517a71440bd91da67816286a444cb91753e22889b425f43df91256dc6`。七视图预测进行中，目前没有完整测试结果。
