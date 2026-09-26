# Phase2残差信道共享数据：7RX、6旧类+20新类

- run_id：`20260927-phase2-practical-data-manytx-s2026092705-r01`
- group_id：`phase2-practical-residual-shared-data`；类别：`diagnostic`；阶段：`Phase2-data-builder`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只按物理元数据覆盖选类，不读取模型或性能；36support池+30query/scene/class/RX，单物理记录单观测。

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

## 2026-09-27发布读回

VERIFIED：36036条received IQ、2100个split已生成；capsule_id=residual-noeq-ba667eee4fb061055e4c08b5。远端builder_report已读回。

代码版本：`7dd3e79a62c8f7e8487ceeff2dd9ae85ca1e74ee`。
