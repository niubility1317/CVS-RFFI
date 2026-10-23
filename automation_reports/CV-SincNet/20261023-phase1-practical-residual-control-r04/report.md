# Full practical residual旧Torch兼容与健康进程接管r04

- run_id：`20261023-phase1-practical-residual-control-r04`
- group_id：`phase1-full-practical-residual-factorial`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只修复Torch1.11缺少的新精度设置API；接管健康G/cache，技术替代2个F启动失败及4消费者。

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

## 实际发布核验

状态 **VERIFIED / RUNNING**。实际执行 commit `dd2f4bfb389a9ff684fcb0d071d194543699d895`，owner PID 1925715，CWD/argv和物理GPU独立核对通过；旧r03 controller正常交接退出，健康G与CPU增强producer保留原PID/CWD/config。

两次观测间隔 193.9 秒，9 条相同 PID、9 条日志增长，其中 6 条为实际外部 Full F 训练。11合同已准备，真实checkpoint/source smoke PASS，U真值/target读取0。当前本任务 9 个GPU worker；GPU0–4分别为 {2: 1, 1: 1, 0: 1, 4: 3, 3: 3}，5/6/7保留，其他owner不动。无未解决失败。

有效阶段状态：`{"RUNNING": 9, "QUEUED": 1870, "COMPLETE": 1}`。600新F+4完整G保持，27既有T1只补读出；1880逻辑阶段、627行冻结预测/独立评分。14项聚焦检查和限定P0/P1/直接修复复核通过。尚无正式新性能或整链加速倍数；加速仅按后续真实吞吐校准。
私有逐进程证据保存在 local_artifacts/external_practical_residual_2x2_design_20261023/verification_1792746688.json；公开摘要为 launch_verification.json。
