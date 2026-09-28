# D92-BNNA冻结方法rx3透明重复基准

- run_id：`20260929-phase2-d92-bnna-repeat-rx3-m4-r01`
- group_id：`d92-fixed-phase1-bnna-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1和既有received IQ及support/query；当前任务support的4个确定性相位视图训练有界非线性适应器；完整物理样本fold重拟合全部状态；复用原D92预测。

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

BNNA formula frozen; support-only full physical folds; metadata ancestry reused; preflight verified; launch pending local implementation verification.

LOCAL_VERIFIED: frozen formula; core14 and full entry/audit integration passed; independent P0/P1 no blocker. Model deployment and incremental model transmission unknown. Existing checkpoint ancestry remains exact source-only scratch final200. Real native synthetic smoke runs before received IQ is opened.

VERIFIED RUNNING at readback_1790613570.json; actual release 0fce1af680992fac2c343762af0f07915e10b647. Both cohorts execute unchanged; no scores may be downloaded before both reach terminal state.
