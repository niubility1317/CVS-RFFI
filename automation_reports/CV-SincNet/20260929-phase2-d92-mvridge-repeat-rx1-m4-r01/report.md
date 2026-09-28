# D92-MVRidge冻结方法rx1透明重复基准

- run_id：`20260929-phase2-d92-mvridge-repeat-rx1-m4-r01`
- group_id：`d92-fixed-phase1-mvridge-repeated-benchmark`；类别：`cvs`；阶段：`Phase2-repeated-benchmark`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定Phase1与既有四相位received特征；全部注册类support以每物理样本总权重1拟合共享ridge判别头，单位ridge正则，无old/new组权或teacher。query逐样本四视图均值推理。

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

MVRidge PLANNED: frozen physical-weighted four-view supervised ridge1, complete matrix, source-free/cache-only CPU. Resource/path preflight VERIFIED. No experiment launched; root sole launch owner.

MVRidge local validation VERIFIED: 9 core tests, 13 predictor tests, 57 collector tests, and relevant orchestration/scoring/summary tests passed. Independent P0/P1 review found no blockers. Frozen method and full matrix are ready for committed publication; not launched at this entry.

RUNNING VERIFIED by readback_1790618402.json; release 71e4bef490bfae3acf907cc3f488b9ca62172a0f. Full matrix CPU run uses frozen pure received features and zero new source payload. No duplicate launch; await both cohorts before reading results.
