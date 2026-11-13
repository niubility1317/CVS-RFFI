# Phase2 第二组：Hu 源辅助补充三个种子

- run_id：`20261103-phase2-csource-hu-aux-m3-r02`
- group_id：`phase2-group2-csource-hu-aux`；类别：`comparison`；阶段：`Phase2-source-only`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Train Hu source auxiliaries for remaining three model seeds, reusing existing seed-one auxiliaries.

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

## 2026-11-03 启动读回


## 2026-11-03 源辅助训练完成

9 项源辅助模型全部完成，逐项核验 C checkpoint、模型 seed、L_s 权限、200 轮/5000 步和无目标接触。seed01 已由上一个 run 复用。这是源端辅助训练，目标预测及评分尚须独立完成。

| C 主干 | 模型 seed | 源训练秒 |
| --- | ---: | ---: |
| mopc_hr | 2026092702 | 58.49 |
| mopc_hr | 2026092703 | 44.95 |
| mopc_hr | 2026092704 | 45.57 |
| orthogonal | 2026092702 | 44.37 |
| orthogonal | 2026092703 | 41.69 |
| orthogonal | 2026092704 | 57.83 |
| lora_triplet_knn | 2026092702 | 42.67 |
| lora_triplet_knn | 2026092703 | 41.38 |
| lora_triplet_knn | 2026092704 | 58.53 |
