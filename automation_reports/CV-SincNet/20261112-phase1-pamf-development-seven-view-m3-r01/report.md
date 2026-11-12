# 已完成 10 类开发模型的 clean 与六场景补测

- run_id：`20261112-phase1-pamf-development-seven-view-m3-r01`
- group_id：`phase1-pamf-branch-prototype-tuning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED

## 目的与对照

对三枚已完成模型执行冻结默认头直接预测和逐行独立评分，不重新训练。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

三枚冻结模型均完成直接识别预测和独立 truth-last 评分。以下按 cross_rx 与 cross_both 的样本数合并，每景 123021 条；种子为 2026092701，单位为百分比。

| 原型温度 | clean | high | mid | low suburban | high urban | mid urban | low urban | 六景均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 82.728152 | 73.08183 | 69.07439 | 65.64652 | 68.95489 | 52.17483 | 39.97610 | 61.484760 |
| 0.2 | 82.918364 | 73.16556 | 69.16787 | 65.76926 | 69.04837 | 52.29758 | 40.05007 | 61.583117 |
| 0.4 | 82.814316 | 73.01761 | 68.96709 | 65.54491 | 68.85654 | 52.10086 | 39.77370 | 61.376784 |

训练使用过源域有标签数据内部留出（训练 5228、留出 1305），与完整源数据对照的训练预算不同，不能作为公平匹配对比。测试集此前已曝光，属于单种子探索性结果；完整数据续训矩阵已预先固定，不依据此结果修改。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
