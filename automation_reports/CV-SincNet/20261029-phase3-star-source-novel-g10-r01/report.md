# PAMF第三阶段：源端留出设备G10星地观测

- run_id：`20261029-phase3-star-source-novel-g10-r01`
- group_id：`pamf-phase3-star-source-heldout`；类别：`data`；阶段：`Phase3-star-source-heldout-data`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；实际完成结果见下。

## 目的与对照

Preserve exact original source-heldout3150 physical records; one fixed G10 each, no model or target read.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

实际作业已完成，科学结果与证据边界见文末追加。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 2026-10-29 source novel G10 数据实测结果

`20261029-phase3-star-source-novel-g10-r01` VERIFIED_COMPLETE，code `fccbde38a407891db5acce776d6ad7af891f1341`；原3150条source-heldout物理记录按既有mapping生成G10 MIX01，无重采样、模型、OE训练IQ或target读取。实际PID12828、日志[515, 1262]B/18行、foreignstop0。

novel19-15/19-16/7-1各1050、source五RX×三日每TX单元70；这些物理记录及TX此前已暴露，新增的是固定星地实现，非新增独立物理样本。后续本源端12列评分为NO_PROMOTION，仅source eligibility、不代表跨RX/注册成功；详见 `docs/PAMF_PHASE3_STAR_NUISANCE_RESULTS_20261029.md`。
