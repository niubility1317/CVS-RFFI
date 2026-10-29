# PAMF第三阶段：源辅助未知成员G10训练观测

- run_id：`20261029-phase3-star-source-exposure-g10-r01`
- group_id：`pamf-phase3-star-source-membership-training`；类别：`data`；阶段：`Phase3-source-G10-membership-data`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；完整实测与科学判定见下。

## 目的与对照

Exact6134 original sourceOE21 records, fixed G10, preserved nonuniform domains; outside-six membership only, no model or target.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

实际作业已完成，源端结果及权限边界见末尾追加。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 2026-10-29 source exposure G10 数据实测结果

`20261029-phase3-star-source-exposure-g10-r01` VERIFIED_COMPLETE，code `d319e3196f90113045a4f4f0958eeffe497f8785`；复用原6134 source辅助物理映射生成G10，21个非初始六类设备仅公开membership，无TX/K进入模型，未读取target。

只捕获terminal post：PID61824、1321B/18行；无live/GPUgrowth证据，foreignstop0。

这些是旧物理记录的新固定星地计算视图，不新增独立物理样本；与历史OE4重叠537条，非历史独立。后续三arm完整科学结果见 `docs/PAMF_PHASE3_STAR_REPRESENTATION_RESULTS_20261029.md`，不可把source资格写成跨RX或注册成功。
