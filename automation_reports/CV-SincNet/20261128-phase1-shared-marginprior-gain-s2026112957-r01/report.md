# LoRa 与 WiSig 共同有界先验和源教师分类间隔保持筛查

- run_id：`20261128-phase1-shared-marginprior-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源 clean 教师高置信且和固定锚点预测一致时，训练恢复结果保持锚点分类间隔。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

远端发布 VERIFIED：代码 `c26221d3e9359176e01907eeb59abb41d2dcaa59`，单归档本地/远端 SHA 一致、远端编译通过；控制器 4082339、LoRa worker 4082639/GPU 2、WiSig worker 4082640/GPU 3。两侧真实权重源 U smoke 通过，已核实进程 CWD、GPU 与日志增长。独立 P0/P1 审查未发现阻断问题，针对性测试 7 项通过。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
状态 RUNNING。源 U clean 教师高置信且与固定锚点类别一致时增加分类间隔保持，记录符合条件的源训练曝光；不读取 U 真 TX。其他源授权、各数据集独立权重、1% 物理先验与 5% 增益约束和冻结识别器不变；完成后自动生成同包预测并独立评分。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
