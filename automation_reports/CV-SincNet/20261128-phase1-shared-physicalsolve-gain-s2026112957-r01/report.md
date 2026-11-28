# LoRa 与 WiSig 全频实增益与 DURA 物理求解融合筛查

- run_id：`20261128-phase1-shared-physicalsolve-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对共同整包频域融合结果及锚点运行相同加权物理求解，以求解结果之差修正锚点。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

远端发布 VERIFIED：代码 `3bc573221bcfe33f69bf03830349fdb9cbf6777e`，单归档本地/远端 SHA 一致、远端编译通过；控制器 4118337、LoRa worker 4118351/GPU 2、WiSig worker 4118352/GPU 3。两侧真实 checkpoint 源 U smoke 通过，PID/CWD、GPU 使用和日志增长已核实；独立 P0/P1 审查未发现阻断问题，相关测试 8 项通过。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
状态 RUNNING。整包实增益的融合与锚点分别经当前包信道作用和方差的两步加权求解，以求解差修正锚点；两数据集同一规则、各自独立源训练权重，训练推理一致。固定筛查预测后独立评分，已曝光源 V 未参与训练。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
