# LoRa 与 WiSig 整包实频增益与有界展开先验筛查

- run_id：`20261128-phase1-shared-fullspectrum-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

统一在原生整包 FFT 频点学习实增益，并结合 1% 有界 R17 最后先验和物理伴随残差。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

远端发布 VERIFIED：代码 `c936d2e64a705fdea82de9da403547b69580dc3d`，单归档本地/远端 SHA 一致、远端编译通过；控制器 4097199、LoRa worker 4097214/GPU 6、WiSig worker 4097215/GPU 4。两侧真实权重源 U smoke 通过，GPU 使用和日志增长已核实。独立 P0/P1 审查未发现阻断问题，相关测试 8 项通过。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
状态 RUNNING。共同实增益直接操作完整原生包频谱，LoRa FFT 长 8192、WiSig 长 256；保留有界展开先验、物理伴随残差和各自源 U 教师，分别训练权重，推理无需教师。源 V 面板反复曝光；固定预测后独立评分。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
