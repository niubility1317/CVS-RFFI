# 外部对照 Phase1 七场景同样本补测

- run_id：`20261031-phase1-external-matched-nodura-m40-r02`
- group_id：`phase1-external-group1-matched-seven-views`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用10种对照各4个本轮源权重，使用PAMF原同一168000样本及固定七场景，只进行原生冻结直接识别、不接DURA；预测后独立truth-last评分，逐接收机及种子汇总。 用户要求立即启动，扩展GPU总并发至每卡4（含其他owner），本run最多1。

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

## 直接启动计划

用户要求补测直接启动。本run替代仅排队的r01，10方法×4seed、PAMF原168000样本及clean＋六practical，冻结原生分类、无DURA，所有视图预测固定后独立truth-last；逐RX/day/TX汇总。切换前核实r01 PID、命令及零活跃/零完成，精确停止本任务controller，保留原记录和健康其他owner。

每GPU本任务至多1个预测进程，所有owner的CUDA与待启动PID计入总数；新进程启动前总数必须小于4、剩余显存不少于12000MiB。Phase2仍运行时须先核实该卡两个已存活Phase2预测worker均在本次GPU快照中，以避免旧快照与Phase2补位交错；CPU评分最多1。其他owner进程不停止、不热改。首次真实源权重无query smoke通过后自动启动，技术失败停止新预测并保留产物。

独立P0/P1审查及一次定点复核指出Phase2补位与GPU快照次序窗口，已要求两个Phase2 worker PID都属于同一快照，并以针对性交错测试验证。

## 发布读回（2026-10-31，香港时间）

VERIFIED：执行代码commit `01907ff46b01fa991862a69a32f64c8c08d59e9c` 已push并独立核对；一次发布归档传输比较和远端编译通过。r01 controller经PID/CWD/argv核对并暂停后再次核实80项排队、零子进程、零输出，精确停止且确认退出。r02 controller PID 1015035 存活；十个实际源权重source-only CPU smoke PASS。02:06首批7个预测worker在GPU0/1/3/4/5/6/7上运行，PID/CWD/命令和CUDA占用均已读回；日志分别产生cross_rx clean及practical预测输出。Phase2 controller及其worker继续运行。本次为预测启动证据，尚未产生完整row评分或最终结果。

## 四种子结果（2026-10-31 02:20，香港时间）

状态 VERIFIED / ANALYZED：40个外部模型的预测及独立truth-last评分全部完成，无失败。PAMF复用对应四seed的独立评分。每景每种子合并cross_rx 126000及cross_both 42000的混淆矩阵；每接收机每景24000。以下为四seed准确率均值，六景均值对六个场景等权，不接DURA。PAMF原配方无星地拼接增强，外部方法有源域拼接增强和中期普通PL，不宣称训练增强一致。

| 方法 | clean | 郊区 high | 郊区 mid | 郊区 low | 城市 high | 城市 mid | 城市 low | 六景均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PAMF | 86.63 | 80.06 | 77.57 | 75.19 | 76.59 | 62.52 | 51.56 | 70.58 |
| ProtoNet CDA | 69.63 | 61.20 | 60.91 | 60.30 | 60.72 | 57.70 | 53.58 | 59.07 |
| MRIOR-SDA | 68.79 | 62.57 | 62.19 | 61.63 | 62.04 | 58.79 | 54.55 | 60.29 |
| DADDA-SDA | 71.30 | 66.49 | 65.76 | 64.85 | 65.43 | 60.45 | 55.08 | 63.01 |
| Two-stage SDA | 72.20 | 70.58 | 69.28 | 68.03 | 68.46 | 59.85 | 52.66 | 64.81 |
| Hu2024 特征分离 | 16.67 | 16.67 | 16.67 | 16.65 | 16.67 | 16.63 | 16.57 | 16.64 |
| POSTER FT | 68.96 | 63.15 | 61.06 | 59.50 | 59.59 | 46.85 | 38.41 | 54.76 |
| RadioNet ADA+kNN | 72.25 | 70.45 | 69.47 | 68.47 | 69.03 | 62.51 | 55.99 | 65.98 |
| Orthogonal SEI | 70.23 | 71.56 | 69.67 | 68.08 | 68.83 | 57.82 | 49.28 | 64.20 |
| ISSL | 71.98 | 70.33 | 69.70 | 68.88 | 69.16 | 63.91 | 58.05 | 66.67 |
| LoRa Triplet+kNN | 67.09 | 56.76 | 54.91 | 53.09 | 55.16 | 47.31 | 40.68 | 51.32 |

按逐score混淆矩阵独立重算总计616个方法/视图/接收机汇总，与服务器summary逐项比较最大误差小于2e-14个百分点。完整逐seed、逐景、逐RX精确结果位于本机 `local_artifacts/phase1_external_matched_20261031/`；该目录不提交原始实验输出。
