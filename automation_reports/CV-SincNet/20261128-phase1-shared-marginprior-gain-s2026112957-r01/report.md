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
状态 SOURCE_SCREEN_SCORED，两侧均未达到晋级条件。源 U clean 教师高置信且与固定锚点类别一致时增加分类间隔保持，记录符合条件的源训练曝光；不读取 U 真 TX。其他源授权、各数据集独立权重、1% 物理先验与 5% 增益约束和冻结识别器不变；完成后自动生成同包预测并独立评分。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 教师间隔保持 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 34 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 330 |

LoRa 的 R69 对照为 LoRa 源训练结构适配，WiSig 为原固定 DURA v1。LoRa 相对实增益改对 0、改错 0，净 0，95% 配对区间 0 至 0，半区 0/0；WiSig 相对 R17 改对 3、改错 5，净 -2，区间约 -7.025 至 +3、半区 0/-2；clean 均保持。两侧未通过预登记门槛。

相对同源池、seed 和预算的无间隔项 1% 有界版，LoRa clean 和低城区正确包相同；WiSig clean 正确包相同、低城区改对 2、改错 2，净 0，区间 -4 至 +4。逐 TX 及其他对照见 `paired.json`、`readback/*/screen-score.json`；源 V 面板反复曝光。相关字段 teacher/mse 分别指本候选和上一版，均使用教师指导。

源 U 教师伪类别间隔项符合条件的训练曝光 LoRa 1522/2000、WiSig 14423/16000；含 smoke 的教师样本前向 8008/64064，隐藏 U 的真实 TX。增益 11,394 参数、R17 4,665,073 参数，其中末步先验 835,842 可训练，另有冻结参考与实增益专家。含模拟池耗时 1070.15/667.92 秒，PyTorch 峰值约 517.8/206.0 MiB；真实推理时延和正式七景待后续候选通过筛查。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
