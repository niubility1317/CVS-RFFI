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
状态 SOURCE_SCREEN_SCORED，两侧均未达到晋级条件。共同实增益直接操作完整原生包频谱，LoRa FFT 长 8192、WiSig 长 256；保留有界展开先验、物理伴随残差和各自源 U 教师，分别训练权重，推理无需教师。源 V 面板反复曝光；固定预测后独立评分。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 整包频谱增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 33 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 332 |

LoRa 的 R69 对照为 LoRa 源训练结构适配，WiSig 为原固定 DURA v1。LoRa 相对实增益改对 0、改错 1，净 -1，95% 配对区间 -3 至 0，半区 0/-1；WiSig 相对 R17 改对 4、改错 4，净 0，区间 -5 至 +6、半区 +1/-1；clean 均保持。两侧未通过预登记分半净收益门槛。

相对同源模拟、seed 和预算的分组频谱 1% 有界版，LoRa 低城区改对 0、改错 1，净 -1（区间 -3 至 0）；WiSig 改对 4、改错 2，净 +2（区间 -3 至 +7，半区 +1/+1），clean 正确包一致。逐 TX 及其他对照见 `paired.json`、`readback/*/screen-score.json`；面板反复曝光。teacher/mse 字段分别表示本候选与分组版，均使用源教师。

增益网络 11,394 参数、R17 4,665,073 参数，其中末步先验 835,842 可训练，另有冻结参考与实增益专家。LoRa/WiSig 源 U 曝光 2000/16000，含 smoke 教师样本前向 6006/48048；含模拟池耗时 1040.01/569.80 秒，PyTorch 峰值约 517.8/205.2 MiB。实测推理延时及正式七景待有资格候选完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
