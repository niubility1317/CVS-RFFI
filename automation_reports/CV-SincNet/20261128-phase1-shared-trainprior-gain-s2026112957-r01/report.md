# LoRa 与 WiSig 共同实增益与 R17 末步先验联合训练筛查

- run_id：`20261128-phase1-shared-trainprior-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

冻结粗恢复、信道作用估计及前两步先验，共同训练末步先验与实增益网络。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

远端发布 VERIFIED：提交 `de54a6cfda960530478e0ef296811709b312aa0d`，发布归档本地/远端 SHA 一致，远端编译通过。控制器 PID 4055343，LoRa worker 4055356/GPU 2、WiSig worker 4055357/GPU 3；真实权重源 U smoke 均通过，两个子进程 GPU 占用和日志增长已核实。独立 P0/P1 审查未发现阻断缺陷；本地针对性检查 14 项通过。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
状态 SOURCE_SCREEN_SCORED；两侧固定预测后独立 truth-last 评分，均未通过预登记筛查。训练期间未读取源 V 或目标。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 末步先验联合训练 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 34 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 321 |

LoRa 的 R69 对照为同数据集源训练结构适配，WiSig 为原固定 DURA v1。LoRa 相对实增益正确包相同，改对 0、改错 0、净 0，95% 配对区间 0 至 0、半区 0/0；WiSig 相对 R17 改对 1、改错 12、净 -11，区间 -18 至 -5、半区 -6/-5；clean 均保持。两侧未通过预登记门槛。

相对同模拟池但不同 batch、训练目标及可训练先验的原 5% 教师增益版，LoRa 低城区改对 0、改错 1、净 -1（区间 -3 至 0）；WiSig 改对 0、改错 11、净 -11（区间 -18 至 -5）。WiSig 源 TX 1、2、3、4 分别退化 4、2、4、1 包。源 V 面板反复曝光且参与方案研发；逐 TX 和全部对照见 `paired.json`、`readback/*/screen-score.json`。原自动比较元数据误称预算相同，`paired-original.json` 保留原件，`paired.json` 更正说明。

共同增益网络 11,394 参数，各自 R17 4,665,073 参数，其中末步先验 835,842 可训练；全部专家、教师及附加成本另计。LoRa/WiSig 源 U 训练曝光 2000/16000，含 smoke 的教师前向样本 6006/48048；含模拟池训练耗时 874.51/466.18 秒，PyTorch 峰值约 499.4/187.9 MiB。真实推理延时与正式七景均待后续有资格候选完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
