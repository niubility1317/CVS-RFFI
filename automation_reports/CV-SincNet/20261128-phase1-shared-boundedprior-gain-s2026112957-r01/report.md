# LoRa 与 WiSig 共同实增益及有界 R17 末步先验筛查

- run_id：`20261128-phase1-shared-boundedprior-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对可训练末步先验施加原 R17 波形 1% 均方根限幅，训练与推理使用同一限制。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

远端发布 VERIFIED：提交 `eb58403887570a2550fbdf83d4f25dca1289af98`，单归档本地/远端 SHA 一致，远端编译通过。控制器 PID 4068748，LoRa 4068769/GPU 6、WiSig 4068770/GPU 4；两侧真实权重源 U smoke 均通过，PID/CWD/GPU 和日志增长已核实。独立 P0/P1 审查未发现阻断缺陷；相关本地测试 6 项通过。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
状态 RUNNING；逐包原 R17 波形均方根 1% 限制用于训练和推理，权重由各自源 U 教师指导获得；先前 WiSig 源 V 暴露与退化促成假设。后续自动固定预测并独立评分。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 有界先验 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 34 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 330 |

LoRa 的 R69 对照为 LoRa 源训练结构适配，WiSig 为原固定 DURA v1。LoRa 相对实增益改对 0、改错 0，净 0，95% 配对区间 0 至 0，半区 0/0；WiSig 相对 R17 改对 4、改错 6，净 -2，区间 -8 至 +4、半区 0/-2；clean 均保持。两侧未通过预登记门槛。

相对使用相同源模拟池、seed、预算且不限制物理先验变化的上一版，LoRa clean 与低城区正确包相同；WiSig clean 正确包相同、低城区改对 10、改错 1，净 +9，区间 +3 至 +16，半区 +6/+3。这个对无界版本的改善未达到相对所有强对照的晋级要求。逐 TX 和所有配对结果见 `paired.json`、`readback/*/screen-score.json`；源 V 已反复曝光，1% 假设受上一版结果启发。

各侧增益 11,394 参数、R17 4,665,073 参数，其中末步先验 835,842 可训练，另有冻结原 R17 参考及实增益专家；参数和推理成本须计入所有执行路径。LoRa/WiSig 源 U 曝光 2000/16000，含 smoke 的教师样本前向 6006/48048；含模拟池耗时 1023.68/588.78 秒，PyTorch 峰值约 517.8/206.0 MiB。真实推理时延与正式七景待后续有资格候选完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
