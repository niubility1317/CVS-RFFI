# LoRa 与 WiSig 教师逐频恢复的同 seed 无教师对照

- run_id：`20261128-phase1-shared-matched-direct-s2026112945-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

与教师逐频版同结构、seed、源 U 配对、两路专家、物理残差、波形和 clean 保持损失及 1000 步，省略教师 KL。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 `a75b46c32cd4aaf5cabc8e7d8a95a1d45315b2f8`；控制器 3809415，LoRa worker 3809490 / GPU 3，WiSig worker 3809491 / GPU 6。两侧真实专家首步和有限非零梯度通过。 N607 preflight、归档 SHA、远端编译、PID/CWD/GPU 和日志增长已核实；独立 P0/P1 审查通过，其他 owner 进程保留。

## 结果与覆盖

固定源 V 两景预测已独立 truth-last 评分；正式测试尚未进行。与教师逐频版同 seed 2026112945、相同 SharedTwoPriorGain、专家、源 U 模拟、batch、1000 步、优化器和两个非教师损失，省略 KL；新训练不读 F、U TX、V 或目标。LoRa 实增益专家历史继承授权 E104 指导与伪标签。不同 GPU 的浮点轨迹可能不同。 源 V 面板已多轮曝光，训练不读 U TX、V、目标 IQ 或真值；冻结后独立评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 教师逐频 | 匹配无教师对照 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 | 59 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 30 | 16 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 193 | 144 |

LoRa DURA 列为其源训练 R69 架构适配，WiSig 列为原固定 DURA v1。LoRa 教师逐频版相对实增益改对 2、改错 6，净 -4 包，配对 bootstrap 95% 区间 -10 至 +1；相对 R69 适配净 -3。WiSig 相对 R17 改对 10、改错 149，净 -139，区间 -160 至 -118；相对 DURA v1 净 -120。WiSig 源 TX 类别 2 为 36/65、类别 4 为 32/60，对应 R17 分别为 65/65 和 60/60，存在明显单类退化。两侧均未通过全部对照的晋级门槛。

预登记匹配对照独立同包分析：LoRa 教师版相对无教师版改对 19、改错 5，净 +14/60，95% 区间 +5 至 +22，clean 净 +1；WiSig 改对 71、改错 22，净 +49/360，区间 +31 至 +67，clean 净 0。LoRa 源 TX 类别 3、5 及 WiSig 类别 0、5 相对匹配对照各少 1 至 4 包，逐类见 `paired.json`。两组源生成 seed、架构和预算匹配，不同 GPU 浮点轨迹可能不同；源 V 已多轮曝光，不构成未曝光目标测试。

恢复修正网络各 11,394 参数。教师版 LoRa/WiSig 分别源 U 曝光 4,000/24,000、教师样本前向 8,000/48,000，训练 461.14/261.08 秒，PyTorch 峰值约 174.8/78.0 MiB；匹配无教师版 321.38/214.18 秒、约 122.1/55.3 MiB。教师和专家历史成本另计。LoRa 原 scipy 输入前向的可微 WideTracker 替代梯度仅为训练反向近似；端到端推理时延、正式七景和最终验收均待完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
