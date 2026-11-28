# LoRa 与 WiSig 共用 R17 锚定教师逐频恢复筛查

- run_id：`20261128-phase1-shared-r17-teacher-direct-s2026112949-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

共同逐频实增益网络从 R17 输出起步，将实增益专家及物理残差作为修正方向，以合法源 U 冻结教师 KL、波形及 clean R17 保持损失训练。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 `32f939298e93abd8104f941e88489b85b37fd2f4`；控制器 3815826，LoRa worker 3815839 / GPU 0，WiSig worker 3815840 / GPU 5。真实专家、R17 初始输出和教师 KL 对网络的有限非零梯度首步均通过。 N607 preflight、归档 SHA、远端编译、PID/CWD/GPU 和日志增长已核实；独立 P0/P1 审查通过，其他 owner 进程保留。

## 结果与覆盖

固定源 V 两景预测已独立 truth-last 评分；正式测试尚未进行。用户已授权两数据集源 U 冻结教师指导。共同 SharedTwoPriorGainR17 从各自 R17 输出精确起步，用逐频实增益组合实增益方向及物理回投残差；损失为 clean 教师与恢复分布 KL、0.2 归一化 IQ MSE、1.5 clean R17 保持。LoRa 使用原 W24 精确 scipy 重采样前向，反向使用 WideTracker 可微前端替代梯度；WiSig 使用原可微识别器路径。每侧 1000 步，batch 4/24，教师样本前向 8000/48000，另计反向与祖先成本。 源 V 面板已多轮曝光，训练不读 U TX、V、目标 IQ 或真值；冻结后独立评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | R17 锚教师 | 同预算无教师 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 | 59 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 31 | 15 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 178 | 162 |

LoRa DURA 列是源训练 R69 架构适配，WiSig 列为原固定 DURA v1。LoRa 教师版相对实增益改对 3、改错 6，净 -3 包，配对 bootstrap 95% 区间 -9 至 +3；相对 R69 适配净 -2。WiSig 相对 R17 改对 7、改错 161，净 -154，区间 -174 至 -135；相对 DURA v1 净 -135。WiSig 源 TX 类别 0 为 14/68，对应 R17 为 62/68，存在明显单类退化。两侧均未通过所有对照的预登记晋级门槛。

配对无教师版：LoRa 教师版改对 19、改错 3，净 +16/60，95% 区间 +8 至 +24，clean 净 +1；WiSig 改对 51、改错 35，净 +16/360，区间 -3 至 +34，clean 净 0。WiSig 类别 0 比匹配无教师版少 16/68，类别 1 少 9/54，逐类见本地 `paired.json`。训练 seed/架构/预算及源生成规则匹配，不同 GPU 浮点轨迹可能不同；源 V 已反复曝光。

同包独立尺度诊断 `20261128-diagnostic-shared-r17-scale-s2026112953-r01` 对比原 R17(y) 与 R17(y/RMS)*RMS。在 LoRa 60 包和 WiSig 360 包的 clean、低城区两景中，两条路径预测均逐包一致；该事实不证明波形逐值一致或其他数据集条件一致。训练后输出的明显退化保留为本次结果。

修正网络各 11,394 参数。教师 LoRa/WiSig 分别源 U 曝光 4,000/24,000、教师样本前向 8,000/48,000，训练 558.22/373.88 秒，PyTorch 峰值约 176.1/78.3 MiB；无教师对照 503.20/274.75 秒、约 130.6/55.3 MiB。教师、专家及其历史成本另计；端到端时延、正式七景和最终验收待完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
