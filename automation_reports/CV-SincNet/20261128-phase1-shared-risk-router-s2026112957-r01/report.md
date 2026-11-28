# LoRa 与 WiSig 共用源伪标签保守专家路由筛查

- run_id：`20261128-phase1-shared-risk-router-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

共享物理特征网络依据当前包选择实增益或 R17，源域 clean 教师伪标签构造保守切换目标。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 09dc5582d58535793e9312aa1a7f7190c41cb154；N607 控制器 3896268，LoRa 3896335/GPU 7，WiSig 3896336/GPU 1。真实专家首步、有限非零路由梯度、PID/CWD/GPU 与日志增长均已核实；预检、单包 SHA、远端编译及一次独立 P0/P1 审查通过。

## 结果与覆盖

共同物理特征路由网络和损失，LoRa 实增益及 WiSig R17 为分别独立训练时的固定默认锚点。源 U clean 教师 argmax 为伪标签，只有其他专家正确且锚点错误时目标为切换；锚点正确的负例权重 2，两者均错忽略，并以 clean BCE 防止切换。使用 LoRa W24 E104 和 WiSig Tri-SIG-TIME SSL F0 冻结原默认头，推理只使用当前包和恢复专家。预定 LoRa/WiSig 教师样本前向 12000/72000 次，专家和教师祖先成本另计。源 V 多轮曝光并用于锚点及方案选择；不使用 U 的真实 TX、V 或目标训练，固定两景预测已独立 truth-last 评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 保守路由 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 34 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 332 |

LoRa DURA 对照是 LoRa 源训练 R69 结构适配，WiSig 使用原固定 DURA v1。候选相对各自强锚点 LoRa 实增益和 WiSig R17 均改对 0、改错 0，低城区两固定半区净变化均为 0，配对区间均为 [0,0]，clean 净变化均为 0；因此均未通过预登记晋级门槛。正确包集合相同不表示所有错误类别预测相同。

相对同 seed 2% 锚点保持版：LoRa 改对 1、改错 1，净 0（95% 区间 -3 至 +3）；WiSig 改对 1、改错 2，净 -1（区间 -4 至 +2）。clean 均净 0。逐 TX 与其他对照见 `paired.json`、`readback/*/screen-score.json`；配对文件的 `mse_correct` 为旧字段名，实际 control 是 2% 锚点保持版。不同 GPU 浮点轨迹可能不同；源 V 已反复曝光并用于设计。

每侧路由网络 11,105 参数；LoRa/WiSig 曝光源 U 4000/24000 次、教师样本前向 12000/72000 次，有效伪标签 2765/16786 次，其中纠错正例 351/3531 次；训练 435.20/249.37 秒，PyTorch 峰值约 117.9/55.3 MiB。两恢复专家及识别器祖先成本、端到端推理时延另计。目标完整七景及最终独立评分待满足条件后执行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
