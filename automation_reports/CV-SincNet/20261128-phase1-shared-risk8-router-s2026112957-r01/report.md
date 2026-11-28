# LoRa 与 WiSig 共用源伪标签纠错加权专家路由筛查

- run_id：`20261128-phase1-shared-risk8-router-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

共享逐包专家路由：将源伪标签切换纠错正例权重固定为 8、锚点正确负例权重为 1。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 8af783b1e1b79a5bbf4408e6fc15013e8f6bbc65；N607 控制器 3904145，LoRa 3904219/GPU 2，WiSig 3904220/GPU 3。真实专家首步、有限非零路由梯度、PID/CWD/GPU 和日志增长已核实；预检、单包 SHA、远端编译及一次独立 P0/P1 审查通过。

## 结果与覆盖

共同物理特征路由网络和损失；源 U clean 冻结教师伪标签仅用于训练，切换纠错正例权重 8、锚点正确负例 1，两者均错忽略，并加 clean 不切换损失。LoRa 实增益、WiSig R17 为源 V 已曝光结果选定的默认锚点。WiSig 首版路由对 360 个低城区样本与 R17 正误完全一致时提出本版，LoRa 首版当时尚在训练。每侧独立权重，推理仅用当前包与恢复专家；预定 LoRa/WiSig 教师样本前向 12000/72000，专家和识别器祖先成本另计。无 U TX、V 或目标训练；固定两景预测已独立 truth-last 评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 纠错加权路由 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 34 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 190 |

LoRa DURA 对照是 LoRa 源训练 R69 结构适配，WiSig 使用原固定 DURA v1。LoRa 相对实增益改对 0、改错 0，净 0，半区均为 0；WiSig 相对 R17 改对 9、改错 151，净 -142，95% 配对 bootstrap 区间 -163 至 -123，两个半区 -70/-72；结果与 WiSig 较弱实增益对照的正确包集合一致。两侧 clean 净变化均 0。两侧均未通过门槛，未进入正式测试。

相对同 seed 保守路由对照，LoRa 净 0；WiSig 改对 9、改错 151，净 -142。完整配对区间与逐 TX 见 `paired.json` 和 `readback/*/screen-score.json`；配对文件的 `mse_correct` 为旧字段名，实际 control 是保守路由。源 V 已反复用于方案选择，非盲测；不同 GPU 浮点轨迹可能不同。

每侧路由网络 11,105 参数；LoRa/WiSig 曝光源 U 4000/24000 次、教师样本前向 12000/72000 次，有效伪标签 2765/16786 次，其中纠错正例 351/3531 次；训练 409.66/248.60 秒，PyTorch 峰值约 117.9/55.3 MiB。两恢复专家及识别器祖先成本、端到端时延另计。目标完整七景和独立最终评分待完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
