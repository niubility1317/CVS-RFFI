# LoRa 与 WiSig 共用大容量逐频实增益与 DURA 残差筛查

- run_id：`20261128-phase1-shared-wide-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

128 通道扩张卷积共同预测逐频实增益，组合 DURA 物理残差，使用 5% 修正上限及冻结教师和锚点保持损失。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 f3047d963cd70989b9a17bef13ba143db6a6529e；N607 控制器 3934603，LoRa 3934615/GPU 2，WiSig 3934616/GPU 3。两侧真实专家和教师首步、有限非零梯度、PID/CWD/GPU 已核实。预检、归档 SHA、远端编译及独立 P0/P1 审查通过。

## 结果与覆盖

两数据集共用 128 通道四层扩张卷积逐频实增益网络、DURA 物理残差、5% 修正上限与损失，分别训练权重；LoRa 实增益、WiSig R17 锚点由已曝光源 V 选择。训练使用用户授权的各数据集源 U 冻结教师，LoRa W24 E104 原 scipy 前向和替代反向，WiSig Tri-SIG-TIME SSL F0，预定教师样本前向 12000/72000 次；推理无教师。原专家和教师祖先成本另计，不读 U 真实 TX、V 或目标训练；固定两景预测已独立 truth-last 评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 大容量增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 31 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 331 |

LoRa DURA 对照是 LoRa 源训练 R69 结构适配，WiSig 使用原固定 DURA v1。候选相对 LoRa 实增益改对 0、改错 3，净 -3，95% 配对 bootstrap 区间 -7 至 0，半区 0/-3；相对 WiSig R17 改对 1、改错 2，净 -1，区间 -5 至 +2，半区 0/-1。clean 两侧净 0；均未通过门槛，未进入正式测试。

相对同 seed 5% 小网络对照：LoRa 改对 0、改错 2，净 -2（区间 -5 至 0）；WiSig 改对 3、改错 3，净 0（区间 -5 至 +5）。逐 TX 与其他对照见 `paired.json`、`readback/*/screen-score.json`；配对文件 `mse_correct` 为旧字段名，实际 control 是小网络。不同 GPU 浮点轨迹可能不同；源 V 已反复用于设计。

每侧共享网络 467,842 参数；LoRa/WiSig 源 U 曝光 4000/24000 次、教师样本前向 12000/72000 次，训练 488.51/477.63 秒，PyTorch 峰值约 469.3/927.5 MiB。专家和教师祖先成本、端到端时延另计；完整目标七景和独立最终评分待条件满足后执行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
