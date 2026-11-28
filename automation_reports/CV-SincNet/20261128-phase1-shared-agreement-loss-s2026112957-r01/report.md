# LoRa 与 WiSig 共用教师一致性门控锚点损失筛查

- run_id：`20261128-phase1-shared-agreement-loss-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

仅在源模拟包的 clean 和锚点教师预测类别一致时保持锚点 KL，避免不一致时抑制 clean 教师纠错。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 36fe3ee02a6fdc279709ee9c118aebedbcd294ac；N607 控制器 4036548，LoRa 4036625/GPU 6，WiSig 4036626/GPU 4。两侧前 16 包真实专家和教师梯度 smoke 通过，PID/CWD/GPU 与日志增长核实；预检、单包 SHA、远端编译和独立 P0/P1 审查通过。

## 结果与覆盖

两数据集共用 5% 实增益与 DURA 物理残差网络，分别训练权重。仅在源 U 模拟包的 clean 与锚点冻结教师预测类别一致时施加锚点 KL，clean 教师 KL、波形损失和 clean 锚点保持继续；门控仅训练时使用，聚合一致次数将记录，推理无需识别器。LoRa 实增益/WiSig R17 锚点由已曝光源 V 选择；每侧源 U 4096 包、clean 512、低城区 1792、其他五景各 358/359，源训练专用模拟流。未读取 U 真 TX、V 或目标参与训练；教师前向预定 12008/72032（含 smoke），源模拟及专家/教师祖先成本另计，两侧固定预测已完成，随后独立 truth-last 评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 一致性锚点损失 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 35 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 332 |

LoRa DURA 对照为 LoRa 源训练 R69 结构适配，WiSig 为原固定 DURA v1。LoRa 相对实增益改对 1、改错 0，净 +1，95% 配对区间 0 至 +3、半区 +1/0；WiSig 相对 R17 改对 3、改错 3，净 0，区间 -5 至 +5、半区 0/0。clean 均保持；两侧均未通过预登记分半收益门槛。

相对无一致性门控、同源模拟和预算的教师版，LoRa clean 和低城区改对 0、改错 0；WiSig clean 改对 0、改错 0，低城区改对 1、改错 1，净 0，区间 -3 至 +3。逐 TX 与全部对照见 `paired.json` 和 `readback/*/screen-score.json`；源 V 面板反复曝光。

源 U 教师一致计数 LoRa 3118/4000、WiSig 22187/24000；额外教师样本前向含 smoke 分别 12008/72032。每侧增益网络 11394 参数。含模拟池的训练耗时约 727.15/444.15 秒，PyTorch 峰值显存约 175.5/79.4 MiB；专家、教师祖先成本与实测推理时延另计。正式七景和最终独立评分待后续候选通过筛查后进行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
