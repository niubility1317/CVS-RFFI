# LoRa 与 WiSig 共同恢复的源域 practical 模拟配对筛查

- run_id：`20261128-phase1-shared-practical-pairs-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

两数据集各自源 U 抽 1024 包，利用独立源训练随机流生成 clean 和六景 practical 模拟配对，训练同一实增益与 DURA 物理残差恢复规则。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 49efe714e29b1513202add888c116dee5cd7949d；N607 控制器 3953233，LoRa 3953439/GPU 2，WiSig 3953440/GPU 3。源模拟前 16 包上的真实专家、教师及有限非零梯度 smoke 均通过；PID/CWD/GPU 和日志增长核实。预检、单包 SHA、远端编译和独立 P0/P1 审查通过。

## 结果与覆盖

每侧从合法源 U 固定抽 1024 个物理包，128 clean，896 包近似等量覆盖六种已登记 practical 模拟配置。使用源训练专用 namespace 和独立随机种子，未读取目标信号、目标统计、V 或源 U 真实 TX。训练 clean 和受扰信号先用模拟器定义归一化，再共同除以当前受扰包 RMS；两侧共用 5% 逐频网络、损失和训练预算，LoRa 实增益及 WiSig R17 锚点由已曝光源 V 选定。冻结教师仅训练时使用，模拟 CPU 成本及专家／教师祖先成本另计；固定两景预测已独立 truth-last 评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | practical 配对训练 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 34 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 334 |

LoRa DURA 对照为 LoRa 源训练 R69 结构适配，WiSig 为原固定 DURA v1。LoRa 相对实增益改对 0、改错 0，净 0，固定半区 0/0、配对区间 [0,0]；WiSig 相对 R17 改对 5、改错 3，净 +2，95% 配对 bootstrap 区间 -4 至 +8、半区 +3/-1。两侧 clean 净变化均为 0。两个数据集都未达到预登记每半区门槛，未进行正式测试。

相对同 seed 5% 通用随机信道训练对照：LoRa 改对 2、改错 1，净 +1（区间 -2 至 +4）；WiSig 改对 6、改错 3，净 +3（区间 -3 至 +9）；clean 均净 0。逐 TX 和其他对照见 `paired.json`、`readback/*/screen-score.json`。通用 paired scorer 的原始 `paired-original.json` 在 caveat 中误写“两运行训练生成器相同”；这里刻意改变了源训练生成器和样本，修正后的 `paired.json` 只更正该元数据，配对计算未变。不同 GPU 数值轨迹亦可能不同；源 V 多轮曝光并用于设计。

各侧增益网络 11,394 参数。每数据集训练源 U 模拟池 1024 包，128 clean、其余六景各 149 或 150；LoRa/WiSig 训练曝光 4000/24000 次，教师样本前向 12008/72032 次（含 smoke），包含模拟池生成的训练耗时 532.30/431.14 秒，PyTorch 峰值约 175.5/79.4 MiB。教师与专家祖先成本、实际端到端时延另计。目标完整七景、逐 RX/TX/day 及独立最终评分待条件满足后执行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
