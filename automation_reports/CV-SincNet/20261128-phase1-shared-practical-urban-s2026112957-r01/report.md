# LoRa 与 WiSig 共同恢复的低城区加权源域模拟训练筛查

- run_id：`20261128-phase1-shared-practical-urban-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

按用户优先低城区要求调整源 U 模拟场景比例，保留 clean 和其他五景；共同模型、固定筛查不变。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 8b7e1aeb2a96ac7a2db85f5013527df381317a80；N607 控制器 3981450，LoRa 3981463/GPU 2，WiSig 3981464/GPU 3。两侧前 16 包真实权重及教师梯度 smoke 通过，PID/CWD/GPU 和日志增长已核实；预检、归档 SHA、远端编译和独立 P0/P1 审查通过。

## 结果与覆盖

两数据集各自从合法源 U 固定抽 4096 包，其中 clean 512、低城区 1792，其他五景各 358 或 359；使用独立源训练模拟命名空间和随机流。该比例响应用户优先关注低城区的要求，其他场景与 clean 保留。共同 5% 网络、损失及 1000 步，分别训练权重；LoRa 实增益和 WiSig R17 锚点来自已曝光源 V。未读取 U 真正 TX、V 或目标进行训练；WiSig 均衡 4096 包版为 333/360，半区 +1/0 未晋级，LoRa 当时尚未读取。预定教师样本前向 12008/72032，含 smoke；源模拟、专家和教师祖先成本另计，固定两景预测已独立 truth-last 评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 低城区加权训练 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 35 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 332 |

LoRa DURA 对照为 LoRa 源训练 R69 结构适配，WiSig 为原固定 DURA v1。LoRa 相对实增益改对 1、改错 0，净 +1，95% 配对区间 0 至 +3，两个固定半区 +1/0；WiSig 相对 R17 改对 2、改错 2，净 0，区间 -4 至 +4，半区 -1/+1。clean 两侧净 0；均未达到预登记每半区门槛，未进入正式测试。

相对同 seed 的 4096 包均衡训练版，LoRa 改对 1、改错 0，净 +1（区间 0 至 +3）；WiSig 改对 1、改错 2，净 -1（区间 -5 至 +2）；clean 均净 0。逐 TX 及全部对照见 `paired.json`、`readback/*/screen-score.json`。通用 scorer 原始 `paired-original.json` 的同生成器 caveat 未说明场景分配变化；`paired.json` 仅补正元数据，配对计算未改。两版使用相同源包抽样，但部分包模拟场景不同；不同 GPU 数值轨迹可能不同，源 V 多轮曝光。

各侧网络 11,394 参数；每数据集源 U 模拟池 4096 包，clean 512、低城区 1792、其余五景各 358 或 359。LoRa/WiSig 训练曝光 4000/24000 次、教师样本前向 12008/72032（含 smoke），包括模拟池生成的耗时 705.97/474.57 秒，PyTorch 峰值约 175.5/79.4 MiB；源模拟内存及专家/教师祖先成本另计。正式七景和独立最终评分待条件满足后执行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
