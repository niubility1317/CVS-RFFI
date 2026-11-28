# LoRa 与 WiSig 源域 practical 配对训练的无新增教师匹配对照

- run_id：`20261128-phase1-shared-practical-mse-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

以相同 4096 个源 U 模拟包、seed 和 1000 步，仅关闭冻结识别器教师损失，检验指导训练的实际影响。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 d0977fba91ceae71494cb2d154f7cc35166e0318；N607 控制器 4019329，LoRa 4019351/GPU 6，WiSig 4019352/GPU 4。两侧前 16 包真实恢复专家与波形梯度 smoke 通过，PID/CWD/GPU 和日志增长已核实；预检、归档 SHA、远端编译与独立 P0/P1 审查通过。

## 结果与覆盖

匹配 5% 低城区加权教师版的共同网络、源 U 4096 包及模拟、seed、锚点、1000 步和非教师损失；本次新增训练只用 0.2 波形误差加 1.5 clean 锚点保持，不加载/调用识别器。LoRa 实增益父专家仍有 E104 指导及伪标签祖先，WiSig R17 为 F-free；两数据集筛查仍使用各自冻结默认 F。源 V 已曝光并参与锚点选择，无 U 真 TX、V 或目标参与训练；固定两景预测已独立 truth-last 评分。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 无新增教师 | 匹配教师版 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 32 | 35 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 330 | 332 |

LoRa DURA 对照为 LoRa 源训练 R69 结构适配，WiSig 为原固定 DURA v1。无新增教师版相对 LoRa 实增益改对 2、改错 4，净 -2，95% 配对区间 -7 至 +3、半区 0/-2；WiSig 相对 R17 改对 1、改错 3，净 -2，区间 -6 至 +2、半区 +1/-3。clean 两侧净 0，均未通过预登记晋级门槛。

匹配教师版相对本对照：LoRa 改对 4、改错 1，净 +3，95% 区间 -1 至 +7、半区 +1/+2；WiSig 改对 5、改错 3，净 +2，区间 -3 至 +8、半区 -2/+4。两侧 clean 净 0；这表明小样本中的差异方向和不确定性，教师版相对强锚点依然未晋级。`paired.json` 使用本对照作为第一个输入，历史字段 `teacher_correct` 实际指本对照、`mse_correct` 指教师版；上文已反向换算。逐 TX 和全部对照见 `paired.json`、`readback/*/screen-score.json`；不同 GPU 数值轨迹可能不同，源 V 已多轮曝光。

每侧网络 11,394 参数；源 U 4096 包中 clean 512、低城区 1792、其余五景各 358/359。LoRa/WiSig 训练曝光 4000/24000、F 前向均为 0；含模拟池生成耗时 566.76/343.56 秒，PyTorch 峰值约 122.3/55.3 MiB。LoRa 实增益父专家有 E104 教师与伪标签祖先，WiSig R17 为 F-free；专家和教师祖先成本及实际推理时延另计。正式七景和独立最终评分待条件满足后执行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
