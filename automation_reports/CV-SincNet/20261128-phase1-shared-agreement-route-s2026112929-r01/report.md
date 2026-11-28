# LoRa 与 WiSig 共用冻结识别器输入一致性路由筛查

- run_id：`20261128-phase1-shared-agreement-route-s2026112929-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两数据集均未通过源域筛查。

## 目的与对照

同一逐包规则：当原输入默认预测只与一路专家相同时选择该专家，否则选择默认头 softmax 最大值较高的专家；不训练或拟合阈值。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `a389fc4b261e2747d7d4048ce8bb784886b3a0c2`。N607 preflight、单一归档传输 SHA 与远端编译通过；控制器 PID 3752556、LoRa PID 3752573 / GPU 7、WiSig PID 3752574 / GPU 1。独立核查 CWD、命令、显存、真实专家和识别器首步，预测后 CPU 独立评分。一次独立 P0/P1 审查通过，15 项本地测试通过。

## 结果与覆盖

两侧预测固定后独立连接源 V 真值。此规则逐包用原输入和两路专家的冻结默认头预测：只有一路与原输入预测一致时选择该路，其他情况按最大 softmax 置信度选路、平分选 R17；再对实际选中的信号运行原默认头。无新训练/阈值或跨包统计，属于识别器指导推理；评测使用同一识别器。筛查每包含四次原默认头前向，端到端时延待测。LoRa 专家继承授权 E104 指导和伪标签，WiSig 实增益专家 F-free；源 V 多轮曝光。

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 共同候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 28 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 297 |

LoRa DURA 列为其源训练 R69 架构适配，WiSig 列为原固定 DURA v1。LoRa 相对实增益改对 2、改错 8，净 -6 包，配对 bootstrap 95% 区间 -12 至 0 包；相对 R69 适配净 -5。WiSig 相对 R17 改对 2、改错 37，净 -35 包，区间 -47 至 -24.975；相对 DURA v1 改对 22、改错 38，净 -16。逐对照分半结果见各 `screen-score.json`。无新增训练参数；两专家及识别器历史成本均计入整体。正式七景与最终验收均待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
