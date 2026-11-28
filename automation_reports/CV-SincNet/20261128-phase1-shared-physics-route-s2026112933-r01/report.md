# LoRa 与 WiSig 共用 DURA 前向残差路由筛查

- run_id：`20261128-phase1-shared-physics-route-s2026112933-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两数据集均未通过源域筛查。

## 目的与对照

同一逐包规则利用 R17 估计的信道作用比较两位专家输出的归一化前向残差，选择残差较小的完整信号；不训练或拟合阈值。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `60d50160070dc38bb417620d170a95ed7d431943`。N607 preflight、单一归档传输 SHA 与远端编译通过；控制器 PID 3758468、LoRa PID 3758490 / GPU 7、WiSig PID 3758491 / GPU 1。独立核查 CWD、命令、显存、真实专家首步，预测后 CPU 独立评分。一次独立 P0/P1 审查通过，12 项本地测试通过。

## 结果与覆盖

两侧预测固定后独立连接源 V 真值。共享规则按当前包 RMS 归一化观测和两路专家输出，用 R17 估计的同一作用计算两个前向残差，选择残差较小的完整信号，平分选 R17。路由不读识别器输出；原冻结默认头只用于预测。不训练或拟合阈值，不读真值或跨包统计。LoRa 实增益历史继承授权 E104 指导和伪标签，WiSig 实增益专家 F-free；源 V 面板多轮曝光。

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 共同候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 59 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 18 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 332 |

LoRa DURA 列为 LoRa 源训练 R69 架构适配，WiSig 列为原固定 DURA v1。LoRa 相对实增益改对 4、改错 20，净 -16 包，配对 bootstrap 95% 区间 -25 至 -8 包；相对 R69 适配净 -15。WiSig 相对 R17 改对 0、改错 0，净 0，两个固定半区均为 0，未达每半区至少 +2 的预登记门槛；相对 DURA v1 改对 28、改错 9，净 +19，区间 +8 至 +31。其余逐对照、分半结果见各 `screen-score.json`。新增训练参数为 0；两专家和原识别器历史成本需计入整体。WiSig 实现执行两次 R17，LoRa 重估一次 R02/R07，另有两次物理作用前向；实测端到端时延、正式七景与最终验收均待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
