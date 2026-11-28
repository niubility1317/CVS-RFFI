# LoRa 与 WiSig 共用冻结识别器置信度整包路由筛查

- run_id：`20261128-phase1-shared-confidence-route-s2026112925-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；原评分技术失败后，复用两侧固定预测完成数值恢复和独立评分。两数据集均未通过筛查。

## 目的与对照

同一逐包规则比较冻结默认头对 R17 与实增益恢复结果的最大 softmax 概率，选更高者并使用同一默认头预测；无新训练及阈值拟合。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `3a998f6dcb09c235956178ad8b60526c346b10f3`。N607 preflight、单一归档传输 SHA 与远端编译通过；控制器 PID 3740877，LoRa worker PID 3740887 / GPU 7，WiSig worker PID 3740888 / GPU 1。独立核查 CWD、命令、GPU 显存占用，真实专家和识别器无 query 检查后 WiSig clean 预测日志已增长。发布前 GPU 7、1 分别约 16.8、15.5 GiB 可用，其他 owner 进程保留。一次独立 P0/P1 审查未发现直接缺陷；12 项本地测试通过。

## 结果与覆盖

源域独立评分已完成，正式测试尚未进行。共同逐包规则对两个专家输出分别运行原冻结默认头，选择最大 softmax 概率较高者，平分选择 R17；再使用同一默认头形成预测。不训练、不拟合阈值、不读真值或跨包统计。此为识别器指导推理，评价使用同一识别器；需按该依赖解释结果。筛查实际执行每包六次默认头前向，包含两次路由探测和四路候选/对照预测，端到端时延待测。LoRa 实增益历史继承授权 E104 指导与伪标签，WiSig 实增益专家 F-free。源 V 已多轮曝光。

N607 上 PyTorch 的 `.numpy()` 与 NumPy 版本接口不兼容，导致两侧已冻结的类别预测保存为 object 数组，原 scorer 的 `allow_pickle=False` 正确拒绝读取；两侧预测阶段均已完成，评分阶段在连接真值前失败。原文件与失败日志保留。修复提交 `5bc6b851b1107131be67d8387afecb3dd53da783` 使用 `.tolist()` 后显式生成 `int16`；技术恢复脚本仅处理本 run 自产、已冻结的预测文件，逐值验证整数范围与转换一致性，写入独立恢复目录后再进行独立评分，不重新预测。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 面板结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 共同候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 26 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 333 |

LoRa DURA 列为 LoRa 源训练的 R69 架构适配，WiSig 列为原固定 DURA v1。LoRa 候选相对实增益改对 4、改错 12，净 -8 包，配对 bootstrap 95% 区间 -16 至 -1 包；相对 LoRa R69 适配净 -7 包。WiSig 相对 R17 改对 2、改错 1，净 +1 包，区间 -2 至 +5 包，两个固定半区分别 +1、0，未达每半区至少 +2 的预登记门槛；相对 DURA v1 改对 28、改错 8，净 +20 包，区间 +9 至 +32 包。各对照和分半结果见恢复目录 `screen-score.json`。

数值恢复产物独立位于 `runs/20261128-phase1-shared-confidence-route-s2026112925-r01-numeric-score-recovery-r01`。脚本先验证两侧已冻结结果及逐值整数一致，再使用安全数值数组重新读取并调用原 CPU truth-last scorer；未重算预测，原失败 run 保留。该候选无新增可训练参数；两位专家和冻结识别器及其历史成本均需计入完整系统。筛查为每包六次头前向，独立部署时延待测；该结果依赖参与路由的同一识别器。正式七景、逐接收机与目标验收均待完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
