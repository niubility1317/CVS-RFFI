# LoRa 与 WiSig 共用源域对比表征选择器筛查

- run_id：`20261128-phase1-shared-contrastive-selector-s2026112921-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两数据集均未通过源域筛查，未进入正式测试。

## 目的与对照

同一整包门控结合实增益、R17 和物理作用残差；以源 U 同包模拟双视图自监督表征与 clean 参考误差构造无 TX 标签选择目标。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `bc7920c5e70b63b97bda6e642465e051ca1860f1`。N607 preflight、单一归档传输 SHA 与远端编译通过；控制器 PID 3727327，LoRa worker PID 3727398 / GPU 3，WiSig worker PID 3727399 / GPU 2。独立核查 CWD、命令、GPU 显存占用、真实专家首步，双侧表征与门控日志增长。发布前 GPU 3、2 各约 14 GiB 可用；保留其他 owner 进程。独立 P0/P1 审查通过，13 项本地测试通过。

## 结果与覆盖

已固定筛查预测并独立连接源 V 真值评分。共同表征对合法源 U 同包两次模拟受扰进行 400 步对比训练；不同包可能来自同一 TX，属于无标签负例的限制。表征冻结后比较专家与源 clean 的表征误差，训练相同的整包门控 1,000 步；表征和 clean 只用于训练，部署只使用当前接收包。新训练不读取 F、TX、V 或目标。LoRa 实增益专家历史继承授权 E104 指导和伪标签，WiSig 实增益专家 F-free。源 V 面板已多次曝光。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 面板结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 共同候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 59 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 15 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 190 |

LoRa DURA 列为源训练的 R69 架构适配，WiSig 列为原固定 DURA v1。LoRa 相对实增益改对 4、改错 23，净 -19 包，配对 bootstrap 95% 区间 -28 至 -10.975 包；相对 R69 适配净 -18 包。WiSig 相对 R17 改对 9、改错 151，净 -142 包，区间 -163 至 -123；相对 DURA v1 净 -123 包。逐对照和分半结果保存在各 `screen-score.json`。

训练表征各 31,552 参数，门控各 11,105 参数；推理仅门控与两位冻结专家。LoRa 表征 400 步 / 1,600 次源 U 曝光、门控 1,000 步 / 4,000 次，合计训练 318.98 秒，PyTorch 峰值约 116.7 MiB；WiSig 分别 9,600、24,000 次，230.96 秒、约 55.3 MiB。父专家和识别器历史成本另计，端到端时延待测。源 V 曾多轮曝光；最终正式测试、全部接收机及验收尚未完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
