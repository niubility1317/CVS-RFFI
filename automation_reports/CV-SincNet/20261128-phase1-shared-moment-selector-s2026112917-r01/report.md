# LoRa 与 WiSig 共用增益不变矩选择器筛查

- run_id：`20261128-phase1-shared-moment-selector-s2026112917-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两数据集均未通过源域筛查，未进入正式测试。

## 目的与对照

同一整包门控结合实增益、R17 和物理作用残差；以模拟 clean 配对的相位及增益不变包统计特征误差构造无 TX 标签选择目标。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `b047cfe761595f4ebc432c7faf070c2070ceaa1e`。N607 preflight、单一归档传输 SHA 与远端编译通过；控制器 PID 3718024，LoRa worker PID 3718041 / GPU 7，WiSig worker PID 3718042 / GPU 1。已独立核查 CWD、命令、GPU 显存占用与真实权重首步；WiSig 日志增长至第 50 步。发布前 GPU 7、1 分别约 16.9、15.6 GiB 可用，资源动态分配，其他 owner 进程保留。一次独立 P0/P1 审查发现步数校验不一致，已修复并针对性复核通过；14 项本地测试通过。

## 结果与覆盖

已固定筛查预测并独立连接源 V 真值评分。共享选择器以源 U clean 配对中的归一化包络高阶矩、多个时延复自相关和八段频谱能量误差构造训练目标；不读 TX 或 F，推理只用当前包。LoRa 实增益专家历史继承授权 E104 指导与伪标签，WiSig 专家 F-free。此前源 V 面板已多次曝光，目标数据不用于此轮训练与筛查。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 面板结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 共同候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 59 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 15 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 190 |

LoRa DURA 列是 LoRa 源训练的 R69 架构适配，WiSig 列为原固定 DURA v1。LoRa 相对实增益改对 4、改错 23，净 -19 包，配对 bootstrap 95% 区间 -28 至 -10.975 包；相对 LoRa R69 适配改对 4、改错 22，净 -18。WiSig 相对 R17 改对 9、改错 151，净 -142 包，区间 -163 至 -123；相对 DURA v1 改对 8、改错 131，净 -123。完整分半结果和其余对照见本地 readback 中 `screen-score.json`。

选择器各 11,105 参数。LoRa 训练 1,000 步 / 4,000 次源 U 曝光，319.80 秒，PyTorch 峰值分配约 115.3 MiB；WiSig 1,000 步 / 24,000 次曝光，202.20 秒，约 55.3 MiB。完整系统另含两路专家及其历史成本；端到端时延待测。两侧均固定最后一步，新增门控未使用 F、TX、伪标签、V 或目标。源 V 面板在多轮研发中曝光，正式七景、逐接收机及最终验收尚未完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
