# LoRa 与 WiSig 教师连续融合的同 seed 无教师对照

- run_id：`20261128-phase1-shared-matched-mixture-s2026112941-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

与教师连续融合版同结构、seed、源 U 配对和五点权重网格，用 clean IQ MSE 而非教师 KL 选训练目标。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `f91fec9517be623f19cf02aceda3d137bfd406ae`。N607 preflight、单一归档传输 SHA、远端编译通过；控制器 PID 3794781、LoRa PID 3794860 / GPU 0、WiSig PID 3794861 / GPU 5。独立核查 CWD、命令、显存、真实专家首步及有限非零梯度，日志增长；教师融合版使用 GPU 3/6，其他 owner 进程保留。一次独立 P0/P1 审查通过，16 项本地测试通过。

## 结果与覆盖

固定源 V 两景预测已独立 truth-last 评分；正式测试尚未进行。与教师融合版相同 seed 2026112941、结构、专家、源 U 抽样、模拟配对、优化器、五点融合权重网格和各自 batch，固定 1,000 步；本对照以 clean IQ MSE 选网格目标，不读取教师 F、U TX、V 或目标训练。不同 GPU 浮点轨迹可能不同。LoRa 实增益专家历史继承 E104 指导和伪标签；WiSig 实增益专家 F-free。源 V 面板多轮曝光。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 教师融合 | 匹配无教师对照 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 | 59 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 25 | 15 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 220 | 199 |

LoRa DURA 列为其源训练 R69 架构适配，WiSig 列为原固定 DURA v1。LoRa 教师融合相对实增益改对 4、改错 13，净 -9 包，配对 bootstrap 95% 区间 -17 至 -2；相对 R69 适配净 -8。WiSig 相对 R17 改对 10、改错 122，净 -112，区间 -133 至 -92；相对 DURA v1 净 -93。WiSig 源 TX 类别 3 仅 10/55，对应 R17 为 46/55，存在明显单类退化，不能用总均值掩盖。两侧均未达晋级门槛。

预登记匹配对照的独立同包分析：LoRa 教师融合相对无教师版改对 12、改错 2，净 +10/60，95% 区间 +3 至 +17，clean 净 +1；WiSig 改对 28、改错 7，净 +21/360，区间 +10 至约 +32，clean 净 0。WiSig 源 TX 类别 0 和 3 各较对照少 1 包，逐类详见 `paired.json`。两组源生成 seed、结构和训练预算匹配，不同 GPU 的浮点轨迹可能不同；源 V 已多轮曝光，此结果不构成未曝光目标测试。

门控各 11,105 参数。教师融合 LoRa/WiSig 分别曝光源 U 4,000/24,000 条，教师样本前向 48,000/288,000，训练 699.44/385.43 秒，PyTorch 峰值约 117.4/55.3 MiB；匹配无教师版分别 270.54/225.64 秒、约 115.3/55.3 MiB。专家和识别器及其历史训练成本另计，端到端推理时延、正式七景与最终验收待完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
