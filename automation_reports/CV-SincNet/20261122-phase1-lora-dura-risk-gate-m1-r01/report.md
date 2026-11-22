# LoRa DURA 只改旁路门控的配对试验

前一轮固定相干度门控把源 V 城区低可恢复覆盖由旧逐 chirp DURA 的 382/800 降为 178/800。投影网络残差 MSE 由 0.3691 降至约 0.296，但没有增加目标正确数。为检验范围限制，保留匹配同源的旧逐 chirp DURA 权重、原相干度剂量和已冻结的 adaptive_tri，只对恢复准入做对照。

源侧门控只读 IQ。11 维可观察特征含去啁啾相干度、幅度变化、跨 chirp 差异和局部残差，二层小 MLP 区分 1600 条原始 L 与相同物理包的 9600 条 practical 观测，固定种子 721 和 45 轮 BCE。校准阈值取全部源 L clean 与 2560 条盲 U clean 的最高分加 1e-4。无任何 TX、F、源 V 或目标用于门控训练和阈值拟定。源 V 只在门控冻结后审计。真实 IQ 本地核对额外 U 校准后，V clean 0/800 触发，城区低有 406/800 可恢复候选。分数是源域覆盖证据。

| 读出 | 干预 |
|---|---|
| direct | 冻结 adaptive_tri 直接识别 |
| old_chirp | 旧恢复器和旧准入 |
| risk_filter | 只有旧门控和学习门控都允许的包才走旧恢复剂量 |
| risk_extend | 原来可恢复且被学习门控允许的包保持旧剂量，新增包以置信差为剂量，最高门控强度 0.30，所有恢复沿用旧骨架和旧 RMS 剂量上限 |

四路共用 1920 个历史开发物理目标包、15 台接收机、clean 加六种 practical，全部 28 个读出都须完成。独立 truth-last 报告逐景净纠错和 clean 逐点透传，目标不回流决策。小试若有收益，再在完整主划分上合法冷启动恢复器，与冻结三种子 Tri 同行比较。

执行 `python -m experiments.pamf_lora_main.refine_publish --dura-gate`，在 N607 普通 CVS-RFFI 环境。工作目录 `CV-SincNet/releases/20261122-phase1-lora-dura-risk-gate-m1-r01-release`，输出 `CV-SincNet/runs/20261122-phase1-lora-dura-risk-gate-m1-r01`，GPU0–6 按现有容量每任务单卡、排除 GPU7。依赖冻结的同源 old_chirp G 和已冻结 adaptive_tri F，训练配置不含目标路径；非有限、父权重不合或评分缺行按既有策略停止新派发并保留产物。评分命令用 `packet_score --dura-gate` 与相同原档案。预期为 `G/gate/training.json`、`source_audit.json`、盲预测和独立分数。
