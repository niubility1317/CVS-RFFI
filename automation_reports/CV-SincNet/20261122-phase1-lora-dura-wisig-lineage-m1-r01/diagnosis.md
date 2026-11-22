# LoRa DURA 与 WiSig 的训练差异和恢复缺口

## 正式成绩与验收线

| 数据和配对 | Clean | Practical 直通均值 | Practical 恢复均值 | 恢复净增益 | 收回直通缺口 | 恢复后缺口 |
|---|---:|---:|---:|---:|---:|---:|
| WiSig 原 DURA v1.0，固定 F50 | 84.6780% | 37.3600% | 78.4277% | +41.0677 个百分点 | 86.79% | 6.2503 个百分点 |
| LoRa 已完成最佳高相干半量，固定 Tri-set，三种子 | 71.6556% | 55.1180% | 55.8517% | +0.7337 个百分点 | 4.44% | 15.8039 个百分点 |
| LoRa 物理轨迹原 R69 半量，固定 Tri-set，种子 02 | 70.3633% | 55.3956% | 55.5017% | +0.1061 个百分点 | 0.71% | 14.8617 个百分点 |

收回比例按「恢复六景均值减直通六景均值」除以「同一冻结识别器直通 clean 减直通六景均值」计算。WiSig 和 LoRa 的直通缺口分别是 47.3180 和 16.5376 个百分点。以 LoRa 的三种子 clean 为参照，差距至多 3 个百分点要求六景至少 68.6556%。收回与 WiSig 接近的 86.79% 缺口对应约 69.4711%。正式判断同时核对绝对 clean，并报告六景、种子和未见接收机的各自结果。

[WiSig DURA v1.0 定义](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/docs/DURA_V1.md) · [LoRa 三种子评分](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_detail_confirm_readout/scores.json) · [LoRa 轨迹版评分](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_r69_trace_readout/scores.json)

| Practical 场景 | LoRa 直通 | LoRa 三种子高相干半量 | 净增益 | WiSig 原 DURA |
|---|---:|---:|---:|---:|
| 郊区高 | 63.6900% | 63.8156% | +0.1256 个百分点 | 82.8250% |
| 郊区中 | 59.5478% | 60.4300% | +0.8822 个百分点 | 81.5607% |
| 郊区低 | 55.5322% | 56.8344% | +1.3022 个百分点 | 80.0583% |
| 城区高 | 61.8444% | 62.0100% | +0.1656 个百分点 | 81.7458% |
| 城区中 | 49.4267% | 50.2467% | +0.8200 个百分点 | 75.4863% |
| 城区低 | 40.6667% | 41.7733% | +1.1067 个百分点 | 68.8899% |

LoRa 高相干半量三种子合计纠正 16,011 次，同时改错 12,049 次，净增 3,962 次。45 个种子/RX 组合中有 42 个取得净增益，3 个下降。改善出现在多数接收机，发送机之间差别更大。TX 31 净增 2,849 次，TX 35、37 分别净少 595、323 次。这支持检查指纹保真和错误转移。TX 编号来自冻结后的单独评分，仅用于归因。

## 原训练链与前几轮的实际差异

| 阶段 | WiSig 同契约保留流程 | LoRa 已测原 R69 适配的差异和影响 |
|---|---|---|
| 粗恢复 | 单独训练时频 IQ 恢复器，得到后续共同初值 | 主要依赖缩短的教师训练或冷启，粗估计质量随联合校准变化。固定源 V 实验中，单独使用旧粗恢复在 5,335 次高相干曝光上净少判对 529 次。 |
| 接收作用和方差 | 模拟器给出 64 抽头复数 FIR、主系数、镜像系数和实噪声方差。冻结粗恢复，作用估计器用探针一致性和对数方差学习 | 早期由 IQ 恢复误差间接学习，之后才补可重放轨迹。轨迹教师第 900 步独立校准结束时，方差项为 0.03391。此后第 1,100 步为 4.77995，结尾为 3.49789。日志来自不同批次，只支持校准漂移的怀疑。同一批固定轨迹上的跨阶段验证仍缺。 |
| R17 | 固定粗恢复和作用估计，只训三阶段展开先验，使先验从一致的观测残差中学习 | 前几轮没有完整训练这一代，R65 直接承担校准、恢复与蒸馏教师角色。 |
| R38、R44、R46、R65 | 依次从同契约冻结的上一级蒸馏，R65 到第 10 轮后按原规则剪枝 | 上游多级教师缺席。LoRa 旧 R65 和 R69 各 2,400 步，输出轨迹的起点与 WiSig 差别很大。 |
| R69 | 两阶段学生继承冻结 R65 的后两阶段。用配对 IQ、频谱损失和逐阶段教师轨迹完成 20 轮 | 旧版保留架构，但多数正式读出只取 R69 部分残差、限幅、半量或按门控跳过。部署输出与原版整段 IQ 参数相同的比较曾缺失。 |

对照的保留实现是每阶段 20 轮、每轮 247 次更新、批量 256，共每阶段 4,940 步和约 126.5 万窗次。八级合计 39,520 次更新。WiSig 和 LoRa 可以共享训练程序和架构，源 IQ、接收机身份和 checkpoint 继承必须各自保持匹配。LoRa 新版仍以 LoRa 合法 L/U 波形从头训练，R69 的正常输出是全 IQ，两阶段、四次 CG、266,951 参数。

[原八阶段训练逻辑](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/experiments/cric_dura_comparison/g_train.py) · [逐阶段冻结及损失](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/experiments/cric_dura_comparison/g_lineage.py) · [物理轨迹适配日志与读出分析](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-r69-physical-trace-m2-r01/analysis.md)

## 观测差异和实际诊断

### 同样 256 点所包含的信息不同

WiSig 的 256 点对应 25 MS/s 下的 10.24 微秒。LoRa 以 1 MS/s 对原包上下文施加 practical 完整链路，再去实现延迟、切片并重采样到 256 kS/s。每个 256 点窗口约 1 毫秒，每包使用六个 chirp。WiSig 的历史测试用 residual、post-sync、noEQ，LoRa 既有正式测试用 full、post-sync、noEQ。保留 LoRa 的载频、时间尺度与 full practical 测试，可检验恢复器对现有任务的迁移。

WiSig 历史 residual 观测带有 +24 点实现延迟。固定源 V 上，仅把该位移归零就让识别率从 48.01% 到 73.69%，同一源诊断中的完整 DURA 为 93.60%。它量化了此源诊断对对齐的敏感性。正式 WiSig 测试成绩由多种变化共同造成，这个对齐实验未覆盖正式目标集。

在已对齐的 LoRa 中，检查三版冻结模型各 18,000 个源 practical 窗口，主峰落在第 24 抽头的数量都为 0。该抽头仍占 5.11% 至 6.29% 的平均核能量，第 0 抽头占 11.64% 至 16.24%。因此新训练按 LoRa 的零延迟坐标监督。核能量质心也单独记录，它并非估计出的整体到达时延。

[WiSig 对齐反事实](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_fingerprint_audit/wisig_shift_control.json) · [LoRa 实际作用核](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_fingerprint_audit/lora_native_dura_kernel_delay_source.json)

### 新增噪声和确定性形变都在影响识别

固定 600 个源 V 包，对其 3,600 次 practical 曝光逐条重放完全相同的模拟器轨迹。原始 practical 为 58.0278%，在 AGC 前精确减去模拟器新增噪声后为 67.8611%，净增 9.8333 个百分点，同一面板的 clean 为 78.1667%。城区低强度从 41.8333% 到 63.5000%，这一景模拟器新增噪声的信号噪声比中位数约 7.89 dB。完美减噪后仍有 10.3056 个百分点的总体缺口，确定性接收变化及其与识别器的相互影响仍存在。这个反事实使用模拟器的真实噪声，仅用于源域定位。它既不给训练提供目标信息，也无法直接换算正式目标成绩。

[同包噪声反事实](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_fingerprint_audit/r69_channel_noise_oracle_source.json)

另一项源 V 分解在 15,374 次满足固定高相干规则的 practical 曝光上替换已知配对差分。只替换六个 chirp 的共用波形差分净多判对 672 次，只替换各 chirp 的独立差分净多 193 次，同时替换两者的非载波分量净多 1,892 次，完整 clean 反事实净多 2,122 次。两部分会交互，单独纠正其中一项的收益远小于同时纠正。此面板采用固定源域样本和实际配对 clean，反映目标恢复方向的可用信息，数值不能外推为真实接收时可直接实现的收益。

[高相干配对波形反事实](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_fingerprint_audit/coherent_shape_oracle.json)

### 覆盖受限，扩大覆盖还要求恢复方向更可靠

轨迹版在种子 02 的 180,000 次目标 practical 曝光上只处理 28,720 次（15.96%），相对直通净多判对 191 次。旧门控旁路了 30,000 个 clean，旁路本身解释了 clean 保持。在另一个固定源 V 实验中，1,730 次门控内观测使用限幅全量净多判对 10 次，直接使用未限幅完整波形净少判对 54 次。在另外 5,335 次高相干曝光上，轨迹半量、完整波形和仅粗恢复分别净多判对 89、24、-529 次。现有模型的主要问题包含输出方向和粗恢复，盲目开门会放大已有破坏。需要用完整训练链重新验证输出，再决定怎样扩大处理范围。

Labeled/Unlabeled 中的 TX 真值不进入 DURA。Practical 训练与对照都以同一物理包匹配，星地落差重点检查新增噪声、主/镜像作用、幅相变动和发射机指纹细节。全流程同时记录源 V 逐景配对波形误差和冻结后逐接收机识别。波形误差是恢复诊断，正式判断以同一冻结 F 的配对正确数和准确率为准。

[源域读出消融](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_fingerprint_audit/r69_trace_readout_source.json) · [门控覆盖诊断](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_fingerprint_audit/r69_transferred_gate_source.json)

## 单种子正式验证与后续判断

先用 2026092702 的冻结 Tri-set。每个读出用 15 台未见接收机、每台 2,000 个物理包、clean 加六个 practical。已有八个同配对读出从冻结预测直接复用。追加两个读出，`lineage_native` 对所有输入运行 R69 原版完整 IQ 输出，`lineage_guarded` 按旧门控复用同一完整输出，其余使用冻结直通。两条都接受 clean。预测全部固定后独立评分；最终阈值按三种子复验，首轮只能验证训练链和单种子效果。

| 首轮正式现象 | 探查重点 |
|---|---|
| 全量能恢复，旧门控收益明显小 | 核对旁路中净纠错和 clean 损伤，使用源域盲验证检查能否扩大门控覆盖。 |
| 全量损伤 clean，门控内依旧接近直通 | 沿 coarse、action、R17、R65、R69 检查何时开始丢判别细节，优先检查冻结作用模型对 LoRa full 的适用性。 |
| 源 V 的配对波形误差持续下降，正式正确数收益仍小 | 用同包固定 F 定位幅相、RFF 频段与净纠错。下一轮保持恢复器无 TX 标签和 F-free 训练。 |

这轮保持原训练损失与优化器、阶段顺序及蒸馏方式。为适配 LoRa 对齐输入，使用 28,000 个源训练包和每包三组对齐 residual 代理生成作用监督。粗恢复混合该代理、原 LoRa full 配对及 clean；R17 至 R69 使用原 LoRa full 配对及 clean。从代理删去初始八点实现延迟及负时间 FIR 尾，记录删除的能量。该代理提供 LoRa 坐标下精确可计算的训练算子，LoRa full 真实模拟路径仍可能有算子无法表达的分量，后续按上述同配对结果判断。

正式续发使用 `20261122-phase1-lora-dura-wisig-lineage-m1-r02`。首发源 smoke 的输入类型错误已修复。续发冒烟使用六景真实 LoRa 源 IQ、真实同契约旧 R69 权重，验证有限作用梯度和输出。正式读出使用原版两阶段 R69 数学路径的 PyTorch 调用。本轮未重复 WiSig CUDA Graph 的时延实验。

84,000 组源作用代理已生成。因果对齐丢弃的 FIR 能量中位数为 0.0028%，第 95 百分位为 0.2116%，最大值为 41.35%。多数代理只需要移除极小的一部分，少量困难条件可能有很大的前游分量。后续应特别核对这类分量是否在 LoRa full practical 中制约 R17 的软一致性更新。

[正式续发登记](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r02/report.md)

已经完成的联合读出也支持先固定识别器分析。以种子 02 为例，新的 Practical REPL 识别器配轨迹 R69 后六景 56.0122%，它自己的 clean 68.6300%，低于原 Tri-set 的 70.3633%。恢复网络相对各自直通只新增约 0.1250 个百分点；相对同种子原 Tri-set 配高相干半量的 56.0417% 仍少 0.0294 个百分点。后续 DURA 评测保持识别器与既有高相干半量对照匹配。

[联合读出正式结果](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/adapted_joint_readout/scores.json)
