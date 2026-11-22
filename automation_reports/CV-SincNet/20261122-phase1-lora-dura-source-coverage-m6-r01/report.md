# LoRa DURA 扩大源配对训练覆盖

上一轮证据显示，WiSig DURA 学习了固定滤波延迟修复和跨信道复原，LoRa 先完成固定对齐，恢复器的原训练只反复使用门控认可的小池。该实验检验是否可以只在训练中扩大配对覆盖，推理仍使用原门控和共享形状保护。

三个种子，每种子两个冷启动 G。原保护投影训练每步 12 个合格 L 配对与 12 个合格 U。新两路每步均保留 6 个合格 L 和 6 个合格 U。

| 新路线 | 其余 12 个训练样本 |
| --- | --- |
| wide | 6 个原门控旁路的 L practical 配对、6 个 U practical 配对 |
| wide_clean | 4 个原门控旁路的 L practical 配对、4 个 U practical 配对、2 个 L 和 2 个 U clean 恒等配对 |

复用已登记的全量 1 MHz 对齐 IQ 缓存。其余 practical 输入从合格池的补集采样，同一禁用剂量预留 U 不进任何 G 优化。旁路组只在训练前向使用 0.60 的恢复强度（输入 RMS 的 15% 上限），部署沿用冻结门槛和原限幅。clean 恒等项直接约束未限幅提案，避免部署旁路使训练梯度为零。两路各 1600 步、batch 24，沿用同一 R69 骨架、初始化方案、配准 IQ 损失、形状投影、AdamW 和调度。先检查扩大当前缓存支持，未加入新的模拟 realizations。训练仅用源 L/U IQ，隔离 F、TX 标签、V 优化与目标。

固定同 seed 的第 168 轮冻结 Tri。读出包含直通、原固定、原固定加推理形状保护、同预算保护投影重训、wide、wide_clean。四个已登记对照的冻结预测逐包原样复用，并核对新增路线每场景实际门控数量与原对照一致。所有方法覆盖相同 30000 包、15 未见 RX、clean 与六景 practical。126 个单元统一独立评分，报告新方法对四个对照的 practical、clean、逐 RX/TX 净纠错。预登记包括全部输出，源诊断只解释过程。此前投影重训的正式目标成绩在该方案固定时尚未读取。

发布 `python -m experiments.pamf_lora_main.full_dura_publish --coverage-submit ...`，N607 普通账户，原 LoRa Python，传入原对齐数据、原 G、保护投影重训及 E168 F 的 release receipt。CWD `CV-SincNet/releases/20261122-phase1-lora-dura-source-coverage-m6-r01-release`，不可覆盖输出 `CV-SincNet/runs/20261122-phase1-lora-dura-source-coverage-m6-r01`。GPU0–6，单卡本 run 最多一进程，本 run 最多四个 worker。真实冻结 G 源 IQ smoke 后启动新训练，原对照预测冻结作为源对照与目标读出的依赖。数据配对、模型祖先、非有限、门控/盲预测数量异常触发技术停止。保留 `source_support.json`、完整计划步训练记录、独立源波形审计、预测证据及 `scores.json`。
