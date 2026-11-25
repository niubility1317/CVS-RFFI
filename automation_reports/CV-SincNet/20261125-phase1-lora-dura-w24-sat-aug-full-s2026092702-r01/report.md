# LoRa 星地增强 W24 与重训 DURA 的 Full practical 配对

本轮固定星地增强 W24 识别器。RFFI 识别器种子为 `2026092702`，从 `20261125-phase1-lora-pamf-w24-sat-aug-s2026092702-r01/F/best.pt` 加载 E104。训练提交为 `502629361a6aafa564a94947f3116e82f58c3692`，原 W24 模型架构提交为 `e6dde7d40f2cc3dd7b21d83daa4bcbabac52d3ae`。恢复器使用 `20261125-phase1-lora-dura-w24-train-s2026112991-r01` 的独立 40,000 步终点权重和源 U 门限。测试期间，权重、门限与半量修正系数固定。

同一批 15 台目标接收机、每台 2,000 个物理包，各有 clean 和同一组六景。按原增强 W24 的 `target` 命名空间逐包复现 1 MHz Full practical，并在全部 105 个接收机与场景行上逐包核对直通预测。对照为增强 W24 直通、同次旧版 `complex_response_half_center` 的已冻结预测、重训 W24 DURA。恢复只接入 W24 接收 IQ，保留八条 chirp 的原生边界，只有中心窗接受学习修正。DURA 另有原冻结 Tri-set 及窄带 Residual 的验收工作，此处为新增的 native W24 单种子 Full 测试。

## 执行

从当前工作树用项目 Python 命令 `python -m experiments.pamf_lora_main.dura_w24_augmented_target publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config> --training-receipt <fit-release>/submit.json` 发布，代码身份为包含该报告预登记的提交。N607 项目 Python 与不可覆盖发布根由登记的运行配置绑定。训练终点冻结后，按 GPU 空闲度及至少 8 GiB 空闲显存选择卡。先在 12 个合法源 U 包核对真实权重、W24 前端、识别输出和 clean 旁路，再进行四个盲预测分片，每个使用三个 CPU 进程重放信道。

停止条件为训练未合法冻结、源检查失败、物理/识别器契约不符、历史直通决策不匹配、分片或预测闭合失败。总体结果的高低不触发停止。预期产物为 105 个双行预测文件、四个分片完成记录、源 U 检查及冻结状态。独立评分侧冻结后才打开测试归档，计算逐景逐 RX 正确数、改对、改错、净变化及在 RX 内重抽同包的 95% 区间，跨景共用同一包抽样。评分使用 `python -m tools.lora_dura_w24_score score --output <readout> --dataset <truth-archive-directory> --augmented-reference <frozen-archive-predictions>`。
