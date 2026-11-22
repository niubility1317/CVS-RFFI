# LoRa DURA full 信道作用监督对照

运行名 `20261122-phase1-lora-dura-full-action-observed-phase-m1-r01`。对照为 [观察相位八阶段版本](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-observed-phase-m1-r01/report.md)。冻结阶段诊断确认观察相位解决原 R17 最后阶段的严重缩幅，弱城区动作误差和源身份损失仍有提高空间。40 个源 V 包的物理反事实中，用从 full 观测轨迹拟合的已知作用替换学习作用，源判对从 114/240 到 145/240。该结果用于决定预先登记的训练变量，目标不参与此选择。

## 训练与源权限

合法源 L 2,800 包、U 25,200 包，两份已登记的 1 MHz full practical 观测。对每份原带噪观测逐点回放，四组固定带限探针拟合 no-EQ 接收的逐时刻复主系数，以实际新增噪声构建方差。主监督系数采用单位 FIR、镜像系数为 0 的稳定拟合。在学习侧完整保留原 64 抽头 FIR、主/镜像估计器与噪声方差参数化。生成过程只读取 L/U 的 IQ 与原接收机、物理 ID 和配对角色。源库不读取 TX、F、V 或任何目标文件。

唯一改变动作阶段非 clean 样本的监督来自同合同 full practical 主观测或独立观测。两份观测在训练轮间固定交替，使用相同的物理行和 chirp 选择。继续使用 15% 恒等 clean、观察相位参考变换和原探针损失。后续 R17 至 R69 的损失、优化器、教师继承、R65 剪枝和 R69 输出保持观察相位对照的方式。

粗恢复沿用同一源合同、种子 2026092702 和观察相位的冻结 E20/4,940 步，原权重、冻结元数据与日志精确复制。后七阶段各 20×247 步，共 **34,580 次新增更新**，最终链包含 39,520 次累计训练步骤。所有新权重从该链上一级直接继承。重新训练从作用阶段开始，无目标反馈。

## 正式矩阵与发布

- 同一冻结 Tri-set 2026092702、原 15 台未见目标接收机每台 2,000 个已登记物理包，clean 与六景 practical。原版全量输出及同输出加旧门控，与八条既有冻结对照一起读出。分别比较 direct、现行高相干半量，以及原坐标和观察相位两条线的同种子正式结果。
- N607 普通账户、已验证 CUDA 环境和独立 release/run 目录。CPU 源库最多 12 个 worker，按 GPU 6、2、1、4、5、7、0、3 的顺序选满足既有可用显存与进程容量的一个 GPU。另两条健康运行保留。
- 技术停止规则为输入合同、确切复放、冻结粗恢复身份或输出校验不符、真实源 smoke 失败、训练出现非有限梯度、正式矩阵缺项。保留已有产物。科学结果不改变运行内阶段和预登记测试。
- 发布入口 `python -m experiments.pamf_lora_main.full_dura_lineage_publish ... --reuse-actions ORIGINAL_SUBMIT --full-action-submit OBSERVED_PHASE_SUBMIT`。产物包括源库元数据、源 smoke、八阶段冻结权重与日志、源 V 诊断和 1,050 份盲预测。七景预测冻结后依旧由本地 `full_dura_score` 做独立评分。最后完成三种子时使用既定汇总方式。

[源动作监督试制](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_lineage_r02_source_panel/full_action_targets_LU_pilot_noise.json) · [源软求解归因](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r02/source-stage-analysis.md)

发布 **VERIFIED**。代码提交 `4c65520e63424882c1d160ceeedfd5145c9ff267` 的归档传输 SHA 比对及远端编译通过。六景实测源输入和父粗恢复冒烟通过，作用梯度范数为 8.2724。独立复核真实 L/U 两种观测的逐值复放、训练/试制探针一致性和源角色索引。已核实控制器与 CPU 建库进程的工作目录、命令和日志增长。已见主 L 观测完成 2,000/2,800 包。候选在构建 56,000 份源观测的动作监督，随后自动调度训练。冻结源读出与正式七景评分的等待进程已经启动。

[发布提交记录](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_full_action_observed_phase_release/submit.json)

三种子补齐入口已准备，待该候选的固定证据决定复验。剩余种子为 2026092701 和 2026092703。各自从头训练 8 段、39,520 步，复用同源 IQ 和两种作用监督缓存，使用冻结的同种子 Tri-set。盲路由沿用同一个冻结的源 IQ 门控。每种子读出六条已冻结同配对对照和两条新输出，共 56 个评分单元。正式发布入口在各自独立 run 使用 `--replica-seed SEED --reuse-actions ORIGINAL_SUBMIT --reuse-full-bank FULL_ACTION_SUBMIT --confirm-submit COHERENT_CONFIRM_SUBMIT`。发布映射和完整评分解析已经用两个种子的既有控制预测与合成评分标签验证。此处仅准备代码，复验未发布。

## 全量缓存与冻结作用阶段

N607 已完成 L 2,800 包、U 25,200 包的两份已登记 full practical，合计 56,000 次源观测精确回放。源库确认每包 6 条 chirp、4 个固定拟合探针，TX/F/V/目标读取均为 0。GPU 6 进程继承相位粗恢复后确实完成了 20×247 次作用更新，随后进入 R17。

对照均用同一批源 V 物理包。算子测试为 20 个包、每景独立留出探针的 NMSE，参考使用观察相位坐标。识别为另一固定 120 包面板和同一冻结 Tri-set。

| 场景 | 直通正确数 | 原代理监督正确数 | Full 监督正确数 | 留出动作 NMSE 原代理 → Full |
|---|---:|---:|---:|---:|
| Clean | 94 | 68 | 65 | — |
| 郊区高 | 85 | 65 | 68 | 0.01013 → 0.00702 |
| 郊区中 | 77 | 65 | 71 | 0.00545 → 0.00534 |
| 郊区低 | 56 | 59 | 51 | 0.01058 → 0.00931 |
| 城区高 | 75 | 65 | 64 | 0.00673 → 0.00475 |
| 城区中 | 58 | 54 | 50 | 0.05758 → 0.05236 |
| 城区低 | 44 | 45 | 47 | 0.05023 → 0.04785 |
| 六景总计 | 395 | 353 | 351 | — |

Full 监督在所有六景降低了未经额外标量校正的 full 动作误差。固定 soft solve 和同一冻结 F 时，六景比代理监督少 2 次、城区低多 2 次、clean 少 3 次。比直通多判对 58 次，同时把 102 次原正确改错。算子监督只能解释目前一部分性能差距，剩余主要看展开阶段对精细波形的保真。两版作用冻结后均继续完整训练，正式性能由各自七景冻结评分判断。

[Full 界面物理探针](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_full_action_observed_phase_source_panel/full_action_probe_gauge.json) · [冻结作用阶段读出](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_full_action_observed_phase_source_panel/action_frozen_tri_set.json)

## 冻结 R17 逐步读出

用同一冻结 Tri-set、同一 120 个源 V 包比较 R17。两条线的 R17 前缀与各自冻结的作用阶段状态逐点一致。固定粗恢复先验和三次各两步 CG 的对照读出与各自作用阶段一致。表中 practical 统计为六景 720 次曝光。

| 读出 | 观察相位原代理 | 观察相位 Full 动作监督 |
|---|---:|---:|
| 直通 | 395 | 395 |
| 固定粗恢复先验 | 353 | 351 |
| R17 第一步 | 354 | 339 |
| R17 第二步 | 354 | 345 |
| R17 第三步 | 341 | 355 |
| 完整 R17 的 clean 正确数 | 68/120 | 65/120 |
| 完整 R17 的城区低强度正确数 | 42/120 | 44/120 |

Full 作用监督对照在最后一步增加 10 次六景正确判定，其 R17 完整输出比代理监督多 14 次。原代理则在第三步少 13 次。两个版本完整 R17 都低于直通，且 Full 监督 clean 略低。Full R17 的最后一步城区低输出能量为参考的 96.7%，相位代理为 96.5%。完整模型已进入 R38，以同一测试流程验证效果能否延续。

[Full R17 逐步结果](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_full_action_observed_phase_source_panel/r17_prior_readout.json) · [配对观察相位 R17](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_observed_phase_source_panel/r17_prior_readout.json)
