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
