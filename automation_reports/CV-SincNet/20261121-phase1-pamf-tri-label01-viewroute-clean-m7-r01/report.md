# 1% 标注 Tri 自适应分配视图 CE

**VERIFIED / ANALYZED**。发布提交 `b156384f83fa2af83c4da3f4f99cb52357642cf2`。7/7 真实 E20 冒烟与 28 项训练、预测和独立评分完成。最佳 81.6393%，比同种子普通 PL 低 0.9173 个百分点。[全组比较](../20261121-phase1-pamf-tri-label01-earlyanchor-seedcheck-clean-m2-r01/analysis.md)

当前逐设置配方在 10/11 种设置的三种子合并 clean 超过 PL。1% 标注平均差 −0.1040 个百分点，集中在仅跨接收机。三个种子对 PL 的 `cross_rx` 净纠错为 −1649、+1475、−713，联合变化为 −25、+334、+54。跨接收机的损失同时出现在多个接收机和类别。1% 的每个源 RX、日期、TX 单元含 7 条 L，5% 对应 35 条。源域 E50 全局降权未通过，故保留 U=6 与原 0.95 掩码，测试改变同一质量的伪标签监督分布。[历史配对](E:/type10-7/local_artifacts/pamf_tri_label01_viewroute_20261121/original_paired.json) · [源域减权证据](../20261120-phase1-pamf-tri-source-dose-e50-m6-r01/analysis.md)

使用源 L 中当前伪类、接收机、日期的标签数 `n`，根据已有两个 raw 和两个混合视图的伪类概率均值 `p_raw`、`p_mix`，把混合视图 CE 系数 `0.5 × 强度 × 14/(n+14) × clip(4(p_raw-p_mix),0,1)` 转给 raw 视图。系数不参与反传。固定转移对照搬运 0.25。所有迁移来自原混合普通/As 两个半批的 CE，保持系数和、接受率门槛、总 U 权重、六视图前向、KL、外推路径和梯度保护。候选为两档自适应强度 0.5、1 和一档固定转移分别与融合正对均衡开关交叉，外加仅融合均衡，共 7 行。融合均衡分布各半混合，与既有机制相同。[逐行登记](experiment.json)

先用预先固定的种子 2026092701 探索。各行从同源合同自身 E20 到固定 E50，共续训 6660 次，自动生成三个 raw clean 预测并在冻结后独立评分。另有冻结后的盲 U 诊断。主指标与普通 PL、原 Tri 按同物理样本配对的 `cross_rx + cross_both`，兼看最差接收机、联合变化、CE 转移质量和梯度。已有目标测试设计历史曝光。此轮所有候选固定后报告；超过 PL 且联合变化相对原 Tri 不下降是后续独立种子复核的研究门槛。

在 N607 普通账号的独立发布目录运行，GPU 0、4、5、6，已存在的进程计入每卡最多两条，准入 12 GiB。命令为 `python -m experiments.pamf_repl_source_adapt.view_route publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG`。真实 E20 源输入冒烟通过后自动训练、冻结预测、独立评分。阻断性技术错误停止新派发并保留产物。
