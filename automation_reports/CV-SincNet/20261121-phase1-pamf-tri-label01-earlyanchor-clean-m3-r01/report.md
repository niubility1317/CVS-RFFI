# 1% 标注 Tri 的冻结 E20 伪标签校验

原 Tri 在合法源 V 上把真类 3 高置信误判为类 1 的条数从 E20 的 104 增至 E50 的 1008。同一批 27000 条 V，E50 通过 0.95 掩码、E20 置信度至少 0.85 且两者类别分歧的 387 条中，E20 判对 375 条，E50 判对 11 条。E20 门槛为 0.95 时，对应 201 条、198 条和 3 条。源 L 真标签原型在同一融合表示上无法有效纠正高置信误判。[源 V 逐候选统计](E:/type10-7/local_artifacts/pamf_tri_label01_viewroute_20261121/source_e20_disagreement.json) · [原型对照](E:/type10-7/local_artifacts/pamf_tri_label01_viewroute_20261121/source_proto_calibration.json)

复用匹配的本种子 E20 为固定参照。训练前只遍历一次合法、真值盲的源 U，按原物理索引保存 E20 类别和置信度在进程内。只对当前学生达到 0.95 且与 E20 预测分歧的样本考虑修正。E20 门槛为 `min(0.99, b+0.15*n/(n+14))`，`n` 是源 L 中对应 E20 类别、RX、日期的真标签数量。1% 中 `n=7`，两个门槛分别为 .85 和 .95。三行分别为门槛 .85 时暂缓该样本参加 U 损失，以及门槛 .85、.95 时用 E20 预测替换它的伪标签。替换保留学生原通过集合，U=6、KL 系数、视图、对比和原梯度保护不改。[逐行登记](experiment.json)

固定种子 2026092701，在同契约 E20 续训到 E50/11100。目标 clean 分别生成三个 raw 子集后独立 truth-last；对应训练盲 U 记录独立离线评分。报告修正曝光比例及 `cross_rx + cross_both` 对配对旧 Tri、普通 PL 的净收益，另列联合换域和误标类别。该目标设计历史曝光，先做小矩阵探索。超过 PL 且联合变化至少不低于原 Tri 才达到后续独立种子复核的研究门槛。

命令 `python -m experiments.pamf_repl_source_adapt.early_label_anchor publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG`。使用 N607 普通账号独立发布目录，GPU 0、4、5、6，计算进程总上限 2、至少 12 GiB 准入。真实 E20 无目标冒烟核实源池参照与原训练图后自动启动。系统性技术故障停止新派发并保存产物。
