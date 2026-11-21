# 1% 标注 Tri 的 E20 校验范围

复用前一轮冻结 E20 替换机制与同契约种子 01。只用合法源 V 的错分复查决定本批两个扩展阈值。在 E50 已过 0.95 且与 E20 类别不同的集合中，E20 置信至少 .75 时，500 条里 E20 正确 472、E50 正确 24。门槛 .60 时，615 条里 E20 正确 563、E50 正确 39。两档的源校验准确率分别为 94.4%、91.54%。这些诊断只决定本批候选门槛，V 不进入训练损失或预测。[完整源 V 对照](E:/type10-7/local_artifacts/pamf_tri_label01_viewroute_20261121/source_e20_disagreement.json)

两行分别固定 E20 门槛下界 .70 或 .55。按原公式加上 `0.15*n/(n+14)`。1% 原合同每组有 7 个标注，计算门槛分别为 .75 和 .60。两行均只替换已通过学生原门槛 .95、类别与早期锚点不一致的源 U。保持原 Tri 六视图、接受集合计算前的学生门槛、U=6 和梯度保护。原始 .85、.95 复用历史同实验配方的结果用于解释，所有行的目标测试设计已曝光。[生效矩阵](experiment.json) · [原 E20 校验机制](../20261121-phase1-pamf-tri-label01-earlyanchor-clean-m3-r01/report.md)

在 N607 普通账号固定运行 GPU 0、4、5、6，每卡最多 2 个总计算进程且保留 12 GiB 准入。入口为 `python -m experiments.pamf_repl_source_adapt.early_anchor_reach publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG`。两条真实同契约 E20 源冒烟通过后，自动续训固定 E50、预测 clean 三子集、独立 truth-last 评分及冻结后盲 U 正确率诊断。阻断性技术问题停止新派发，保留既有任务和产物。主指标为 `cross_rx + cross_both`，同时比较原 Tri、普通 PL 和联合变化子集。超过 PL 且 `cross_both` 至少不低于原 Tri 才达到独立种子复核门槛。
