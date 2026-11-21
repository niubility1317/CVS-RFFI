# 1% 标注 E20 替换的独立种子复核

探索种子 01 的 .60 和 .75 E20 门槛均超过配对普通 PL，且 `cross_both` 均超过原 Tri。固定源 V 核对上净纠错更高的 .60 门槛，新增种子 2026092702、2026092703。两者分别生成自己的合法同契约 E20 源 U 锚点。目标 clean 设计历史曝光，两个模型种子未用于候选设计。[原候选预登记](../20261121-phase1-pamf-tri-label01-earlyanchor-reach-clean-m2-r01/report.md) · [本批逐行配置](experiment.json)

数据适用条件为源 RX、日期、类别单元内标注数的中位数小于 14。原 11 个设置只有 1% 标注触发（中位数 7）。十类中位数 14，5% 标注为 35，其他设置为 70。门槛保持每样本的 `min(.99,.55+.15*n/(n+14))`，学生原伪标签门槛 .95。1% 本合同所有局部支持均为 7，实际 E20 门槛 .60。未触发的设置沿用原逐设置配方，单源仍使用历史 Hybrid。复核的科学方法、E50 终点、U=6 和其他原 Tri 损失保持。

主指标为固定 E50 对普通 PL 的 `cross_rx + cross_both`，与原 Tri 逐物理样本配对。每个种子另报告两个测试子集、盲 U 诊断和修正次数。三种子汇总含探索种子 01，并单列两个复核种子。训练只访问原合法 L、盲 U 及日常源验证；冻结 E50 后才独立读目标及诊断真值。

N607 普通账号，GPU 0、4、5、6，计算总进程每卡至多 2 且准入预留 12 GiB。入口为 `python -m experiments.pamf_repl_source_adapt.early_anchor_seedcheck publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG`。各自真实 E20 源冒烟通过后启动续训，之后 raw clean 自动预测和独立评分。阻断性的技术问题停止新派发并保留现有产物。
