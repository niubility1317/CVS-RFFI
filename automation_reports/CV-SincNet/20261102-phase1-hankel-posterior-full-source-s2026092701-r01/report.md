# Hankel 物理后验：完整源域固定参数复核

沿用既定 pilot 的单条 IQ Hankel 收缩，窗口 32、噪声尺度 0.1、混合比例 0.5、相同物理先验和门，绝不根据 pilot 结果改变参数；不训练或更新模型。固定已存在的 11,761 条源物理记录，并逐条复现此前 672 条 pilot 决策及对齐 IQ 误差；先固定全部新预测，随后才连接合法 L_s TX 标签进行源评分，再以独立排序物理 ID 连接和集合计数复核。源 F 的伪类别先验须披露；不读取 U_s 身份真值、V 或地面测试。源记录此前被探索，非盲测或地面成绩。

CPU 两线程、CUDA 禁用，服务器负载超过 72 时等待。唯一源 run `20261102-phase1-hankel-posterior-full-source-s2026092701-r01`，预检和评分分别使用对应 `full-smoke`、`full-score` 根；代码/config 为本报告所在 Git commit。入口 `python -X utf8 -u -m experiments.phase1_hankel_posterior.full_controller --commit <本报告所在 commit>`。先做 64 条真实源预检，完整输出固定后进行 16 分组、736 字段独立复核；输出冲突、协议错误或非有限值停止并保留，低分不停止。

比较的物理后验和 DURA 来源于既定源预测，候选为 CPU 重放；强制复核 raw、门与物理分类逐条匹配，但这些匹配不能保证 DURA CPU/GPU 数值完全一致。此项为未训练源域诊断，不替代正式地面七景。
