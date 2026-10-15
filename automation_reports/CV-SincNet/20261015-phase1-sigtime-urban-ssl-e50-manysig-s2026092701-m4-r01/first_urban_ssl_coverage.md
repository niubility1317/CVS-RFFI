# E50 首轮额外 urban SSL 的实际覆盖

状态：第 34 轮已完成；整体 E50 训练尚在运行。这里的 U_s 接受率是额外软监督资格比例，U_s 真值仍隐藏，不能计算其伪标签精度，也不等于目标准确率。

| 行 | 原生 U_s 数 | urban U 前向数 | 接受软监督数 | 接受率 (%) | 额外弱教师前向数 | native 轮耗时 (s) |
|---|---:|---:|---:|---:|---:|---:|
| SSL_CONF-s2026092701 | 56700 | 56700 | 53115 | 93.6772 | 0 | 331.03 |
| SSL_WEAK-s2026092701 | 56700 | 56700 | 52891 | 93.2822 | 56700 | 331.30 |

每行有标签 satellite 累计 966144 条＝34×222×128，全部使用真实源 TX 标签参与 CE。新增 U urban IQ 全部沿原 CVS 和默认头前向，资格不消费 urban 学生预测，也不叠加原 batch-neighbor 门控；原生旧门控半监督损失保持。训练只生成 mid/low urban U 视图，硬件 session 由 RX/day/eq 定义，训练随机流按 epoch/step/view 刷新。
两行实际 `phase=pseudo`、`train_pseudo_truth_available=0`。自有 EMA 首步 V 校验的额外前向数分别为 27256、54256，见 teacher_launch_health.md。此表 native 轮耗时不单独代表完整 source 墙钟，最终资源表将使用 source/completion.json 的总耗时并分别列出所有前向成本。
实际训练代码仍为 `7a366c506a736c34e5bba335821f4867f2d3706b`。所有 source-only 资格、阈值、测试数组和预算均保持原发布值，没有根据目标分数调整。
