# PAMF 干净／信道配对 U：源域后续假设

状态：仅固定假设，未实现、未登记启动、未发布、未训练、未读取本版目标。当前 VIEWANCHOR 三个健康任务继续原配方。本假设在首轮 PSM/RXPSM 目标结果已经公开、任何 VIEWANCHOR 目标访问之前记录；不声称全新独立盲测。下列系数由现有源端视图和损失质量分解固定，不从目标分数搜索或选择父模型。

当前 RXPSM／VIEWANCHOR 对每批 256 个 U，只让 student 看到 128 个普通噪声视图和 128 个物理信道视图，teacher 看到全部原 U。源端日志确认 U 损失、0.10 物理特征项、其他 RX 标签原型和连续计数均实际工作。它没有完整的 256 条原 U student 分类分支；这个结构事实不是 clean 损伤的因果证明，也不证明新增分支必然有效。

固定复用同一首轮 RXPSM 各 seed 的 E30/6660：2026092701/02/03 的学生、自身 EMA、Gaussian 统计及仅由可见 L 更新的原型。追加 20 轮/4440，累计 E50/11100、13L＋37LU；不串接 VIEWANCHOR 的 E50，也不测试 E30 来挑父模型。匹配的 VIEWANCHOR E30→E50 配方已独立固定。所有数据、监督 T1／默认联合头、teacher0.996、置信权重、原型及优化日程保持原定义。

E31 起，对于每个真实 U IQ 同时给 student 原 IQ 和一个同记录的 Practical 物理视图，按相同顺序拼成 512 条，domain 标签按两视图复制；teacher 只计算一次原 IQ。保留同一 soft target、连续权重 w 及跨 RX 标签原型，不接触 U TX 真值。

固定新增阶段的无标签目标为：

`0.50 * mean_clean(w * CE_soft) + 0.25 * mean_physical(w * CE_soft) + 0.0125 * mean_clean(w * prototype_distance) + 0.0125 * mean_physical(w * prototype_distance) + 0.10 * mean_physical(w * (1 - cosine(z_student, stopgrad(z_teacher_weak))))`。

物理 CE 的平均系数 0.25、两个原型分支总系数 0.025 和物理特征系数 0.10 均来自现有 VIEWANCHOR 的分解；将普通噪声 U 子集替换为完整原 U student 分支，并把 clean CE 系数固定为 0.50。这个整体变体同时改变了视图数量和 clean 权重，不把它描述成单一因素因果消融。PAMF 有标签监督分支完全保留，U 分支不新增普通噪声，不增加模型／推理参数或另训头。

每步仍读取 256 个 U、更新一次自身 EMA 和一次 L-only 原型，但 student U 视图由 256 增为 512，不能宣称相同 FLOPs 或相同训练耗时。最终 teacher U 曝光预期仍为 2097900，E31 后新增 student clean／physical 各 1134000 次；累计物理 U 视图为 481950＋1134000＝1615950。最终 EMA 更新 11100、原型可见 L 标签计数 1051392。须在实现和实际运行时逐项核实，当前仅为预算预声明。

只作为未完成原目标时的预备路线。若进入实现／发布，按当前最小流程：严格固定源父来源、聚焦梯度和物理对齐负例、一次 P0/P1 核查和真实父 checkpoint no-query smoke，独立不可覆盖三 seed run。现有 3 卡每卡最多 2 个本任务 worker、保护其他任务和实际容量规则保持；本文件不启动额外任务。

验收仍只看固定 F0 未见 RX 目标集，raw clean＋六 practical、每视图 168000 条。最终 E50 学生全部预测固定后独立 truth-last 评分；同一方法对 F20、F21 的 clean 三 seed 均值分别至少 +1 pp、六景平均分别严格超过 +1.5 pp。报告全部 seed、逐景／RX／TX／day、实际 U 视图及累计成本；未满足时保持目标未完成。
