# Affine 训练标量视图 r02

完整保留 20564 条曲线和 3240 个阶段；JSONL/CSV 仅含实际标量，原始数组及归档引用保留在 training_diagnostics。

本版修正了 r01 元数据中不存在的 accepted_ce_prox_conflict_count 被默认写为 0 的问题；该旧统计无效。r02 逐阶段读取必需字段 CE_vs_proximal_conflict_count，统计的是梯度事件的 CE 与 proximal 冲突，不是接受步骤后的 CE 上升。后者未派生，记 N/A。r01 原产物保留。

训练阶段没有 epoch；源域验证禁止访问，均记 N/A。inner-held 标签参与训练，训练准确率不能作为独立泛化结果。缓存时间可能重复，不求和曲线显示时间。数值状态或文件字节不等于星载内存或传输。
