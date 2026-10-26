# 单视图 REPL：双 L/U 信道增强，关闭低置信同包一致性

本试验补齐一项现有源消融：保留 H2LU 的全部 L/U 双信道视图、全 L 真标签 CE/As KL0、原 ordinary 浅中/joint 跨 RX、高置信 raw95 PL、weight6/cap1/style0，只关闭低置信同包 Ha/Hb 互概率一致性。不增加 L-anchor 项、网络或前向；被拒绝 U 的计数继续记录，但该项 loss/weight 为 0，不能称其已得到低置信优化。

依据来自此前完整源结果：U 双 H 的 raw/H3 正确数为 26549/62255，增加低置信 pair 后是 26547/62199。源 V 的拒绝子集中存在两 H 同错、甚至三 H 同错情况。这不足以证明 pair 有害，却说明不能先假定该项有独立收益；原矩阵缺少“L 也双 H、但无 pair”的单因素。此处补齐它，正在运行的 L-anchor 学生完全保持原配置。

仅 seed2026092701，从自己的同契约普通 PL 单 SIG_TIME scratchE20 分叉 +10+20 到 E50/11100（13L37LU）；不继承 H2LU E50 或 L-anchor 学生。完整原 PAMF 和默认 6×160 联合头保持。训练只读 clean/直接 practical residual/post_sync/noEQ IQ，U TX 隐藏、RX/day 可见。V27000×4 冻结评价，已有 H2LU 参照不重训、不重复预测；无 G/恢复 IQ、目标或新诊断头。

固定源门槛 raw≥26568/27000、H3≥62109/81000，与 L-anchor 源试验相同。两个独立源试验都完成后，在过门槛者中按 raw 正确数、H3 正确数依次排序；完全同分选无 pair 以减少训练损失计算。没有通过者则保留负结果，不扩种子或目标。只允许另行登记选中方法的三 F seed 正式确认，源通过不是目标成功。源选择计划在两个新源结果产生前固定，见 [source_selection_plan.json](source_selection_plan.json)。

此前主目标结果已公开，本试验不宣称新的独立盲测；不根据目标弱类、RX、分数调整损失、权重、门槛、候选顺序或选择性重跑。原目标 raw clean≥85.98174603174603%、六景>72.0961970899471%，且较匹配监督/普通PL clean≥+1pp、六景>+1.5pp 不下调。当前是未启动的新源小试验，不能写成已经有效。

本地 23 项 focused 检查及 compile PASS，其中新增回归验证禁用项不执行 peer 计算且原 U 梯度保持；独立 P0/P1 审查尚待完成。真实 E20 联合 backward smoke 是 launcher 第一步，PASS 后立即训练；源 E50 后自动冻结并评价四景。GPU0–7 按全球 CUDA+pending<2/free≥10000MiB 调度，计入其他 owner、保护健康任务；无空位仅 CPU 控制器排队。独占输出、唯一 owner、系统失败停止新派发且不自动重跑。

代码、环境/命令/CWD、祖先与数据句柄、精确矩阵和产物见 [experiment.json](experiment.json)。当前目标 ACTIVE。


本地验证收尾：23 项 focused 和 compile PASS；本候选唯一独立 P0/P1 审查 PASS（0 P0、0 P1）。真实 E20 smoke 及新学生尚未启动。见 [验证范围](local_verification.json)。
