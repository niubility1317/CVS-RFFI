# 单视图 REPL：双 L/U 信道增强，关闭低置信同包一致性

当前状态：真实同契约 E20 联合 backward smoke 已通过，唯一 NoPair 源学生在 GPU0 实际训练，压缩逐步训练日志持续增长。当前没有源 E50 或目标性能结论；以下保留初次登记及执行记录。

本试验补齐一项现有源消融：保留 H2LU 的全部 L/U 双信道视图、全 L 真标签 CE/As KL0、原 ordinary 浅中/joint 跨 RX、高置信 raw95 PL、weight6/cap1/style0，只关闭低置信同包 Ha/Hb 互概率一致性。不增加 L-anchor 项、网络或前向；被拒绝 U 的计数继续记录，但该项 loss/weight 为 0，不能称其已得到低置信优化。

依据来自此前完整源结果：U 双 H 的 raw/H3 正确数为 26549/62255，增加低置信 pair 后是 26547/62199。源 V 的拒绝子集中存在两 H 同错、甚至三 H 同错情况。这不足以证明 pair 有害，却说明不能先假定该项有独立收益；原矩阵缺少“L 也双 H、但无 pair”的单因素。此处补齐它，正在运行的 L-anchor 学生完全保持原配置。

仅 seed2026092701，从自己的同契约普通 PL 单 SIG_TIME scratchE20 分叉 +10+20 到 E50/11100（13L37LU）；不继承 H2LU E50 或 L-anchor 学生。完整原 PAMF 和默认 6×160 联合头保持。训练只读 clean/直接 practical residual/post_sync/noEQ IQ，U TX 隐藏、RX/day 可见。V27000×4 冻结评价，已有 H2LU 参照不重训、不重复预测；无 G/恢复 IQ、目标或新诊断头。

固定源门槛 raw≥26568/27000、H3≥62109/81000，与 L-anchor 源试验相同。两个独立源试验都完成后，在过门槛者中按 raw 正确数、H3 正确数依次排序；完全同分选无 pair 以减少训练损失计算。没有通过者则保留负结果，不扩种子或目标。只允许另行登记选中方法的三 F seed 正式确认，源通过不是目标成功。源选择计划在两个新源结果产生前固定，见 [source_selection_plan.json](source_selection_plan.json)。

此前主目标结果已公开，本试验不宣称新的独立盲测；不根据目标弱类、RX、分数调整损失、权重、门槛、候选顺序或选择性重跑。原目标 raw clean≥85.98174603174603%、六景>72.0961970899471%，且较匹配监督/普通PL clean≥+1pp、六景>+1.5pp 不下调。当前是未启动的新源小试验，不能写成已经有效。

本地 23 项 focused 检查及 compile PASS，其中新增回归验证禁用项不执行 peer 计算且原 U 梯度保持；独立 P0/P1 审查尚待完成。真实 E20 联合 backward smoke 是 launcher 第一步，PASS 后立即训练；源 E50 后自动冻结并评价四景。GPU0–7 按全球 CUDA+pending<2/free≥10000MiB 调度，计入其他 owner、保护健康任务；无空位仅 CPU 控制器排队。独占输出、唯一 owner、系统失败停止新派发且不自动重跑。

代码、环境/命令/CWD、祖先与数据句柄、精确矩阵和产物见 [experiment.json](experiment.json)。当前目标 ACTIVE。


本地验证收尾：23 项 focused 和 compile PASS；本候选唯一独立 P0/P1 审查 PASS（0 P0、0 P1）。真实 E20 smoke 及新学生尚未启动。见 [验证范围](local_verification.json)。


## 实际发布、检查通过与训练启动

VERIFIED_RUNNING。执行 commit `99550ef10a6cfcffa5e6fd2cfd204f45c2e11d47`，发布前本地/上游 OID 一致。第一次入口 CWD 误指 carrier，读取本地 SSH 配置即退出；独立核实远端 run/release/archive 均不存在后，改从项目根目录复用同一冻结 archive 继续，单 archive 传输比较 VERIFIED、远端 compile PASS。没有更改模型、预算或现有进程，不能把最初提交失败写成成功启动。

真实 E20 联合 backward smoke PASS：L768、U256×6，L/U 两 H 实际不同；H2 state 的 dual_L=true、pair_enabled=false、pair_weight=0，无低置信 peer 或 L-anchor 单项梯度。原完整 U 目标对 SIG_TIME/head/other 的梯度有限且非零。检查没有 optimizer 更新、U 真值、目标或恢复器读取。

唯一学生已在 GPU0 实际启动。控制器/worker 的 OS 身份、CWD/完整命令及显卡归属独立核实，控制器不占 GPU。先核算当时全部完整可读 24 条训练记录，L768/U6、双 H、peer loss/weight=0、cap1/weight6/style0/无非有限跳步均符合；后一次观察完整记录增长至 93 条。第一轮尚未结束，stdout 字节暂未变化，增长证据是压缩逐步训练日志，不能把 24 条审计扩大称为 93 条或全 E50 审计。

正在运行的 GPU7 L-anchor 学生保持原配置。两学生都完成 E50 和四景源评估后，继续按先前固定 source raw/H3 门槛与排序决定是否进入另行登记的同一方法三种子确认；目标未达成。证据：[实际检查、进程和日志增长](launch_evidence.json)。
