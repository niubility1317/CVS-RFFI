# 同 As 蒸馏三模型种子确认

固定训练代码：`4919fbe67f7fe0f197a0378a0fc39b1c5f1a9470`。候选在读取 P84/P85 正式测试分数前，依据种子 2026092701 的源域门槛通过结果锁定：mid/low 平均 +0.777778 个百分点，full V clean −0.074074 个百分点。

复用已完成的种子 2026092701 T1/KD；新增 2026092702、2026092703 的 T1/KD，共 4 次完整 E30（scratch20＋own student/EMA10），120 个模型 epoch、26640 次更新。每模型 5114880 次 L 前向样本；KD 额外 426240 次冻结 CSIL 前向。原 U/EMA 成本独立记录。相同种子使用相同初始化、源物理样本和固定增强流程。

保持 P85 beta1/common5、默认头联合训练和所有 As 真标签 CE，冻结教师只读取学生同一 As。教师固定历史 s392005/E200；三学生种子结果不能说明教师种子稳健性。六个 row 固定 FINAL30 并自动完成 clean＋六种 practical 预测及独立 truth-last 评分。负结果保留，不据 target 调参、排序、停训或选择重跑。

执行：N607 普通账户；CWD PROJECT/releases/cvs_FcrossRX_sameAsKD_confirm_E30_20261019_r02；命令 CVS-RFFI/python -u experiments/cvs_same_as_kd_confirm86/runner.py；输出 PROJECT/runs/20261019-phase1-FcrossRX-sameAsKD-confirm-e30-m6-r02。每 GPU 含其他 owner 最多 3 个计算进程，空闲显存至少 12000 MB，CPU 线程 2。只因自有系统性协议、数据角色、继承、非有限数或输出技术失败停止并保留产物；不干预健康任务。

48 项种子、路径、预算及非法配置检查和 14 文件语法解析通过。既有 P85 独立 P0/P1 审查覆盖科学机制；种子扩展不重复审查。launcher 首步执行真实源 checkpoint no-query smoke 和新种子梯度/预算 smoke，PASS 后立即训练。

当前状态：PLANNED。三种子确认完成后报告全部 7 视图的均值、标准差、逐 seed 配对差和缺项。

种子修复：r01 在训练前被 smoke 拦下；r02 改用实际新 seed 的原生构建参数。同 seed T1/KD 的 206 个初始张量完全一致，seed02/03 有 93 个张量不同。定点 P1 复核 PASS。此后未改变首次正式读数前已锁定的矩阵、损失或教师。

发布状态：VERIFIED。真实 checkpoint＋梯度 smoke 共 6402 次前向样本通过，4 个新 worker 的目录、命令、GPU 归属和日志增长已核实。实际训练保存的每个同 seed T1/KD 初始权重 206 项完全一致。当前 RUNNING。

历史目标对照复用：原 As／NoAs 的 seed02/03 既有 all7 产物可用，源角色和实际 native seed 一致，头拟合更新为 0；本轮与历史同 seed 初始状态 202/206 张量一致，4 个 sequence_pool 投影权重不同。本轮沿用 seed01 的投影子初始化，改变骨干模型 seed。因此主机制结论只依据本轮 T1/KD 的全 206 张量精确配对；历史目标参照如实披露这一差别，不作为严格相同初始化的因果消融。

源域确认语义：三种子矩阵在首次新正式分数读取前锁定；seed01 正式结果随后已读。三种子 source_decision 中 target_scores_read=false 仅表示该源域汇总函数不读取正式分数，不能解释为全团队从未见过 seed01 分数。新增两个 seed 的正式结果在全源域确认前保持未读；本次固定矩阵完成后作描述性确认，不称为新的独立盲测。

最终状态：ANALYZED / VERIFIED。三学生种子两方案全 6 行 all7 完整；12 个训练 segment 全量日志 PASS。KD−T1 low urban +0.801786pp、mid urban +0.618651pp、clean −0.514087pp。源门槛 PASS，正式联合目标 NOT_MET。完整结论见 final_report.md。
