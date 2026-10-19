# R07：从源域模拟作用学习观测条件算子

本轮落实用户六类恢复器报告的 R3，不增加设计审批。既有报告已经授权先验证真实作用、再研究观测估计；R06 已完成这一步。通用技能要求的额外设计审批记为 `REJECTED_EXTRA_GATE`，按项目最小流程继续。

当前阶段是单 seed、单候选的源域机制验证，不是最终身份恢复完成。R06 的源证据表明真实作用软修正有潜力，而逐样本静态最小二乘估计没有兑现潜力。本轮不消费 R04/R05 的目标结果来设计或选模。

冻结 R02 源波形 A 选中的粗恢复器 b(r)。新增网络只读取当前 received 和 b(r)，输出长度 64 的单位能量复 FIR、逐时刻复数主/镜像系数和单样本实噪声方差。A(x)=a·(h*x)+b·conj(h*x)，使用原生因果 reflect 边界。六 practical 的原生 kernel 长度为 62；仍在生成时检查，禁止截断。恢复以原粗恢复为先验，执行 6 步实内积共轭梯度软 MAP 更新；不硬投影，不裁剪 IQ。所有 PAMF 输入从一份恢复 IQ 计算。

训练只用原 6300 L / 56700 U 的 IQ，0 TX 标签、0 F 信号。每 epoch 重新生成原生 residual/post_sync/noEQ 观测，15% clean、85% residual；不把此配方冒称 R03 的 15/60/25 mixed。模拟器真实作用仅作源训练监督，以每步两个与 clean 无关的新复高斯探针监督复合算子的作用，辅以 0.1 倍对数噪声方差 MSE。没有配对恢复损失或分类器损失；这是有 clean 源波形的模拟自监督，不是完全无 clean 或真实卫星无先验训练。真实作用/AGC/lock/场景均不输入部署网络。

固定 E20 / 4940 AdamW 更新、batch256、LR2e-4，一 epoch warmup 后 cosine 至2e-5、weight decay1e-4、FP32、clip1。父 E20 冻结，累计恢复研发训练预算 E40。单候选固定最后 checkpoint，无 V 参数更新或目标反馈。源 V 原 1800 物理记录、clean 加六 practical，比较 identity/coarse/learned_operator；先完整固定预测，再独立 CPU scorer 读取 V TX 标签，使用原 F50 默认头。报告三郊区、三城区、clean、TX/RX/day 分层。

源波形可行性预先定义为六 practical 平均 IQ NMSE 相对 coarse 改善至少 5%，且 clean NMSE≤0.02。身份诊断另外报告原 F50 的逐视图 accuracy/Macro-F1、repair/damage 和每 TX 变化，不以 NMSE 代替身份结论。本 run 不读取 target；是否具备后续固定正式身份验证资格，须由这些合法源证据判断。多 seed、独立物理族和最终身份结论仍未完成。

|ID|报告要求|目标文件|状态|验证|
|---|---|---|---|---|
|A07-01|正确有限窗口广义线性作用及实伴随|pamf_action_restore/operators.py|verified|原生四场景重放、实内积、稠密 MAP 和6步目标下降测试通过|
|A07-02|观测估计网络没有 oracle / clean / TX 输入|pamf_action_restore/models.py|verified|接口、教师隔离、真实梯度和冻结父测试通过|
|A07-03|L/U 全量、fresh residual、0 TX/F 训练|pamf_action_restore/data.py、runner.py|implemented|源计数和实际曝光检查已接线，待完整日志|
|A07-04|冻结父来源、固定预算、checkpoint 往返|pamf_action_restore/runtime.py|implemented|权限/来源/预算负测通过，待真实源 checkpoint smoke|
|A07-05|源预测先固定、独立 F50 默认头评分|pamf_action_restore/evaluation.py|implemented|七视图完成负测通过，独立 CPU scorer 已接线|
|A07-06|三档郊区及三档城区分别报告|run report/results|implemented|七视图分层已接线，待完整源结果，禁止冒称 target|
|A07-07|不可覆盖输出、单 owner、共享容量锁|pamf_action_restore/runner.py|implemented|一次独立 P0/P1 无阻断，N607 预检 GPU5 空闲，待 post-state|
|A07-08|多 seed、独立物理族、正式目标身份验证|后续阶段|deferred|本阶段不产生相关完成声明|
