# 独立 P0/P1 发布正确性审查

结论：**PASS，未发现未关闭的 P0/P1 问题。**

审查范围为本包设计、配置与来源、新 clean 候选、训练调用及日志分析、模型冻结与预测评分、发布依赖、容量与输出保护，以及 `20261012-phase1-scratch-identity-style-manysig-m9-r01` 登记。审查者未参与实现，仅写本文件；沿用上一轮已核实的未修改基础机制，未重复全量审查。

## 核验结果

- 登记与矩阵：DELAY_JOINT、CAP_JOINT、DELAYCAP_JOINT 各 3 个固定种子，共 9 行；登记配置逐项等于 `design.config`。学生 scratch20、4440 次更新，合计 39960 次更新、180 个学生模型轮次。相对前批相同 arm，multilevel 唯一变化是 `include_identity_candidate=True`，新 run/release/owner/output 独占。
- 来源与数据：学生不加载初始化 checkpoint；每行固定另外两个原 E20 教师且排除当前 seed，完整合法来源核验路径沿用。L 为唯一候选选择/校准监督来源，U 的 TX 真值隐藏，V 仅 eval/no_grad 健康诊断。校准策略全部保留，固定 P0 和前批 JOINT 仅由评分/报告比较，无目标反馈。
- 新选择机制：四个增强先按整批 CE 最大值选择，增强平局取最小索引；仅 clean CE 严格较大时选择 clean，精确平局保留增强。clean 索引为 -1、seed 为 null、实际层/hook 字典为空、改变数为零。仍执行 5 个 eval/no_grad 候选读出和 1 个可微训练重放，不逐样本拼接候选。
- 梯度与状态：clean 重放不安装风格 hook，使用普通 clean 前向之前的 Dropout RNG；增强重放保持原有合法 L 供体与顺序 hook。退出及异常恢复随机流/训练模式并移除 hook。新分支仍进入真实 style CE 与身份梯度，联合 cap 的身份参照包含该 CE。原校准 0/444、首次辅助 448、999 次辅助及 3996 次额外 L 训练前向预算不变。
- 记录与分析：索引 -1、空层集合与零 changed 均可被源分析消费；分析核对五候选最大值、平局、实际 hook、物理 ID 与完整计数，单列 clean 选择次数。raw/effective Fishr 与校准 ready 生命周期沿用已审查路径。两处旧 12-arm/random 文案已修正。
- 测试与控制：每行 final20/4440 冻结后自动预测相同 168000 条 clean query，prediction 固定后独立 truth-last 评分；9 行共 1512000 条决策。调度和评分以新包/9 行闭合；共享每 GPU 至多 3 个总进程，包含已有所有者和预留。既有输出拒绝覆盖，UNKNOWN 不自动重试，健康旧任务不修改。59 个运输路径均存在，包含新包及既有依赖，远端入口指向新 preflight/dispatch。

## 已读回的本地验证证据

本 run 的 `evidence/` 中：

- `identity_local.json`：PASS，9 行配置/预算及模型 checkpoint 往返、非法来源拒绝。
- `identity_style_cpu.json`、`identity_style_cuda.json`：PASS，clean 获胜、增强获胜、精确平局三分支；实际原生前后向、非零有限身份梯度、同 mask 重放、模式/RNG/hook 恢复、异常清理和非法角色/不一致日志拒绝。受控 CE 仅用于强制覆盖分支，另有未覆盖 CE 的实际读出检查。
- `identity_integration_cuda.json`：PASS，三臂完整 objective 局部集成，目标访问为 false。
- `reference_checks.json`：PASS，三个预固定跨批配对及缺种子时完整均值 N/A；仅合成评分 fixture。

本次未启动训练、占用 GPU、访问远端或读取任何目标成绩/预测，仅进行了本地只读核对并创建本文件。PASS 表示已审范围内的发布正确性，不代表正式训练已完成或性能改善已获验证。

审查时间（UTC）：2026-10-11T22:33:09.473547+00:00
