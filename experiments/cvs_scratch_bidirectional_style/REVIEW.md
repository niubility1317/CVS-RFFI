# 独立 P0/P1 发布正确性审查

结论：**PASS，未关闭的 P0/P1 为 0。**

审查范围为本包的新有界变换与方向池、设计和配置、训练调用/日志分析接口、来源与冻结评估、调度运输，以及 `20261012-phase1-scratch-bidirectional-style-manysig-m18-r01` 登记。审查者未参与实现，仅编写本文件；未修改的原基础机制沿用既有核验，不重复全量检查。

## 关键核验

- 18 行登记配置逐项等于 `design.config`：三种固定校准策略 × POS/BIDIR × 三种子。每行 scratch20/4440，总计 79920 次更新、360 个学生模型轮次。学生不加载初始化 checkpoint；合法 peer 教师仍为另外两个原 E20 seed，完整来源核验保留。
- `bounded_style.transform` 使用 detached μ/σ 与合法同 L 身份、同 day、跨 RX、不同物理 PID 供体；私有种子决定供体及幅度。构造的均值位移不超过 ramp×自身 σ，缩放范围为 [2^(-ramp),2^ramp]，显式特征梯度为该正缩放因子。无供体或零 ramp 的输出通过分支逐位保留输入。常量特征、重复 PID、有限性和私有 RNG 路径已有直接检查。
- 同策略同 seed 的 POS/BIDIR 配置，multilevel 仅 `style_pool` 不同；方向依次为全正与 [1,-1,1,-1]。两池同索引使用相同四个私有种子、供体及幅度。浅中层按原顺序处理；中层实际统计取自该完整候选的上游结果，没有跨候选拼接。
- clean＋四增强按整批合法 L CE 取最大。clean 仅严格较大时被选中，平局保留原增强；增强平局取最小索引。选中 seed 和 direction 在训练中重放，clean 分支不装 hook；恢复原 clean Dropout RNG，训练前向可微，候选 eval/no_grad 不代替训练梯度。仍是 5 eval＋1 train，3996 次训练重放、19980 次候选 eval。
- 校准、二阶 margin/Fishr、联合 cap、20/4440 固定终点及实际辅助预算未改变。U 真 TX 隐藏，仅固定 peer soft KL；V 无梯度健康诊断，不参与拟合或选择。无 EMA、星地增强、U 风格或辅助可训练头。三个校准策略全部保留，无目标反馈。
- 源分析校验方向池、选中方向、每层实际构造界、截断率、零 hook clean 分支与全部执行计数。新增报告按 clean/正/反三类核对总搜索次数，层统计排除 clean 调用；截断率注明是调用内通道比例再取调用平均。BIDIR−POS 用于方向比较；与旧 m9 的固定比较明确还包含 log-scale 与截断差异，不宣称仅方向变化或真实 IQ 物理不变性。
- 每行 final20/4440 冻结后自动预测 168000 条 clean query，prediction 固定后由独立 scorer 连接 truth；18 行共 3024000 条决策。主比较为 DELAYCAP_BIDIR−DELAYCAP_POS。固定跨批比较只读已有评分 JSON，缺种子完整均值为 N/A，不影响队列。
- 新 run/release/owner/output 独占；共享每 GPU 至多 3 个总所有者，包含既有任务/预留，UNKNOWN 不自动重试，不覆盖旧输出或修改健康任务。发布 preflight/dispatch 指向新包，60 个运输路径均存在。

## 已关闭的问题

审查中发现运输列表曾漏掉 `experiments/cvs_scratch_identity_style`，而新 `design.py` 依赖该包。主线程已补回；独立执行 `publish.configure()` 只读核对，新旧两包均在运输列表且路径存在。该启动依赖 P1 已关闭。

## 验证证据

本 run 的 `evidence/` 已读回以下 PASS：

- `bidirectional_local.json`：18 行预算/配置、六臂模型 checkpoint 往返与错误来源拒绝。
- `bidirectional_integration_cuda.json`：六臂原生完整 objective 局部集成与有限梯度，目标访问为 false。
- `bidirectional_style_cpu.json`、`bidirectional_style_cuda.json`：两池共六个受控分支，实际前后向、同 mask 重放、状态/RNG/hook 恢复、异常清理与实际非覆盖 CE 读出。
- `bidirectional_transform_cpu.json`、`bidirectional_transform_cuda.json`：三个 ramp 的统计界、互逆显式缩放梯度、相反均值位移、常量输入、无供体、重复 PID 与 RNG。
- `bidirectional_report_checks.json`：两池实际合成读出聚合与方向计数/层统计可渲染。
- `reference_checks.json`：六项固定参考比较与缺种子行为，仅合成数据。

本次只进行本地只读代码/配置/证据检查并创建本文件，未启动训练、使用 GPU、访问远端或读取目标成绩/预测。PASS 为已审范围内的发布正确性结论，不表示正式训练完成或性能收益已验证。

审查时间（UTC）：2026-10-11T23:15:03.977943+00:00
