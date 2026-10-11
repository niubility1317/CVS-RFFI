# 独立 P0/P1 发布审查

结论：**PASS，未发现尚未关闭的 P0/P1 问题。**

- 审查对象：`experiments/cvs_scratch_calibrated_style` 的设计、来源与模型、校准与训练目标、源训练、预检、预测评分、调度和运输，以及登记与相关汇总接口。审查者未参与这些实现，仅编写本记录。
- 矩阵与来源：登记的 36 行配置与 `design.config` 逐项一致；DELAY/CAP/DELAYCAP × M_SHALLOW/M_MIDDLE/M_BOTH/JOINT × 3 个固定种子。学生从零初始化，固定 20 轮、4440 次更新，无学生初始化 checkpoint；其他两个原 E20 教师的合法来源完整核验，固定 P0 与前批对照仅在独立评分/报告中使用。
- 校准时序：CAP 在初始化校准；DELAY/DELAYCAP 在第 444 次 optimizer 更新后校准，首次辅助步骤为 448。固定 12 对、1440 条 L 读出；校准不更新身份参数或辅助头，恢复模型状态、模式与 RNG，记录唯一 ready 事件与最终 resolved 配置，校准失败不会继续训练。
- 联合 cap：对完整、互斥的 shallow/middle/rest 可训练身份参数组测量梯度，覆盖 Hz 和 complex 参数。一次合成的两层 Fishr 项使用同一个 detached 标量缩放；身份参照含 clean CE、KL 与 style CE，不重复计入 KL 审计分解。保留负 Fishr 原值；raw/effective 日志与汇总校验一致。此限制是当前损失梯度比例，不宣称约束 AdamW 实际位移。
- 数据与目标：校准和风格机制仅使用合法 L；U 的 TX 真值隐藏，使用固定合法教师概率；V 仅无梯度健康验证。无 EMA、星地增强、目标训练或目标反馈。复用的浅中层 margin/Fishr、同 mask 风格路径与新时序接口一致。
- 控制与结果：固定 final20/4440 冻结；每行自动 clean 168000 条预测，prediction 固定后由独立 scorer 连接真值。36 行总预算 159840 次更新、720 个学生模型轮次；总预测数 6048000。发布包含本包与依赖，唯一 owner、已有输出拒绝、UNKNOWN 不自动重试，共享每 GPU 至多 3 个总进程/预留容量。前批固定配对报告只读已有 score JSON，缺失配对不补值、不据成绩改变训练。

已读回的验证证据：

- `calibrated_local.json`：PASS，36 行预算/配置、12 臂学生 checkpoint 往返与错误父来源拒绝。
- `calibrated_integration_cuda.json`：PASS，12 臂实际原生模型目标/梯度集成；这是局部集成检查，不冒充完整正式训练。
- `scratch_calibrated_cap_cpu.json`：PASS，完整参数覆盖、复数/频率/零梯度、联合缩放、关闭时等价，以及实际非零 KL/style 的 JOINT 目标。
- `scratch_calibrated_calibration_cpu.json`：PASS，三种时序、旧初始配方等价、第 448 步首辅助、非法角色拒绝、异常状态/RNG 恢复及唯一事件。

本次仅本地只读审查和配置核对，未使用 GPU、启动训练、访问远端或读取任何目标成绩/预测。正式训练与性能结果不在本审查的完成声明内。

审查记录时间（UTC）：2026-10-11T21:32:54.545925+00:00
