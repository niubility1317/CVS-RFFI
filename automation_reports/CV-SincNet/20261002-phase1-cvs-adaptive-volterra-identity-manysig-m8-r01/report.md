# CVS 可学习相位记忆残差：源实验预登记

状态 LOCAL_VERIFIED／未发布。两个全局零初始化 tanh 系数保留 coupled_lag4 初始函数，由 CE 学习三阶/五阶相位记忆残差。固定lag1/4×四seed，共8行；仅CE、无增强、身份骨干、原物理划分、scratch E200×50，202555参数（控制202553，新增2）。控制仅四份源元数据，不继承weights或读取测试成绩。

[完整数学、通信、物理、RFF结构分析](../../../docs/CVS_ADAPTIVE_VOLTERRA_PHASE_MEMORY_20261002.md)。62项聚焦检查及8个丢弃CPU模型24次公共CE更新通过；零初始化输入/共享state/logits与控制一致，两门实际取得CE梯度，整个网络仍不能声称频偏/RX/LTI不变或唯一TX硬件恢复。公共数值不等于识别提升。

固定源规则为最高四seed mean(0.5V+0.5最差RX)，完全并列才比较成本；只用E200。若新候选胜出，冻结后默认4份新clean预测＋28份固定控制，32行预测全部固定后独立truth-last评分；否则复用保留控制的既有测试，未选候选N/A。不追加LEO/SFT/support，不反馈测试调参。逐步及epoch日志记录两门梯度/优化前后原值/系数，完整源分层、实际12项输入、六block归一化、公共物理反例和资源全部保留。

[本地验证](evidence/local_validation.json) · [公共运行](evidence/local_cpu_smoke.json) · [公共汇总](evidence/public_probe_summary.json)。当前没有正式训练或新clean成绩，目标未达到。

发布前独立P0/P1审查PASS，23项直接检查通过；只读preflight核实原源控制、身份、资源及无输出碰撞。[审查](evidence/independent_review.json) · [preflight](evidence/preflight.json)。
