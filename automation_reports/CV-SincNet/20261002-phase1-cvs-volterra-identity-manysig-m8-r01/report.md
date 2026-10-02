# CVS 复数相位记忆：源实验预登记

状态 LOCAL_VERIFIED／未发布。仅 CE、无增强、身份骨干、原划分、scratch、固定 E200×50。两个前瞻相位延迟 lag1/4×四种子，共8行；当前 coupled_lag4 仅四份源元数据作为控制，不加载权重或目标成绩。

[结构与数学/通信/物理/RFF解释](../../../docs/CVS_VOLTERRA_PHASE_MEMORY_20261002.md)。仍为12复数输入、202553参数，同seed初始state与控制逐项相同；保持原六block与读出。公开合成/代数测试通过尚不证明性能或唯一TX器件恢复。源规则固定最高四seed mean0.5V+0.5最差RX，完全并列才比成本。

若新候选被源规则选中，冻结后默认4份新clean预测＋28份旧控制，32行预测全部固定后独立truth-last评分；否则复用保留源控制的已验证测试，未选候选N/A。没有LEO/SFT/support/query拟合或测试反馈。详细日志、全源分层、实际输入执行、公共物理诊断和资源按预登记保存。

本地验证 VERIFIED：56 项聚焦检查 PASS，8 个丢弃模型的24次公共 CE 更新 PASS；唯一独立 P0/P1 审查 PASS。只读 preflight 确认合法四份当前源控制、不可覆盖路径及空闲资源。[验证](evidence/local_validation.json) · [独立审查](evidence/independent_review.json) · [公共烟测](evidence/local_cpu_smoke.json) · [preflight](evidence/preflight.json)。尚无源或本轮clean性能结论。

## 实际发布与启动

状态RUNNING／VERIFIED。发布commit `ed6bdf431c4bad584fef774ea12ff23e9d20167a`，远端CPU检查PASS。独立读回dispatcher PID 1696395 和8个source进程、CWD/argv/独占输出/GPU0至7，全部已有实际epoch及日志增长。实际仅CE、scratch、无增强、身份骨干、固定50步/轮和完整FP32；202553参数均参与梯度，两个候选alpha0，实际12项Volterra输入、phase lag1/4和固定envelope lag4与配置一致。当前轮数 {'volterra_lag1-s2026092701': 9, 'volterra_lag4-s2026092701': 7, 'volterra_lag1-s2026092702': 7, 'volterra_lag4-s2026092702': 6, 'volterra_lag1-s2026092703': 5, 'volterra_lag4-s2026092703': 5, 'volterra_lag1-s2026092704': 4, 'volterra_lag4-s2026092704': 3}。尚无E200冻结或新clean成绩，不能宣称性能提升。[实际证据](evidence/running_readback.json)。
