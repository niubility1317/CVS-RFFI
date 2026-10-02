# CVS 可学习相位记忆残差：源实验预登记

状态 LOCAL_VERIFIED／未发布。两个全局零初始化 tanh 系数保留 coupled_lag4 初始函数，由 CE 学习三阶/五阶相位记忆残差。固定lag1/4×四seed，共8行；仅CE、无增强、身份骨干、原物理划分、scratch E200×50，202555参数（控制202553，新增2）。控制仅四份源元数据，不继承weights或读取测试成绩。

[完整数学、通信、物理、RFF结构分析](../../../docs/CVS_ADAPTIVE_VOLTERRA_PHASE_MEMORY_20261002.md)。62项聚焦检查及8个丢弃CPU模型24次公共CE更新通过；零初始化输入/共享state/logits与控制一致，两门实际取得CE梯度，整个网络仍不能声称频偏/RX/LTI不变或唯一TX硬件恢复。公共数值不等于识别提升。

固定源规则为最高四seed mean(0.5V+0.5最差RX)，完全并列才比较成本；只用E200。若新候选胜出，冻结后默认4份新clean预测＋28份固定控制，32行预测全部固定后独立truth-last评分；否则复用保留控制的既有测试，未选候选N/A。不追加LEO/SFT/support，不反馈测试调参。逐步及epoch日志记录两门梯度/优化前后原值/系数，完整源分层、实际12项输入、六block归一化、公共物理反例和资源全部保留。

[本地验证](evidence/local_validation.json) · [公共运行](evidence/local_cpu_smoke.json) · [公共汇总](evidence/public_probe_summary.json)。当前没有正式训练或新clean成绩，目标未达到。

发布前独立P0/P1审查PASS，23项直接检查通过；只读preflight核实原源控制、身份、资源及无输出碰撞。[审查](evidence/independent_review.json) · [preflight](evidence/preflight.json)。

## 实际发布与启动

状态RUNNING／VERIFIED。发布commit `db3860587aa6d1c031a9dd9b9226cfc38f51f513`，远端CPU检查PASS。独立读回dispatcher PID 1728043 和8个source进程、CWD/argv/独占输出/GPU0至7，全部已有实际epoch及日志增长。实际仅CE、scratch、无增强、身份骨干、固定50步/轮和完整FP32；202555参数均参与梯度，两个候选alpha0，实际12项可学习相位记忆输入、phase lag1/4和固定envelope lag4与配置一致；两个新增系数和梯度已实际记录。当前轮数 {'adaptive_volterra_lag1-s2026092701': 21, 'adaptive_volterra_lag4-s2026092701': 20, 'adaptive_volterra_lag1-s2026092702': 19, 'adaptive_volterra_lag4-s2026092702': 19, 'adaptive_volterra_lag1-s2026092703': 18, 'adaptive_volterra_lag4-s2026092703': 17, 'adaptive_volterra_lag1-s2026092704': 16, 'adaptive_volterra_lag4-s2026092704': 15}。尚无E200冻结或新clean成绩，不能宣称性能提升。[实际证据](evidence/running_readback.json)。

当前独立读回8个进程均存活，已完成 50 至 57 轮。实际两门系数/梯度、202555梯度参数和公式输入均已测量；完整源分析工具已就绪，但完整E200与固定源选择尚未完成。[当前读回](evidence/progress_readback.json) · [简表](evidence/progress_summary.json)。

实际远端Torch2.1下，8个独立公共scratch模型共享state与控制逐项相同，零门初始logits完全相同；未访问正式数据/weights/query。[远端初始化验证](evidence/remote_initial_control_equality.json)。
