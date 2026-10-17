# 同 TX/RX/day 包络波动尺度的独立源检验

状态：已登记，未发布。只重用既有合法 L6300 的派生观测，0 新 IQ／checkpoint／网络／训练／增强／target。完整联合目标仍未达成。本分析不会启动训练或修改健康旧任务。

唯一问题：同一 TX/RX/day 内的包络波动尺度，是否存在跨 TX 和日期可迁移的 RX 共用规律，而不是 TX 特有变化？旧包络实验检验的是 RX 均值，已失败并保持原判定；本次检验离散程度，不调整旧均值试验的统计量或阈值。

输入固定为 PROJECT/runs/20261017-diagnostic-F-envelopeProfile-fullL-manysig-m1-r01/source_observations.npz 的 DCT1、opaqueID、可见 L 标签、RX/day；原全256点/16×16/logpower中心化定义不变，不重新读 IQ，不将 DCT2 作为失败后的替代。全部五源RX [1,3,4,6,8]、三日、六TX、90单元每单元70包，split392005/L6300/U56700隐藏/V27000保持，U/V不消费。复用现有 VALIDATED_ONCE 源句柄，无数据重建或哈希。

每单元计算 median(abs(DCT1−median(DCT1)))。固定 logMAD floor1e−12；每折同时排除一个 TX 和一个 day 拟合共享 RX 模式，五个 RX 全部保留。这是统计交叉拟合，不是 CVS 训练域留出。预测量和被留出单元均在 RX 维中心化；零模式、固定 RX 循环错配和同 TX 其他两日预测是独立控制。

预登记来源条件：共享 RX 的相对 MSE≤0.9，至少15/18折好于零模式，各 RX 不差于零模式，共享 RX 好于固定错配，且共享 RX SSE 不比 TX 条件 SSE 高出5%以上。任一不满足则关闭固定 DCT1／MAD／来源问题，不改坐标、分位数、RX、估计器或阈值挽救。所有90单元/18折均完成后判定，不删困难包。

通过也只支持来源相关性；不能证明 AGC 因果、身份保持、跨 RX clean 提升、星地收益或联合目标。当前尚未实现 IQ 增强，更未将该统计量作为拟合诊断分类器、部署状态或新增损失。后续如有依据，仍须独立的有界 IQ/default-head 验证及匹配 E30；本行不自动扩展。

执行：N607 普通账户，现有 NumPy，CPU2/CUDA空、无GPU。CWD PROJECT/releases/cvs_F_envelope_variance_source_20261018_r01；命令 CVS-RFFI/bin/python -u experiments/cvs_envelope_variance18/runner.py。唯一 owner codex/root/F-envelope-variance-source；不可覆盖输出 PROJECT/runs/20261018-diagnostic-F-envelopeVariance-sourceL6300-m1-r01；Git commit 为实际代码身份，发布写入 release_commit.txt/result.json。

技术停止仅限角色/输入缺失、非有限、输出冲突及系统失败，保留旧产物，不自动重试。预期 result.json（90单元/18折/5RX/成本）、completion.json、同row报告。数据源父行只有0NN观测，不涉及模型继承。此 checkpoint-free G0 不消费网络；真实checkpoint smoke不适用，不额外构建模型。

本地3闭式科学控制、完整90单元夹具、8个协议负例及compile通过；唯一限定P0/P1审查 PASS，未读取真实数据/权重或执行远程。发布遵循一次普通账户预检、一个归档的一次传输对比、远端compile和针对性检查，随后原进程/日志增长或自然终态独立核实。没有增加审批、签名或审计链。
