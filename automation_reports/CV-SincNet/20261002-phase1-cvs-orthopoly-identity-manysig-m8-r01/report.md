# CVS 包内正交包络输入：源实验预登记

状态 LOCAL_VERIFIED／未发布。两个固定输入配方 orthopoly_instant／orthopoly_memory4，各4seed，共8行。仅改变首个行为卷积输入：保留原IQ，利用同包加权矩去除三阶、五阶项的相关部分，不增加网络深度/宽度或参数。202553参数，控制202555。原物理划分、scratch E200×50、CE唯一、无增强、身份骨干、FP32不变；当前源控制仅4份adaptive_volterra_lag4元数据，不继承权重或读取测试成绩。

[数学、通信、物理及RFF设计](../../../docs/CVS_PACKET_ORTHOGONAL_ENVELOPE_20261002.md)。65项聚焦检查与8个公共CPU模型24次CE更新通过。公共30种TX/RX链上的实际卷积输入Gram、原IQ与三阶/五阶重建、能量、常相位/频偏响应及TX/RX不可辨识反例均保存。包内正交改善坐标条件是待验证假设，不代表准确率提高或硬件因果解耦。

固定源排名为最高四seed mean(0.5V+0.5最差源RX)，只用E200，完全并列后比较成本。若新候选胜出，冻结后默认4新clean预测＋32固定控制，共36行，全部预测固定后独立truth-last评分；否则复用保留控制既有测试，未选候选N/A。只测clean，不追加LEO、SFT、support或新类，目标结果不回流结构/尺度/lag/选模/重跑。

详细文本、逐步JSONL、epoch完整/紧凑JSONL与CSV保留；诊断四延迟的Gram、原IQ、重建、各阶能量和近退化，六归一化块、完整源分层、公共物理及资源。

[本地验证](evidence/local_validation.json) · [公共运行](evidence/local_cpu_smoke.json) · [公共汇总](evidence/public_probe_summary.json)。当前没有正式源训练或新clean成绩，目标未达到。

发布前独立P0/P1审查 PASS，无阻断项。源侧 preflight 核实实际四份源控制完整角色/来源/预算/FP32、身份、GPU容量、磁盘及无新输出碰撞。[审查](evidence/independent_review.json) · [preflight](evidence/preflight.json)。

正式发布与运行 VERIFIED：release `541889b1ec5982773d20f17315d0e0d6300ecee1`，dispatcher PID 1765080；独立读回8个worker真实PID/CWD/argv/GPU、实际202553参数/完整FP32/alpha0/CE唯一/全参数梯度与28包输入测量，进度E14至E20。远端8公共模型24次CE smoke PASS。完整E200源冻结及条件clean尚未完成，不称识别性能提升。[启动读回](evidence/launch_readback.json)。
