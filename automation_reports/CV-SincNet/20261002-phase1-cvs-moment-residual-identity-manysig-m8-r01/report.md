# CVS可学习包内矩残差：源实验预登记

状态LOCAL_VERIFIED／未发布。瞬时与记忆4两种固定矩定义各4seed，共8行。沿用adaptive_lag4相位记忆输入，加入两个零初始化全局tanh系数，由源交叉熵学习三阶、五阶包内矩残差。202557参数，比控制增加2；原通道、深宽、读出、物理划分、scratch E200×50、CE唯一、无增强、身份骨干、FP32不变。当前控制仅读取4份adaptive源元数据，不继承权重或使用目标分数。

[数学、通信、物理及RFF设计](../../../docs/CVS_LEARNED_MOMENT_RESIDUAL_20261002.md)。66项聚焦检查及8个公共模型24次CE更新通过。保留原参照和初始函数，使非零新系数能够改变网络；是否改善源性能和独立clean性能尚待验证。包内矩不是TX硬件系数，不宣称完整混合输入正交、CFO/RX/LTI不变或实测在轨验证。

固定源排名为最高四seed mean(0.5V+0.5最差源RX)，只用E200，完全并列后比较成本。若新候选胜出，冻结后默认4新clean预测＋32固定控制，共36行、288总体/RX指标，全部预测固定后独立truth-last评分；否则复用控制既有测试，未选候选N/A。只测clean，不追加LEO、SFT、support或新类，目标结果不回流结构、矩定义、系数、epoch、seed或选择性重跑。

保存详细文本、10000步JSONL、完整/紧凑epoch JSONL及CSV。四个全局系数均记录实际梯度、raw/tanh更新前后值；每轮末源包测量实际12项输入、原一阶IQ、相对残差、矩及六归一化块。最终保存完整90源分层、公共通信链与资源。公共阈值只报告，不参与选模或低性能停机。

[本地检查](evidence/local_validation.json) · [公共运行](evidence/local_cpu_smoke.json) · [公共汇总](evidence/public_probe_summary.json)。尚无正式训练或新测试成绩，总体目标未完成。

发布前独立P0/P1审查 PASS，无阻断项。实际源 preflight 核实4份控制的完整角色、来源、预算与FP32，确认用户/主机、8张GPU空闲、磁盘容量和新输出无碰撞。条件clean路径与既有兼容路径62项合成fixture检查通过，尚未创建正式clean配置或访问目标。[审查](evidence/independent_review.json) · [preflight](evidence/preflight.json) · [条件clean检查](evidence/conditional_clean_validation.json)。
