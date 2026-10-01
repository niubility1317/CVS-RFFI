# CVS 全路径复相位约束：完整 FP32 源实验

状态 PLANNED，尚未发布。目标仍是普通 CE、无训练增强、仅身份骨干、相同物理划分及 clean-only 的性能提升，参数轻量其次，并取得整网可检验的 RF 物理一致性。

本轮网络结构保持原整网复相位等变记忆网络，202553 个参数、7199008 实数 Conv/Linear MAC；没有新增结构或参数。依据是新 query 前的源物理问题，以及固定公共输入的冻结单变量诊断：仅关闭 cuDNN TF32，原相位 logit 误差 0.0268415 降到 0.00003433。本轮从零训练，不继承原权重。原训练、源选择、clean 预测和评分均保留；不使用目标成绩、RX/TX 分层或失败幅度制定本轮策略。

固定 4 个 seed，L6300/V27000，U56700 不使用；原 source RX13468/day123/all6TX/完整物理ID；equalized1/center256/unitRMS/25MHz。E200×50、batch128、AdamW2e-4/wd1e-4/cosine1e-6、CE 权重1、无裁剪/teacher/EMA/域骨干/额外损失/增强。4 个新 scratch 与 4 个不可变 residual_fusion 源控制按原四seed mean(0.5V+0.5最差源RX)选择，性能最高优先、完全并列后才比较成本。

数值策略固定为 cuDNN.allow_tf32=False；matmul TF32=False、benchmark=False、deterministic=False、matmul precision=highest 必须与环境默认一致，唯一被切换的标志是 cuDNN TF32。训练、V 验证、冻结物理诊断、profile，以及选中后 clean 预测都采用同一策略。resolved 与 completion 记录实际 flags，40000 步和 800 轮记录实际 TF32 状态；进程内策略结束后恢复。

固定公共 30TX/RX 设置、3 相位与真实源 V 的 TX/RX/day 聚合只作机制证据。物理容差 0.001 继续仅报告，不新增选模或停机条件；不声明任意接收机不变或唯一 TX 硬件系数恢复。新候选胜出后默认 20261002-phase1-cvs-equivariant-fp32-clean-manysig-m24-r01：4 个新 clean 预测加20个旧冻结控制复用，168000原query、6TX、7RX，全部固定后独立truth-last评分。若原残差胜出，不测试未选新模型，不重跑历史。原控制使用历史精度，新候选使用全精度，报告明确此差异，不伪称完全同精度控制。

每 GPU 至多2个训练任务，只有已授权新实验使用空闲名额；不干预其他进程，不因低性能停机。详细文本与完整stepJSONL/紧凑epochJSONL/CSV均保留，实际成本完成后填入。GPU 训练、源表现、新 clean 结果当前 N/A，尚未证明性能提升或整体目标完成。

[源控制实时核实](evidence/source_control_preflight.json) · [源数值归因报告](../20261002-diagnostic-cvs-equivariant-numerics-public-m12-r01/report.md) · [既有数学与物理边界](../../../docs/CVS_EQUIVARIANT_IDENTITY_HYPOTHESIS_20261002.md)。

92 项相关检查 PASS，覆盖精度单变量/异常恢复、源实际 flags 与 payload 来源、全部日志状态、合法源冻结与 truth-last 以及控制复用。独立 P0/P1 审查 PASS；条件 clean 矩阵指错的唯一 P1 已修正并定点验证，完整旧矩阵不改写。
