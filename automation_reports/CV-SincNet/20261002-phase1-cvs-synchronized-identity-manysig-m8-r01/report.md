# CVS 全路径逐包频偏同步：两核心纯 CE 源实验

状态 PLANNED，尚未发布。目标保持性能优先、普通 CE、无训练增强、身份骨干、相同物理划分与 clean-only，并要求可检验的整体 RFF 物理性质。数学原型不等于识别性能提升，整体目标尚未完成。

依据为新 query 前的全量冻结源接收诊断及既有周期/已知激励诊断，未用 clean 成绩或 RX/TX 目标分层制定候选。逐包 80:160/20 点相关估计相对线性相位，全部 256 点同步后才进入全部身份路径。固定两个候选：synchronized_equivariant（202553 参数）和 synchronized_gauge（164225 参数），均零新增学习参数；仅普通 CE 联合训练身份核心。同步是确定的结构前端，没有随机训练增强。

估计主值 −625 至 625 kHz、1.25 MHz 歧义；有效相关且不跨主值边界时，仿射相位变换在同步后变为公共相位，后续核心处理公共相位。CFO 仍包含 TX−RX 相对频率，可能带有身份信息，不能直接称 TX 晶振参数。固定 coherence 阈值 1e−6、能量地板 1e−12，退化窗口不纠偏并记录。atan2 先替换退化输入，不隐藏分支歧义。保留逐点幅度及非仿射相位结构，不承诺唯一 TX PA/IQ 恢复或任意 RX/LTI 不变。

固定两候选×四 model seed（2026092701..04）新 scratch，L6300/V27000、U56700unused；原 RX13468/day123/all6TX/完整物理ID/equalized1/center256/RMS/25MHz/split392005。每行 E200×50、batch128、AdamW2e−4/wd1e−4/cosine1e−6、CE权重1、无裁剪/teacher/EMA/resume/域骨干/额外损失。新8模型均完整 FP32：cuDNN TF32=False，其余 matmul=False、benchmark=False、deterministic=False、highest 实际记录；旧残差控制保留原精度，解释为方法性能对比，不声称单因子因果控制。

8 个新源记录加 4 个不变 residual_fusion 控制，固定四 seed mean(0.5V+0.5最差源RX)最高选定，性能完全并列后再比较 V、最差RX、MAC、参数及固定顺序。物理误差只报告，不新增排名/停止门槛。源码无 target 读取路径，不加载任何旧权重。详细文本、完整 step JSONL、紧凑 epoch JSONL/CSV，实测前端相对 CFO/coherence/fallback/梯度/学习率/源 V/时间/显存，以及全 V 90 单元同步统计和几何全部保存。

若新候选胜出，默认 20261002-phase1-cvs-synchronized-clean-manysig-m24-r01：选中4行 clean 新预测加20个旧冻结控制只读复用，共24行168000原query/6TX/7RX，固定后独立truth-last评分。若旧残差胜出，不测试未选新模型，不重跑历史；保留已核实控制测试。测试结果不得回流调参、候选重排或选择性重跑。clean 已暴露，按既有代理基准解释；LEO/SFT/新增类及D92三阶段N/A。

每 GPU 最多2个训练任务，只使用授权空闲容量；不干预其他进程，低性能不能停机。技术异常保留原输出，仅本次所属 run 按预登记规则处理，不自动重发。源训练、最终源性能、物理误差、资源和新 clean 当前 N/A，尚未启动或证明目标。

[前瞻结构与边界](../../../docs/CVS_SYNCHRONIZED_IDENTITY_HYPOTHESIS_20261002.md) · [源控制实时核实](evidence/source_control_preflight.json) · [全量源敏感性依据](../20261002-diagnostic-cvs-received-sensitivity-source-manysig-m56-r01/report.md)。
