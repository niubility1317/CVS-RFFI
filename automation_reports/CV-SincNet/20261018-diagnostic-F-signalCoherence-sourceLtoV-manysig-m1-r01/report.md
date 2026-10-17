# 已完成：周期信号形状源统计

ANALYZED／VERIFIED。V差异SSE降94.711077%，18/18折、5/5RX改善，但共享残余为同TX控制2.609059倍，固定条件失败关闭。0真实CP/NN/更新/teacher/target，无新IQ增强或身份成绩。

[完整同一行结果与解释](source_analysis.md)；[独立复算](independent_recount.json)。执行提交 a4c557c5706a85427e190906c231f7cbe231de4e。以下保留原冻结预登记，以本段和实际完成产物为当前状态。

---

# 周期信号相关形状：全源 L→V 单项预登记

状态：IMPLEMENTED_LOCAL_VERIFIED，尚无真实源结果。唯一问题：消去公共增益和相位后，周期信号的相关形状是否具有可由其他 TX、其他日期预测的共享 RX 差异。这不是 IQ 增强、身份识别或联合成功；不自动启动模型训练。

最小一行 `PERIODIC_SIGNAL_PROJECTOR_L_TO_V`，实际唯一配置见 experiment.json 和 experiments/cvs_signal_coherence18/design.py。全部源 RX[1,3,4,6,8]／day[1,2,3]／6TX、完整 L6300 与原 V27000、256 点、split392005 保持。所有 RX 始终参与统计；leave-TX/day 是统计预测折，不是留出训练域。U56700 的容器和 TX 真值不读取，只用原不透明 ID 句柄核对角色互斥。既有 source_cache 的未使用特征随容器反序列化，不参与计算；不重建或重复验证数据。

旧周期残差实验先去掉周期均值，再测残差 lag1 至 4 相关。本项保留周期均值，测其全部20×20复相关矩阵。旧频域相位闭合量、cyclecov、采样抖动、FIR、酉相位和 RXMIX 的固定失败结论保持；不改变其统计量、核、幅度或阈值救跑。本项不会将新的统计关联自动解释成那些旧算子的身份保持证明。

[WiSig 原论文第 IV-B 节](https://arxiv.org/pdf/2112.15363)说明均衡后重新采样回25 Msps再取256点；[Non-HT PPDU 文档](https://www.mathworks.com/help/wlan/gs/non-ht-ppdu-structure.html)说明20MHz L-STF的10个短周期总长8μs。二者给出名义20点周期的动机。这并不证明每个当前 IQ 窗口严格周期化；实际逐包 coherence 和周期均值能量比例完整报告，不过滤低质量样本。

固定使用已有256点中的[80,160)，盲lag20相位进展仅用于本观察量的逐包补偿，估计值不作为身份坐标或持久模型状态。四个20点周期的均值为m，平均周期总能量为E，定义

    u = m / sqrt(||m||² + 1e-12 E)
    P = u uᴴ

分母为0时u=0，样本仍保留。P为有界半正定矩阵，消去公共相位、非零增益；这些只是数学不变量，不构成任何 IQ 变换的标签保持证据。它包含TX、信道、RX、均衡与有限噪声，不能识别纯硬件原因。原网络 NoExplicitCFO 和必要内部补偿完全不变，因为本行没有网络或输入变换训练。

每个TX/RX/day的全部70个L和300个V分别均值聚合P。每个18折由L中其他5TX和其他2day的50个单元估计5RX共享偏移；先在每个TX/day内对RX中心化，再跨训练单元平均，系数固定1。预测对应V的5RX差异，V只用于评价，不进入拟合、优化器、EMA、归一化或分类器状态。控制为零RX偏移与同TX、其他2day的L条件预测；不另拟合诊断分类器。

固定反证条件全部满足才称“源周期形状关联”：V中RX差异的平均Frobenius范数至少.02；相对零偏移SSE至少降低20%；至少15/18折SSE改善；每个RX总SSE不增加；共享RX的总SSE不超过同TX条件控制的1.05倍。只用1e−20绝对SSE数值容差。任一失败关闭本固定周期信号投影问题，不改窗口、周期、矩阵条目、特征向量、RX集合或阈值。通过仅为统计来源证据，后续仍缺可解释 IQ 算子、默认头身份保护及匹配 E30 的真实识别检验；不将本条件当成联合目标替代。

成本固定：33300个原源IQ观察／666000个保存复因子条目／180个20×20单元矩阵／18统计折；0模型构造／CP／NN／teacher／训练轮／真实更新／头拟合／target。CPU两线程，无GPU。私有observations.npz保存20复数因子与质量量，source_ids.json保存物理ID，均不提交Git；独立recount从全部保存因子重算矩阵和全部统计，不再次读取IQ。聚合统计与报告可以提交。

执行CWD `PROJECT/releases/cvs_F_signal_coherence_source_20261018_r01`，命令 `CVS-RFFI/bin/python -u -m experiments.cvs_signal_coherence18.run`；本地发布入口 `F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_signal_coherence18.publish`。唯一输出 `PROJECT/runs/20261018-diagnostic-F-signalCoherence-sourceLtoV-manysig-m1-r01`，唯一owner `codex/root/F-signal-coherence-source-20261018`。仅所属技术异常、权限/角色错误、非有限、形状或输出冲突停止并保留产物；不自动重试，不干预健康旧任务。预期checks、resolved_config、私有因子/ID/矩阵、完整result/completion、独立recount、同row源报告。

本地16个聚焦负例、增益/相位/小频偏不变量、共享RX正例和TX-only／L-V反转／TX反转／日期反转共5个统计控制通过。41个合成IQ包观察、0真实IQ／CP／NN／更新。一次直接P0/P1独立审查PASS，未访问真实IQ/CP/Utruth/target或远程、未扩大测试。实际代码/config由一次Git提交标识，publish将该实际HEAD填入不可变release登记。普通N607一次preflight、单archive本远一次SHA比较和remote compile，launcher第一步同一组角色/测量smoke通过即继续；G0不引入未消费的CP或NN。历史多审查、设计SHA、签名链、强制64行或审批为REJECTED_EXTRA_GATE。

整体基线仍为F＋跨RX同TX／默认cosine正常联合训练／完整L-U-V／原U规则／全部5RX／256点／E30。上一逐视图二阶矩四行已经正式失败关闭；当前没有已验证新Ag，本源行不会改写旧结果或预告low超过60%。
