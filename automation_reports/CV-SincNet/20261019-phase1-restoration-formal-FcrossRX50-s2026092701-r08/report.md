# 原 F50 默认头对完整冻结恢复 IQ 的正式读出

状态：REPLACED / VERIFIED。原GPU4在预检后被其他三个健康worker占用，本行仅容量等待；尚未CUDA初始化、source smoke或目标读取。精确核实并撤下唯一run-owned等待PID，独立读回已退出，保留全部原记录，由R09 GPU5接替相同冻结矩阵。方法代码 `d5d014de2840baa4fe608412b29ff54cfb87f064` 已提交/push并核实远端OID，7项新增覆盖/协议负测通过；复用原R02/R03及F50各自一次独立P0/P1，不增加重复runner审查。N607普通账户预检GPU4空闲约24GB，R04/R05完整输入和原F50权重可用，新root不存在。

用户要求接原F基座验证恢复细节的身份。本行固定原 `F50/CROSS_RX-s2026092701`、原三路SIG_TIME、原Sinc坐标、原6×160默认头；F50历史E50/11100，新增训练/适配/头拟合均0。此前cleanFE200和AsFE30是不同读出，不能代替原F50。

固定identity、R04 coarse IQ、R05 residual IQ三组。复用原168000条正式received、同物理ID和已完成冻结的恢复字节，clean＋三郊区＋三城区；不重新训练、选模型、重排候选、生成信道或重新运行恢复器。R02/R03均按原source规则在目标前冻结，原F50源诊断也没有改变三组。R07独立训练，不参与本行。

首步真实2条L_s和原F50 no-query smoke，PASS即继续。七视图三组预测和特征全部固定、完整性检查通过后，独立CPUscorer才打开原truth。逐场景/TX/RX/day报告accuracy、Macro-F1、repair/damage；配对clean feature的异TX竞争池固定512位置，仅为truth-side辅助诊断，不是全库最近异TX，也不参与推理或选模。

郊区高/中/低三档分别报告，保留clean及任何类别/RX上的负收益。七视图共享物理样本，只有一个F和恢复seed；该已暴露工程代理基准不是新独立盲测或真实在轨证据，也不能证明每条生成细节均属于TX。完整矩阵、命令、环境、路径、GPU和停止规则见 [experiment.json](experiment.json)，追踪见 [R08追踪](../../../docs/PAMF_FORMAL_F50_R08_20261019.md)。
