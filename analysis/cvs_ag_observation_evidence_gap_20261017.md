# A_g 的当前证据缺口与关闭边界

2026-10-17。状态：没有已晋级或待发布的新 A_g；完整联合目标 ACTIVE。本文是本轮历史证据整理和输入缺项说明，不是新实验、审批层或数据重验证。本轮新增真实 IQ／checkpoint／NN／更新／教师／target 均为 0。

## 当前事实

最新完成的六步 L 重排是训练组织实验，两条同预算 scratch E20／各 4440 更新／源 RX1 七视图共 17640 判决已独立复算。O1 相对 O0 clean −1.825397 pp、Macro-F1 −2.012662 pp、low urban −0.555556 pp，固定方案关闭；正对更均衡并未成为识别收益。完整结果见[两行报告](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01/report.md)。正式 target 未访问。

底座仍为 F＋跨 RX 同 TX、三路 SIG_TIME、统一 Sinc、真实 joint160、默认 cosine 头正常联合训练、NoExplicitCFO 和必要内部逐包补偿、原 13L＋7U／oldgate／ownEMA。最新两行实际 args 均为 wisig_equalized='1'、wisig_out_len=256、source_fold_held_rx=1、seed2026092701、epochs20。source-only 有效契约为 L5040／U45360（TX 隐藏）／V21600，held RX1 原 L1260 固定 final20 评估；不冒充完整 L6300 或 L_train5580/L_dev720。

## 本轮排除的重复入口

以下归纳已有完整记录，不是新评分，不把某个固定配方的失败扩大成整个算子家族不可能。

|入口|已核实的具体结论|当前行动|
|---|---|---|
|同 TX/day 跨 RX 实测 IQ 插值|三行 E20 的 crossRX 相对 raw clean +.476190 pp、相对 sameRX −.476190 pp，未通过原条件；且另一对话曾研究同类方法|固定 .1 对齐混合保持关闭，不换供体或系数救跑|
|周期差异残差有色噪声|L5040／72 格／18 折：合并形状风险改善35.211296%，但43格 q10 coherence 不达标，RX6 风险反向17.668361%|不由这些残差生成 colored-noise A_g，不换窗口/lag/RX/阈值救跑|
|四周期采样抖动估计|4320 合成格／135 组，134组通过，唯一必要组可辨认率30/32低于原95%|原入口关闭；不能改周期、幅度或不确定度规则后晋级|
|频域相位闭合量|L5040 中2548包未通过必要频点，RX3/8几乎全未通过；没有合法全RX折|不只挑RX4/6，不换频点或质量规则救跑|
|非 DC 局部谱峰／窄带成分|另一对话的完整源报告：同 RX/bin 仅14/18；所选峰接近 L-STF 谐波，不能归为独立杂散|不据此抹除频点或追加窄带干扰增强；不改窗、频率或阈值救跑|
|公共相位、常数增益、轻度频偏、有界非恒定相位、独立 AWGN|各固定匹配 E20 已完成并未通过各自原源条件|保持各固定配方关闭，不合并跨实验增益|
|旧 PA／循环移位／AWGN-multipath 修正|源冻结头恢复或旁路统计不等于训练后跨RX收益，已完成的匹配负结果保留|不把 ordinary 修正冒充新增 A_g|

本轮原本检索的同 TX 插值和窄带干扰均被已有记录排除，没有重复发布。谱峰的完整原结果是 source-only 的物理依据检查，没有模型识别结果，见 [canonical 源报告](../../../../automation_reports/CV-SincNet/20261017-diagnostic-source-nonDC-spur-crossfit-manysig-m1-r01/report.md)。其余精确 run 与按时间保留的结果见[活动记录](cvs_joint_augmentation_active_20261016.md)、[周期残差](cvs_cycle_residual_covariance_basis_20261017.md)、[抖动](cvs_sampling_jitter_source_basis_20261017.md)、[频域相位](cvs_frequency_phase_observability_basis_20261017.md)。

## 仍有物理动机、但当前无法构造的观察

未决的一般假设是：**同一个合法物理包的真实接收处理变化，可能比任意随机扰动更适合构造身份保持视图。** 这仍不是有效 A_g 的证明，也不能据此重开已关闭入口。

当前官方流程不保证 raw/EQ 同下标配对：[equalize_dataset.m](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/matlab_equalization/equalize_dataset.m) 仅追加成功均衡的包，保存内容未保留原 pkt_i；[compact 构造](https://github.com/WiSig-dataset/wisig-subset-creation/blob/master/data_utilities.py) 对不同表示各取前缀。因此同 sig_i、相同 TX/RX/day 或 IQ 相似度不能证明同包。当前原角色、准备记录及源折元数据也没有补足 raw↔EQ 映射。实际错配率仍未知，不声称全错配。

[PKL 构造](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/python_pkl_creation/py/build_dataset_multirx.py) 固定 pkt_len=256 并截取前缀，compact 不恢复包外样本；现存输入未提供原 capture 偏移或参考区间。这个长度结论来自构造和元数据，不冒充本轮逐包扫描。详见[配对审计](cvs_paired_preprocessing_basis_20261017.md)、[连续窗口可用性](cvs_contiguous_window_availability_20261017.md)。

本轮仅询问是否另有同一批合法源 L 的原始处理日志、物理包对应表或可追溯长 IQ 文件。该信息尚未收到，不能把等待当作已授权新增数据，也不能把未回答推断为不存在。原 raw/EQ 与连续窗口入口保持关闭。

最小下一动作取决于真实输入：如提供已有记录，先只按原物理 ID 核对是否确为同包及原 L，随后才另行固定身份保护检查和匹配 E20；如没有可靠对应，或需要猜配对、借 U/V/held/target、扩大 L 才能构造，则不开展此实验。即使配对成立，默认头身份保护或后续跨 RX clean 不能通过，也应放弃，不能靠预处理后波形更像、loss 更低或健康检查晋级。当前没有登记、发布或追加训练。

## 文献不能填补的缺口

[Ma 等的多接收机研究](https://arxiv.org/html/2512.12070v1)在 LoRa 谱图、预训练与 Siamese 流程中组合信道增强和跨接收机同设备正对；其高 SNR 分解近似不证明当前 CVS IQ 可任意相加或改相位。[Pan 等的 residual-channel 方法](https://arxiv.org/html/2412.08885v1)从可用 pilot 进行 LS/MMSE 估计并构造不同均衡残差，再结合 SimSiam 和新环境微调。它为真实预处理视图提供动机，但不恢复本项目丢失的物理映射，也不授权新增预测头、目标微调或更改正式 received IQ。本轮定向阅读不是穷尽综述，没有把论文成绩移植为 CVS 收益。

没有声称现有数据上的 A_g 普遍不可能，也没有把目标缩减为关闭候选。尚缺的仍是：有效 A_g 的跨 RX clean 证据、稳定 A_s 的 practical 适应、同预算同契约 B00/B10/B01/B11 及交互项、跨 seed 与正式 clean＋六 practical 的联合验证。low urban >60% 和 suburban 接近自身 clean 均未证明。


本轮实际补充：全频段源PSD正仿射模型已做432个同时留TX/day拟合及独立复算，0/6 RX对符合固定条件，模型关闭。全部输入谱权重低于0.5，最佳相对RX均值风险改善3.094277%低于20%；不能据相对原谱的误差下降生成噪声IQ。它不依赖同包映射，此负结果也不证明其他观测机制不可能。详见[本行完整结果](../automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-PSDaffine-fitL-aggregate-m1-r01/report.md)。
