# 当前普通增强的相位范围核对

本项为只读代码与运行配置核对，未训练、未读真实IQ/权重、未新增模型前向、未访问新正式分数。用途是避免把已存在的算子重新命名为新A_g；不能据此宣称相位噪声已有效或失效。

核对对象是run `20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02`，训练commit `63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47`。B00-s2026092702/stage20/actual_modules.json实际记录的DataAugmentation来自固定旧release的 `experiments/adv3b02_xuc/code/DataAugmentation.py`，不是顶层code；同一metadata中的leo_practical.channel、residual和SSDG.train_ssdg也来自该adv3b02_xuc路径。已直接读取远端该实际文件的函数，确认不是仅根据当前worktree默认值推断。

实际`_apply_phase_noise`生成每样本独立高斯增量`e`，用`phi=cumsum(e,dim=1)`，再计算`x*exp(j*phi)`；这是包内Wiener随机游走相位。实际ordinary_augmentation_actual.json中的base为p_phase_noise=.30、phase_noise_sigma_max=.006；原schedule在E13冻结，强度由原日程缩放。这两个base值不是每epoch/每包有效值，本文不把它们当作实际触发率。该配置同时记录RDrop_same_IQ=true与no_ordinary_before_practical=true：旧ordinary只位于共同普通视图，As直接从raw生成，没有在完整practical前再叠普通相位噪声。

历史C4是整包恒定复旋转：`(-Q,I)`、`(-I,-Q)`、`(Q,-I)`，其设计明确excluded phase noise；phase-origin冻结诊断也是整包公共相位/观察窗原点，不能将C4训练负结果解释成所有时变相位变换都已被检验。历史phasecycle采用旧RX1留出契约，只保留为历史证据，不在当前全部5源RX训练中复用其权重或去掉训练域。

因此，单独提出再加Wiener相位噪声会重复当前算子，并不能仅凭该名称满足新增A_g。没有新的源域可识别接收机机制或独立分类收益证据时，不为该重复项开训练，不扫描sigma/触发概率或将其和已失败的普通观察刷新相加。此结论只排除当前重复提案，不把未验证的其他物理相位模型一概判失败。

另已读取引用对话的完整shadowLHS E20+10终态报告及实际experiment记录：固定阴影抽样分层源判据失败、正式七视图accuracy均低于原As，关闭该固定路线。该4L/step、共用NoAs E20父再追加10的组织不同于本轮6L/step、各自scratch20+10的As基线；不将两者分数当作同预算同配方的直接配对差。

对应文件：`experiments/cvs_f_phasecycle20/design.py`、`analysis/cvs_F_phase_origin_next_20261016.md`；引用终态：`20261018-phase1-F-crossRX-shadowLHS-e20plus10-s2026092701-m4-r01/final_analysis.md`（其训练commit0a9e6541efc3a806cf3dc20e08b4c4ec9dc22ba3）。当前最小待完成实验仍为已发布的B00/B01种子02/03，加固定seed01复用控制的三seedAs确认；无新增候选或预算变更。
