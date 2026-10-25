# 恢复与未恢复 IQ 的任务 CE 源对照

PLANNED，四个候选、一个学生 seed2026092701。代码身份为发布 runtime.code_commit。目标仍要求同一候选三个学生种子同时达到：raw clean≥85.98174603174603%，raw 六景>72.0961970899471%，同 G-LU 六景>78.07463624338624%。本 run 只做源域证伪，不读取目标 IQ/真值，不把源增益当成目标结果。

## 可行判断与已有反证

固定 G 与固定信道对同一包的输出可重复；重复这份恢复 IQ 本身不会创造新的信道变化。恢复输出仍与原始 IQ 有差异，不能假设 G(H(x))=x。原三种子直接 G-softCE 增强比匹配 PL 的 raw clean低1.282540pp、同G六景低1.656746pp，保留为反证。高质量 raw 伪标签的源 V precision 是99.968297%，只是V代理，不是未解封U的正确率；U原缓存共识接受率93.3792%。因此本轮不主要靠继续抬高置信阈值。

固定 L 参照的剂量从1x到3x时，参与额外训练的源RX G3收益上升，heldRX净增却从36降到28次，均未过原+.3pp。最新L诊断还发现：重算当前均值改善G参照，却降低H参照并减弱默认头任务的局部梯度对齐。没有依据直接刷新在线参照。架构为GroupNorm/LayerNorm，没有BN运行均值/方差，不把现有退步归因于常见BN污染。

本轮可检验假设是：给恢复残差分布额外施加分类更新可能影响原分类方向；只让额外任务梯度训练现有特征，并同时覆盖未恢复H，可能更适合 raw 和同G联合目标。该假设尚未获得性能支持。

## 四条固定路线

H表示对原U IQ叠加相同合法源星地信道后的IQ，G表示同一包H经固定原G-LU恢复。三景固定低suburban/中urban/低urban，与旧G缓存和namespace完全相同，不以新名字重复声称增加场景。

| 候选 | 两次额外前向 | 额外CE对分类权重的梯度 |
| --- | --- | --- |
| GJointHard | G、G | 保留，匹配硬标签对照 |
| GFeatureHard | G、G | 阻断 |
| HFeatureHard | H、H | 阻断 |
| HGFeatureHard | H、G | 阻断 |

实际分类阶段为归一化cls_head.weight；同模块内base_norm/pa_norm/gain参与z特征融合，仍可训练。所有组的原L/rawPL主项继续联合训练完整既有默认头。额外CE通过对同一z求特征梯度并detach，构造同值、同一阶骨干梯度的替代项，不产生额外分类权重梯度，不另训头或投影。不能把这称为整个cls_head模块冻结。

四组使用同旧三E30 raw unanimous≥.95 q缓存的argmax硬标签；U真值始终隐藏。额外分支排除RX1，主监督及普通raw selfPL仍使用原全部RX。接受流与全U batch归一保持，两次额外损失先平均，444步升温。每组从原own E20+10 E30独立追加20轮/4440步至E50/11100（13L+37LU），不加载任何现有目标测试后挑选的F50。原T1监督、普通增强/RDrop、As真标签CE且KL0、LS.01、ORD跨RX与默认头保持，nativeU/EMA off。

预期每组U曝光1134000、held曝光226800、额外接受838780、额外学生前向2268000；HG各H/G1134000，其余对应视图2268000。实际逐步曝光、接受、H/G计数、原机制指标与非有限值必须完整核对。新增4尾部共80 F轮；共享三E30前缀历史90轮、旧q/G缓存及PLControl的已有成本单列，不计为新训练也不称零成本。

## 先进行源配对与剂量校准

第一个GPU阶段先完成实际E30/defaulthead/G的L2无query smoke，再生成原U56700的H三景缓存，沿用旧namespace PAMF/restored_dose/U/fixed；同U前2包×3景经固定G与旧G缓存核对。只使用U IQ与隐藏元数据，不生成U真值标签。

同原E30、固定seed01的12批L128/U256测原完整L任务+rawPL以及三个额外分支的完整参数梯度。校准无optimizer/V/目标，temporary buffers恢复，所有学生/教师/G参数与registered buffers不变。Sinc梯度按原CENTER_25K/25000物理坐标换算。令P为primary norm中位数，λG=min(.25,.1P/max(median GJoint,median GFeature))，λH=min(.25,.1P/median HFeature)。G两组共用λG，H用λH，HG各一次按各λ后平均。这只是固定E30的梯度预算代理，不保证后续AdamW更新或目标收益。所有类参数保持原结构和主项更新。

校准完成自动进行HGFeatureHard的实际原L128/U256联合backward smoke，核实原L768前向、AsKL0、有限主分类权重梯度和额外路由；通过后立即并行四条训练，不另等确认。实际校准也核实GFeature/HFeature附加分类权重梯度严格为0、GJoint非零。

## 固定源筛选与后续目标

每条固定final E50后立即完整V27000八视图，对照sourceShift已冻结同seed PLControl的同物理ID/相同场景，逐TX/RX/day与额外held/trained RX报告。native只读V健康检查允许；新objective不读V，全部V状态更新为0。所有四条完成后，沿用heldRX1 G三景均值≥+.3pp、整体V rawclean≥−.1pp的原门槛，合格者选heldG增益最高，精确同分按方法名字母序；无通过者不扩展，不放宽门槛。

这是额外分支留出诊断，原父模型/L/rawPL/G已见RX1，不能称真正未见baseRX测试。后续若晋级，应独立冻结最终三学生种子方案及raw/同G正式七视图矩阵；完整预测后独立truth-last评分。最终必须同一候选同时满足原clean/raw六景/同G目标，而不是把不同候选最优项拼接。

## 环境、发布与停止

N607普通账户既有Python/Torch2.1，FP32/TF32off，无安装。CWD PROJECT/releases/20261026-phase1-PAMF-FeatureTask-sourceHold-release-r01。命令PYTHON -u -m experiments.pamf_channel_pl.feature_task_training dispatch --config该release/runtime.json。独立输出PROJECT/runs/20261026-phase1-PAMF-FeatureTask-sourceHold-m4-r01，旧产物不可覆盖。

所有物理GPU0至7可用，按全局CUDA+pending最少、空闲最多调度，每卡总worker<2后才准入且free≥10000MiB，各CPU2线程，最多4条本run train/eval并行；保护其他owner健康任务。一次archive SHA传输比较、remote compile和启动后独立PID/CWD/完整命令/GPU/loggrowth核实。

首次技术/协议/smoke/非有限/预算/状态错误停止本run新启动，健康已启动子进程继续，保留产物，不自动重提、热改或干预其他任务。科学负结果不提前终止四条固定矩阵。源码/config与公共聚合登记进Git；权重、IQ、逐包预测、日志、runtime留私有。


## 发布前已完成验证

7项聚焦测试通过：同值与精确一阶特征梯度、分类路径梯度阻断、零接受全U归一、错误feature graph/可训练target拒绝、source/seed/budget/head/route负向检查、真实任务依赖图、隐藏U与RX留出/HG曝光、梯度系数规则。四个实现文件local py_compile通过。唯一一次独立P0/P1检查结论为PASS，无P0/P1；未进行重复审查或新增门槛。实际checkpoint smoke仍是发布后的首个GPU阶段，并自动接续固定流程。
