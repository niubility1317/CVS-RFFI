# 同物理包恢复残差插值：四路源验证

PLANNED，只有student seed2026092701，四候选，不是三学生种子。实际代码身份为发布runtime.code_commit。本run没有目标句柄，不把源表现当作目标收益。原同候选三种子raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%的联合目标保持。

## 源依据与固定矩阵

FeatureTask四条完整E50/sourceV已完成并独立核算。G联合/G特征/H特征/HG特征的heldG3增益为+.123457/+.141975/+.086420/+.111111pp，均未过原+.3pp；G分类weight隔离只多净增3次。参与RX的G特征收益+.442901pp高于held+.141975pp，HG对held未恢复H三景+.283951pp却不能扩大G收益。仅扩大固定端点训练或隔离分类weight不足。旧U RDrop/跨RX/统计混合及多种梯度保护已有实验，不重复命名。

本轮改变训练输入支持：m=(1−α)x+αg，其中x为原始源U，g为同一物理包的旧固定G恢复结果，α逐包Uniform[0,1)。只做训练增强，不声称它是真实信道或新恢复器。两端同包、同一高置信raw伪标签，仍需实际识别验证保真。输入凸组合借鉴[mixup](https://arxiv.org/abs/1710.09412)，通用文献收益不外推到PAMF。

| 方法 | 两个额外学生输入 | 最终附加系数 |
| --- | --- | --- |
| MixBase | M1、M2（独立α） | 固定0.05941532657537612 |
| MixDose | M1、M2 | 新12批源梯度的MixMix剂量 |
| PairBase | M1、精确G | 同固定系数 |
| PairDose | M1、精确G | 新12批源梯度的MixG剂量 |

同路线Base/Dose共享α流。两路线每次均由私有CUDA generator抽2n个α，MixG不用第二个α，但保持跨batch首M完全匹配；不改变原生CPU/CUDA RNG。精确α=0/1端点由lerp保留，不额外做相位/幅度拟合或G回退门控。源校准报告raw/G相干、残差幅度和混合RMS，不能仅凭数学插值保证TX保真。

## 配方、剂量与预算

四条均从原own ordinaryPL E20+10 E30独立+20轮/4440步到E50/11100，累计13L+37LU；不继承FeatureTask或其他F50。原完整T1监督、raw selfPL阈值.95/权重1、LS.01/ordinary与raw RDrop/As CE且KL0/浅中层ORD跨RX.1、默认头原位联合训练保持，nativeU/EMA off，原L/U角色及物理ID保持。

附加CE仍用旧三E30 raw全一致且各confidence≥.95的q之argmax，U真值隐藏；只从附加分支排除RX1，主L/rawPL使用全部原sourceRX。额外损失每次按完整U batch均值归一、两前向平均、444步升温。分类阶段weight不接收附加梯度，特征融合norm/gain仍接收；不另训头、不更新教师/G/参照。

Base来源为已完成source-only FeatureTask12批校准的λG，不使用其V或目标选择系数；复用GFeatureHard最终源结果作同系数G/G端点控制，不重复训练。Dose按每route新12批固定源L128/U256全参数梯度计算λ=min(.25,.10*median(primary_norm)/median(route_extra_norm))；primary为原完整L＋普通rawPL，保留CENTER_25K/25000物理单位换算。额外计算同G/G参考，用于解释幅度变化。上限.25可能使实际初始梯度比低于10%；该规则只约束初始代理，不保证后续AdamW更新或目标收益。

校准无optimizer/V/目标，temporary buffers恢复，所有F/G参数及registered buffers不变；实际CUDA检验私有抽样不改变原生RNG，并检验首M匹配、附加classificationweight梯度为0。第一GPU步骤先实际E30/defaulthead/G的L2精确端点/featuregraph smoke，再12批校准；随后PairDose原L128/U256联合backward smoke核实原L768包、AsKL0、有限主头/骨干梯度，通过立即串联四条训练。

预期每row U1134000、held226800、额外接受838780、额外学生FW2268000。MixMix的M FW/使用α计数2268000、G端点0；MixG的M/α1134000、G端点1134000；所有row私有α抽样2268000。逐步保存实际曝光、门控、weight/ramp、M/G计数及使用α的count/sum/square/min/max/四区间bins。新增80 F轮；历史3E30共90轮、旧q/G缓存与PLControl/GFeature控制成本单列，不计新训练也不称零成本。

## 源评估与选择

每row固定final后自动完整V27000八视图，对照原sourceShift同seed PLControl相同物理ID和source namespace，报告raw、Graw、三H及三G、TX/RX/day/held/trained和净纠错。native只读V健康检查允许；objective不读取V、全部V状态更新0。全四条完成后，仍以heldRX1 G三景平均≥+.3pp且overallV rawclean≥−.1pp作门槛；合格者选heldG最大、精确同分方法名排序。无合格者不扩种子/目标。RX1只从额外分支留出，父模型/L/rawPL/G已见它，不是未见baseRX证明。

通过源筛选只支持后续另行冻结三学生种子、raw与同G正式七视图目标矩阵。不能拼接不同方法最优项，也不把本run称为目标成功。

## 环境、发布与停止

N607普通账户既有Python/Torch2.1，FP32/TF32off，各CPU2线程，无安装。CWD PROJECT/releases/20261026-phase1-PAMF-ResidualMix-sourceHold-release-r01。命令PYTHON -u -m experiments.pamf_channel_pl.residual_mix_training dispatch --config该release/runtime.json；不可覆盖输出PROJECT/runs/20261026-phase1-PAMF-ResidualMix-sourceHold-m4-r01。只读取minimal source_context，目标配置不打包。

物理GPU0..7均可准入，按全局CUDA+pending最少/空闲最多调度，每卡总<2后才准入、free≥10000MiB，最多4条本run train/eval，保护所有其他owner。一次archive本地/远端SHA比较、remote compile和独立PID/CWD/完整命令/GPU/loggrowth核实。

首次技术/协议/smoke/非有限/预算/状态错误停止本run新派发，健康已启动子进程继续，保留产物，不自动重提/热改/干预其他任务。科学负结果完成全部四条。7项聚焦测试与local compile已通过，唯一一次独立P0/P1检查PASS，无P0/P1；无额外设计审批或重复审查。权重/IQ/逐包预测/日志/runtime保持私有，源码/config与公共聚合进Git。
