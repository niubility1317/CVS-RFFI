# 非对称H/G任务与关系：源证伪

PLANNED；四候选只有一个学生seed2026092701。原同候选三个学生种子的raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%联合目标保持。本run无目标句柄，源通过不是最终性能。

## 依据与固定矩阵

Bridge四条完整源实验已结束、17760步/四E50默认头/544源单元/72方法配对均核实；heldG3差+.117284/+.166667/+.179012/+.203704pp，均未达原+.3pp，不扩其种子。当前只读源错误重叠表明，相同H/G输入的旧CE相对CL在held G独有纠正23但独有破坏33，在held H独有纠正61/破坏42。该描述统计存在目标/剂量差异，不证明GCE是因果，也不证明两loss相加能得到纠错并集；不做真值择优预测。

新假设是区分受损H的分类任务与恢复G的关系参照。全部使用完全相同的同包H_k(raw)/G(H_k(raw))两次额外学生前向，k按原三缓存场景轮换。原raw普通PL与完整T1 L保持。

| 方法 | 附加分类任务CE | 跨RX关系CL梯度 |
| --- | --- | --- |
| HJoint | H | H和G两侧 |
| HAnchor | H | 仅H；恢复IQ上的F特征detach作参照 |
| GJoint | G | H和G两侧 |
| BothJoint | mean(H CE,G CE) | H和G两侧 |

这里detach的是F在恢复IQ上的joint160，不是新增冻结G恢复器的操作；G在所有方法中原本就不更新。HJoint/HAnchor同一次前向loss数值相同，只有梯度路由不同；实际真实checkpoint中核实。HAnchor的G特征在CL为stop-gradient，G视图仍参与固定数值的跨RX关系；共享F参数仍由H/L/raw任务更新，不宣称整个F的G输出不变。

## 损失、权限与校准

固定相对CE系数1、CL系数1。CE用原raw三冻结E30全同类且各confidence≥.95的硬argmax；HJoint/HAnchor只计算H任务，GJoint只G，Both平均两项。每个CE完整U batch均值，使用既有task_loss精确一阶特征surrogate，附加分类weight梯度0，featurefusion norm/gain仍训练；原主任务继续联合训练默认头。

CL沿用Bridge：当前学生joint160，两个视图受理集合联合detached均值/std floor1e-4、L2eps1e-4，温度.2；正对同伪类不同RX，负对不同伪类，只跨视图，同包/同RX同类不作正对。有效anchor和除以2*完整U batch数。HAnchor仅对CL的G特征detach，不改数值、标准化或pair集合。所有附加loss及其标准化排除RX1，原L/rawPL全sourceRX照常；U真值隐藏。

CE1/CL1是本轮新假设，不声称照搬原监督loss权重或已最优。先12批原L128/U256无更新全参数源梯度，primary为原完整L+rawPL，CENTER_25K/25000物理换算保持；四条共同λ=min(.25,.10*median(primary_norm)/max_method median(CEplusCL_norm))，一次冻结、444步升温，不用V/目标选剂量。记录CE/CL值、H/G特征梯度和完整参数范数/cos；实际核实HAnchor G特征梯度0、其他route H/G均非零、HJoint/HAnchor值一致、全部附加分类weight梯度0及相同pair数量。初始代理不是后续AdamW或性能保证。

所有学生/教师/G参数和registered buffers不变，temporary buffers恢复，optimizer/V/目标/Utruth更新/读取0。72次完整参数梯度调用；另有CE特征surrogate的60次特征梯度计算及48次H/G路由核对，计算成本单列，无新增学生前向或训练步。缓存复用原H/G，不重建或读目标。

## 训练与固定源判断

四条own原ordinaryPL E20+10 E30独立+20/4440步至E50/11100、13L37LU，不继承任一研究F50。原SIG_TIME_X3/CENTER_25K、ordinary/raw RDrop、As CE且KL0、LS.01、浅中层ORD128跨RX.1及默认头保持；raw selfPL.95/weight1，nativeU/EMA off。

每row U1134000、held226800、额外接受838780、H/G学生FW各1134000。CE_H或CE_G实际受理数各838780（启用时），Both分别启用且系数各.5；特征CE梯度计算4440次，Both8880次；CL有效anchor和正负pair逐步记录，不能把anchor当独立物理包。新增80F轮，共享原三E30历史90轮及旧q/H/G缓存成本单列。

首GPU阶段真实E30/defaulthead/G L2 smoke和12批校准，随后BothJoint原L128/U256联合backward smoke核实原L768/AsKL0、隐藏U及有限主头/骨干梯度；PASS后立即四路训练。每rowfinal后自动sourceV27000八视图，原sourceShift同seed PLControl对照，保留raw/Graw/三H/三G、TX/RX/day/held/trained净纠错。native eval-only V健康检查允许，objective无V读取或更新。全四条完成后仍用heldRX1 G3≥+.3pp且overallrawclean≥−.1pp；最高合格heldG，精确同分方法名字母排序，数值使用净正确数计算。无通过者不扩种子/目标。

RX1只是附加分支留出，父模型/L/rawPL/G已见它，不是全网络未见RX。源通过后只支持另行固定同方法三学生种子raw/同G七视图正式验证，不更换默认方法或拼接最优项。

## 环境、发布与停止

N607普通账户，既有Python/Torch2.1，FP32/TF32off，各CPU2线程，无安装；CWD PROJECT/releases/20261026-phase1-PAMF-AsymmetricHG-sourceHold-release-r01。命令PYTHON -u -m experiments.pamf_channel_pl.asymmetric_hg_training dispatch --config该release/runtime.json。不可覆盖root PROJECT/runs/20261026-phase1-PAMF-AsymmetricHG-sourceHold-m4-r01，仅minimal source_context，无目标句柄。

GPU0..7按全局CUDA+pending<2、free≥10000MiB准入，最多4条本run训练/源评估，保护其他owner。一份archive一次本地/远端SHA比较、remote compile、独立PID/CWD/argv/GPU/stepgrowth。首次技术/协议/非有限/预算/smoke错误停止本run新派发，健康已启动子任务完成，保留产物；不自动重提/热改或干预其他任务。科学负结果完成全部四条。

7项聚焦检查及local compile通过；唯一一次独立P0/P1检查PASS，无P0/P1。源码/config与公共聚合进Git，IQ/权重/逐包预测/日志/runtime私有；不增加额外审批/seal/重复审查。
