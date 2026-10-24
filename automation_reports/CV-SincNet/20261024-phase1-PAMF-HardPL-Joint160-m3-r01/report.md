# 普通伪标签与部署融合特征跨 RX 对比

LOCAL_VERIFIED_NOT_LAUNCHED。48 项相关检查及一次独立 P0/P1 PASS；真实原 E20 no-query smoke 为首个空闲 GPU 步骤，验证实际融合特征及签名分支梯度后立即重载父模型继续，无额外审批。

固定 HPL_E20_NAMS_JOINT6G、seed2026092701/02/03，原同 seed F21 E20 独立 native10＋20 到 E50／11100，13L＋37LU。仅原 E20 初始化，严格检查实际 payload、scratch 继承、原物理数据角色和默认联合分类头；不继承原 E30、前一候选或目标挑选权重。沿用 F0 VALIDATED_ONCE L6300／U56700／V27000，U TX 隐藏，V 不更新参数、归一化、EMA、原型或头。N607 普通账户、CVS-RFFI、FP32/TF32off、每 worker CPU2；命令、CWD、GPU、不可覆盖路径和预期产物见 experiment.json，实际代码配置以发布 commit 固定，单 release archive 一次本地/远端 SHA 对照、远端 compile，独立进程/CWD/命令/GPU/日志增长回读。

源依据：原 E20 完整 V27000/seed、0.95 伪标签精度99.9002%／99.8687%／99.9164%，实际 U 精度未知。固定原 E20 的 U960＋V720/seed 无更新配对诊断显示现有统计混合在 U 浅中层平均位移约6%至9%，接受样本无预测变化，伪标签 CE 仅约0.003至0.004；最终160维融合特征仍有条件于类的源 RX 描述性关联。style后继续使用原 IQ RX 标签，跨 RX 混合本身就会改变这种关联，不能将其降低解释为模型已学会不变性或因果去接收机证据。U 按硬伪标签分组，V 的真标签仅用于这次不更新的诊断。源诊断详见 source_style_geometry_E20.json 和原预定计划；历史目标暴露如实披露，新 E20NAMS／Guard 目标分数未读取。

唯一新因素：保留 GuardNAMS 的0.95门控、权重6、LS0.01、ordinary/raw RDrop1、As KL0、CE剂量1.25、浅中层跨 RX0.1、无 U 信道、六 U 前向共享 accepted-only 同伪类跨源 RX 统计混合（p0.5、Beta0.1、eps1e-6）和完整 L 梯度保护；在同一前两个 ordinary U 学生前向上捕获实际 CompositeReference.features 的 joint160，增加原生同伪标签跨 RX 对比。单 joint 系数0.05（等于原0.1两层平均的每层系数），除原 CE 归一化1.25，随后整体 U 乘6。原生温度0.2、分离的 batch中心/尺度、accepted-only 同伪类不同源RX为正、其他类为负；保留浅中层，不强制原始接收机响应参数相同。

实际融合 features 在浅中层后加入可学习 SIG_TIME_X3 分支。临时实例 MethodType wrapper 只覆盖这两次已有 U forward，保存并精确恢复原实例属性；不使用会被 .classify 直接调用绕过的分类头 forward hook。无额外前向、teacher/L捕获、新分类头、推理参数或G训练。真实原E20 smoke必须显示实际 [256,160]×2、512条复用packet、有效跨RX anchors、签名 sequence_pool 参数的辅助梯度非零有限、原L768/U1536/As raw128/Uphysical0与LPractical64、共享style plan及临时 wrapper 退出、保护后完整参数梯度与显式期望相同。探针状态丢弃，正式行重载原父权重。

每模型新增30轮/6660步，U1701000，joint6660次/捕获13320批/3402000条已有Upacket；这些属于原有10206000个新学生U视图，不增加数据或前向。累计 U/teacher2097900、studentU10602900、rawAs850500、style39960、共享plan6660。own10/20两段实际状态分别累加，原生 CENTER_UNITS AdamW、cosine、全模型物理 clip5/decay保持，完整L anchor/t2/cls_head/other投影与U/L范数帽1保持。额外joint对比反向计算和保留图有成本，报告实测父/增量/累计墙钟，不称同FLOP。无 Adam、L损失或目标性能单调保证。

GPU5/6/7一seed一卡；每seed只接替 GuardLS 全部GPU预测完成的卡位，且 GuardNAMS已派发。旧两臂尚未CUDA的PID与新child预留计数，每卡最多2本任务、含外部最多3进程、派发前空闲≥10000MiB。唯一CPU scorer排 Guard6 终态后；不停止、重启或热改健康任务。两个相同技术失败停止新增派发、保留健康进程及产物，不按目标结果停止、调参、排序或选择性重跑。

训练固定最终E50学生，自动raw和同一固定DURA v1.0各clean＋六practical，每view168000，三seed共42cell；该row所有预测固定后独立truth-last。报告每seed/场景/TX/RX/day/RX×day、Macro-F1、修复损伤/U曝光/实际成本；分别比较固定F20/F21和同恢复读出。原raw验收同时clean≥+1pp、六景均值严格>+1.5pp，DURA只作独立参考，G权限和Fseed SD范围保持，不套旧F50时延。已有历史及E20LS2目标结果已观察，不称独立盲测；新假设按固定源证据在新E20NAMS/Guard目标分数读取前冻结，goal ACTIVE。


## 已发布并核实排队

VERIFIED_CONTROLLER_QUEUED：执行 commit `77498c5a8b003f3afc71fab7ee96ca39662593fa`；单包传输对照及远端编译通过，独立核实控制器存活、CWD／完整命令／PID匹配发布目录，状态心跳继续增长。三模型仍排队，新GPU worker为0，实际原E20 joint／签名梯度 smoke尚待首卡位，不能标成训练或smoke成功。只接替同seed GuardLS全部GPU预测完成后的卡位并保留已派发GuardNAMS；原健康任务未改动。


## 真实原 E20 检查通过并实际启动三种子

VERIFIED：发布commit `77498c5a8b003f3afc71fab7ee96ca39662593fa` 的真实原E20 GPU source smoke PASS。实际joint160两次捕获512条已有Upacket，跨RX有效anchors=470；实际签名sequence_pool辅助梯度范数=0.435258482且有限非零，原L768／U1536／rawAs128／Uphysical0／LPractical64、共享style计划、完整L guard参数梯度与显式期望相同及临时wrapper退出均符合。探针丢弃，正式行重新加载原E20。

三个seed已按GPU5/6/7启动并实际产生原生joint／guard源步骤；独立核实进程／CWD／完整命令／GPU及日志增长，逐行现有U复用计数见source_activation.json。只按原矩阵接替释放卡位；当前是训练前缀，不代表全部源训练结束或目标收益。训练、raw＋固定DURA七景及truth-last继续自动执行，goal ACTIVE。


## 固定源特征轨迹诊断

额外的只读源诊断，不改变或阻挡现有训练／冻结／测试。矩阵预定原F21 E20、原F21 E30、EarlyNAMS E30、GuardNAMS E30各三seed（A阶段24个角色行），以及当前Joint固定own E30三seed（B阶段6行）；不按目标分数挑checkpoint。所有模型用相同U960／V720物理源cohort，U统一用原同seedE20伪标签及0.95接受集合，V用全部720真标签仅作不更新诊断。检查实际部署eval特征，无临时style扰动，分别报告浅中层、joint、主干残量、实际签名与含gain的参考响应贡献；同时保留TX分离／总散度／分支幅度，不能将坍塌或接收机关联降低当成成功。unit-only和全cohort CRIC式标准化均为描述性统计，不拟合头／投影、不读U真值／target，不作候选晋级门槛。6项相关检查通过；代码配置以Git版本固定，实际checkpoint严格继承检查及sourceL2 no-query调用在CPU诊断中完成。完整矩阵、指令和边界见source_geometry_trajectory_plan.json。


## 固定源轨迹 A 阶段完成及冻结头干预

A阶段24角色行完成，只读CPU24.025秒，完整源统计见source_geometry_trajectory_A.md/json。U共用原E20接受/伪类、V720全保留，浅中层TX分离提高而joint没有同步改善；参考响应强RX关联可能同时起补偿作用，不能由此删除。预定C阶段仅原E20与GuardNAMS E30各三seed、12角色行，冻结同一默认头，检查零分支及同伪类跨RX分支交换；不另拟合头、不新增骨干前向/训练/GPUworker、不读U真值/target、不作晋级或正式性能替代。八项相关源/冻结头检查通过。当前GPU训练及七景truth-last不变，B阶段固定Joint E30诊断在对应原生阶段冻结后继续。


## 源轨迹 B／C 完成与后续互补监督设计

B阶段Joint固定E30六角色行、C阶段12角色行冻结头检查均完成，详见source_geometry_trajectory_B_C.md/json。源Joint融合TX类间比例约0.771→0.804、条件RX/总散度约0.0859→0.0663，非target准确率；无签名的既有头视图产生约0.17至0.22的源伪标签CE，完整视图仅约0.003至0.004。保留签名部署，研究将ordinary U CE的0.25系数转给无签名视图、总CE剂量1.25不变，其他joint/浅中层/RDrop/style/guard保持；现阶段仅固定设计，未实现／审查／启动新模型。完整公式、三seed、E20+30预算、双七景及卡位安排见complementary_U_followup_design.json。当前GuardNAMS／Joint target分数未读取，原健康任务不变，goal ACTIVE。


## 固定原E20的源域信道学习难度诊断

追加只读CPU2诊断，原三seed F21E20、同一既有U960/V720、raw gate0.95及原生Practical三景训练混合，在固定epoch21/step4441与epoch41/step8881模拟。固定aug gate网格0.8／0.9／0.95，各含仅置信与置信＋raw预测一致两种统计，共18条源role/view行。完整源权限、路径入口和停止规则见source_view_reliability_plan.json；6项相关检查通过。源V真值只统计，模型/归一化/EMA/原型/头更新与Utruth/target/G训练均0，不改现有健康任务、配方、父来源或冻结矩阵。raw伪标签是否正确与扰动后是否易于学习分别报告，后者不能当作U真实正确率或target表现。当前Joint/Complement目标分数未读取，历史目标暴露保留。


原E20源域信道学习难度诊断完整18行返回，见source_view_reliability.md／JSON。nativeAs与raw标签一致率约74%至77%、hardCE约0.61至0.66；这不是U真值正确率。源依据固定下一独立原E20+30对照：只有当前eval学生对模拟As置信>=0.95且与raw伪标签同类时采用该As，否则精确raw回退，保留所有原acceptedU、原六前向和全部CE/KL权重。增加每步半U的无梯度As eval和模拟生成，成本/实际采用/回退须另计，无新头或推理参数；尚未实现/发布，完整定义见reliable_As_followup_design.json。本设计先于Joint／Complement／Path新目标分数读取，历史目标暴露保留，不改变正在运行的配方。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，无U物理生成、EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。新增rawAs850500、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽1及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。


## HPL_E20_NAMS_JOINT6G 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.938383 | 92.792593 | 96.174074 | 16 |
| 2026092702 | 99.868683 | 99.941833 | 93.074074 | 95.511111 | 15 |
| 2026092703 | 99.916428 | 99.957447 | 93.066667 | 95.740741 | 11 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_NAMS_JOINT6G.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。


## 完整目标测试结果

ANALYZED：各方法三个seed，共3模型、42项raw／固定DURA七景决策均完成独立truth-last评分，六个匹配监督／普通PL对照模型完整复用。逐景／seed／SD／Macro-F1／RX／TX／day、修复损伤及曝光／成本见target_results.md和附属完整表。原raw双门槛尚未满足，研发目标仍ACTIVE。

HPL_E20_NAMS_JOINT6G / raw clean：85.641666667% ± 0.713566851%（三个模型 seed 样本 SD）。
HPL_E20_NAMS_JOINT6G / DURA-LU clean：85.685515873% ± 0.736583593%（三个模型 seed 样本 SD）。
HPL_E20_NAMS_JOINT6G / raw six_mean：70.171230159% ± 0.401872872%（三个模型 seed 样本 SD）。
HPL_E20_NAMS_JOINT6G / DURA-LU six_mean：78.080753968% ± 0.979306516%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。


HPL_E20_NAMS_JOINT6G 固定匹配比较完成，见 HPL_E20_NAMS_JOINT6G_matched_comparison.md；比较不替代原主目标或全六行完成状态，无目标反馈改配方。


## 源域互补分支诊断 D 预登记

固定比较 Joint／Complement／Path 三个方法各三 seed 的既有 own E30 默认头，复用原 E20 硬伪类、接受掩码、U960／V720 与七种既有头特征干预；合计 18 个 source role 行。检查互补 CE 是否改善无签名／主干识别裕量及完整融合，不据此选择目标 checkpoint、门槛或重排当前矩阵。已知 Joint 及此前完整目标结果，尚未读取 Complement／Path 目标分数；只作源域机制解释。源 V 不更新模型或归一化，U 真值隐藏；CPU2、GPU0，无 G、额外训练或新头。命令、固定模型和输出见 source_geometry_trajectory_plan.json 的 phase_D_registration，检查数 8。


源域诊断 D 已 VERIFIED 完成，18行／三方法各三seed、20.490 s、无任何训练或目标读取。全部既有头干预、固定源角色和几何见 source_geometry_trajectory_D.md；不据此改变已发布方法或声称目标收益。


## 完整特征跨接收机外推的源域检验

预先固定原F21E20三seed、U960/V720及alpha0/0.25/0.5/1，只在高置信同伪类、不同源RX的两个完整joint160之间作h_i＋alpha(h_i−h_j)，保留签名/参考/主干及原默认头。类共同项在加性假设下保留，这只是待检验假设，不是实际信道或目标收益证明。选择最大非零alpha须每seed同时达到V真实正确率≥99%、U原伪类一致率≥99%、U hardCE较raw至少增加25%，并满足预登记有效样本数；无幅度通过就不启动候选。无V梯度/任何模型或状态更新、Utruth/target/G读取、新头或GPUworker。原父权重、伙伴、网格、规则、命令和产物见source_receiver_extrapolation_plan.json；14项focused检查通过。当前其余健康实验保持。


完整特征接收机外推源域检验已VERIFIED：6行/24case，CPU2 6.561s，无状态更新/V梯度/Utruth/target/GPU。仅alpha0.25通过预先固定的所有seed保真和hardening规则；完整结果及alpha0.5/1失败均见source_receiver_extrapolation.md。独立Joint6G后续RXE配方已固定在receiver_extrapolation_followup_design.json，实现/审查/真实smoke待完成，未启动新训练，不据目标调门槛或替换父。


## 完整源梯度记录诊断

固定只读GuardNAMS、Joint、Complement、Path各三个seed的全部E21至E50原生记录，共79920步，核实权重6已包含的U梯度在完整L保护后的保留比例、各组冲突/范数帽和逐轮分布。后续RAS/组合/外推仅在各自三seed源E50完成后按同一规则补充，当前健康训练不等待本诊断。记录量、矩阵、命令和解释边界见source_gradient_trajectory_plan.json。12项计算/输入检查通过；不读IQ/模型权重/Utruth/target、不更新模型，不构成新候选或发布门槛。日志测量发生于共同clip5及AdamW/decay之前，不能当作实际优化步或独立改变权重的因果效果。
