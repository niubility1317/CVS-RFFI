# 任务梯度缩放身份参照的源留出验证

PLANNED。三个方法变体，只有一个 student seed2026092701。原目标仍为同一个三种子方法达到 raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G六景>78.07463624338624%。本run不访问目标、不自动扩种子，源结果不能替代该目标。

从同一原ordinary-PL E30/6660独立追加20轮/4440步到E50/11100，累计13L+37LU。原完整L监督、raw ordinary/RDrop、As真实CE且KL0、LS.01、浅中层ORD跨RX、raw U自身eval PL阈值.95/权重1及默认头联合训练保持。native U/EMA关闭。L6300/U56700/V27000原物理角色、VALIDATED_ONCE句柄和源合同保持。

## 方法与剂量

| 方法 | 两次新增输入 | 新增身份参照损失 |
| --- | --- | --- |
| GUniform | 相同G(scene_k)两次 | 原均匀参照关系CE |
| GTaskScaled | 相同G(scene_k)两次 | 按当前默认头的硬身份任务梯度范数缩放 |
| RawTaskScaled | 相同raw两次 | 与GTaskScaled相同的缩放规则 |

各包参照关系CE为r，现有默认头硬伪标签CE为h。缩放a=stopgrad(min(32,||grad_z h||/(||grad_z r||+1e-12)))，两次完整学生前向损失先平均，完整U batch归一化。缩放只提供特征梯度范数约束，不承诺骨干或AdamW更新界限。参照项不向现有头增加参数梯度，不增加新头/投影。

共同新增权重0.03257680992215036，444步线性升温。已完成12批无更新全模型源校准按min(.25,.10*median(primary)/max_variant median(extra))确定；GTaskScaled加权范数比为10%、GUniform约4.49%、raw缩放约0.78%。不从源V或目标选权重。证据见[源剂量](../20261026-diagnostic-PAMF-TaskScaledRef-dose-m1-r01/source_decision.json)。

复用同一原三E30 raw置信度≥.95且同类共识q/mask及三景U缓存，不读取U真值。每条新增分支排除RX1；原L/raw U主项保持全源RX。参照仅来自旧RXReference-dose保存的visible-L6300、own E30坐标，不从新剂量run寻找references.pt，不跨seed平均特征。参考、raw教师和G不更新。

G固定原F0 G-LU；源U恢复缓存保持low suburban/mid urban/low urban顺序，global step轮换，硬件/信道seed20261016423/20261016421、namespace PAMF/restored_dose/U/fixed。每条预计U曝光1134000、heldRX曝光226800、新增接受838780、两次新增学生前向曝光2268000；报告实际动态scale sum/max/cap counts。接受率不是U真值准确率。

## 源证伪与收尾

三条固定final E50后逐行立即预测完整源V27000八视图：raw/Graw及三种H/GH，完全匹配既有sourceShift同seed PLControl的物理顺序和source scene。保留逐TX/RX/day和新增分支留出/参与RX的净纠错。原native无状态V健康推理允许；新objective_V_reads=0，所有V更新=0，不能据此声称训练进程从未读取V。

预先固定原判据：heldRX1三个G恢复场景均值相对PLControl≥+0.3pp，整体V rawclean≥−0.1pp。三条全部完成，只在通过者中按held G增益最大选未来三种子候选，精确相同则名字母序；无通过者不扩展。原父模型、主项与L参照见过RX1，因此这里只是新增U分支留出，不能称为真正未见RX。当前控制器没有目标预测或后续扩展阶段。

## 发布与停止

原父E20/own10实际CP/header、refs/cache/source-only剂量来源和精确合同在加载前核对，源context无目标句柄。代码commit由runtime.code_commit记录，使用不可覆盖run/release；不增加数据重验或成员哈希。

命令：PYTHON -u -m experiments.pamf_channel_pl.task_scaled_training dispatch --config PROJECT/releases/20261026-phase1-PAMF-TaskScaledRef-sourceHold-release-r01/runtime.json。CWD为该release，输出PROJECT/runs/20261026-phase1-PAMF-TaskScaledRef-sourceHold-m3-r01。N607普通账户既有Python/Torch2.1、FP32/TF32off、不安装依赖。所有GPU0至7可用，优先全局CUDA+pending负载最少且空闲最多的卡，总进程<2/卡、free≥10000MiB；最多3条本run train/eval，各CPU2threads，保护其他owner。

12条相关聚焦检查和新模块compile PASS。一次独立P0/P1审查在发布前记录。launcher首步为实际own E30默认头L2以及实际L128/U256联合backward smoke，核对L768包、AsKL0、有限梯度和合法额外接受量；通过立即运行三条训练。一次archive SHA传输比较、remote compile及启动后独立PID/CWD/argv/GPU/loggrowth按原最小流程执行。

首个技术/协议/smoke/非有限/预算/状态错误停止本run新启动；健康已启动行完成，不自动重试、不热改或干预其他任务，保留全部输出。科学负结果仍完成全部三条。新增60 F轮；历史共享E30教师90轮（已含own30），合计150唯一F轮。G、L参照与U缓存历史成本单列复用。私有权重、IQ、packet预测、日志及runtime不提交Git。
本候选唯一一次独立 P0/P1 审查 PASS，无 P0/P1。真实 checkpoint 联合 backward smoke 由 launcher 首步执行，通过后立即训练。
