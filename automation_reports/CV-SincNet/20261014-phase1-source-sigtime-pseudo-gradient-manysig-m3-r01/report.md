# 固定 SIG_TIME：源域伪标签错误与特征梯度预登记

状态 ANALYZED，完整三模型源诊断及独立保存产物复算已完成。实现提交 d17f140c3528d8964ffedf464976532bc56e0760；不是新的性能结果。固定已有 SIG_TIME112→joint160/default cosine6×160 scale30，三个模型种子 2026092701/02/03 取预定 RANDOM 控制 final50 及其实际自身 EMA，不依据目标结果选权重。原模型 scratch33+17/11100 更新、旧门控+cos 来源由既有 source_proof 核对。没有另训、替换或单独重训分类头。

## 问题与边界

旧门控源审计、PAIR 覆盖实验和 TRUSTED_WEIGHT 已经完成，本轮不重复把覆盖增加或梯度范数变强当成收益。缺失的观测是：在固定 SIG_TIME 中，错误伪标签对真实骨干参数梯度有什么影响，以及实际部署 joint160 的弱/强视图一致性是否仅在描述上与源域错误相关。本轮只完成这一固定观察，不扫描权重、阈值、轮次或父模型，不推出 U 准确率和泛化有效性。

L_s6300 与 V27000 复用合法源 cache、物理 ID 与 sourcecontract（RX1/3/4/6/8，day1/2/3，6TX，split392005）。U 的 IQ、ID、TX 真值均不访问，目标/query 不访问。V 只前向观察，无梯度、优化器、EMA/BN/参数更新。训练中的 BF 三新模型继续独立运行，不热改。

## 固定测量与计量

每角色固定 B256，包括尾批；shuffle L2026101410/V2026101420，强视图噪声+dropout L2026101411/V2026101421，与既有 gate 审计相同。学生 identity train-mode 原 dropout+原 Gaussian .015，教师 actual ownEMA eval。identity 无 BN；前后逐值核对模型及 buffers 未变。数值 fullFP32，CPU2threads，CUDA 不可见。

原生 confidence RX/dayquantile .86、tau [.92,.97]、batch_neighbor window2/minconf .8、strong argmax agreement，domain gate false。复用原 gate 代码。先固定所有 masks/特征观测，再连接允许的 L/V observer labels。观察 CONFIDENCE_STRONG 与 FULL_OLD 两种固定 scope。

在 L 的每个同一 strong graph 上，测量 lambdaU .16 的 unsmoothed accepted pseudoCE、matched trueCE、正确与错误伪标签 CE 的实际参数导数。四项均用同批总接受数作分母；正确+错误梯度必须还原总 pseudoCE 梯度，不用错误子组独立分母放大贡献。对象是实际 t1PW3456、t2PW6912、SIG localprojection576。默认 W 作为当前常量，完全不拟合或累积 .grad。这里不是正常训练的独立 L128 CE 图，也不是熵项或 AdamW 累积贡献。

逐批保存原始四项导数，重新从 PT 计算摘要；全局按每批接受数加权求和/总接受数得到固定 frozen source risk 梯度。wrong_norm/total_norm 可能因抵消大于1，不称可相加的错误贡献百分比。没有接受或错误时报告0或null，不解释为 U 无错。

对 L/V 仅汇总 teacher confidence、top2 margin、actual joint160 EMAweak/studentstrong cosine 在正确/错误组的 count、mean 与固定 .05/.25/.5/.75/.95 分位数。不拟合分类器、判别函数或阈值。模型种子复用相同源物理样本，不冒充三份独立数据；只观察 final50，不外推早期训练。

## 执行与预算

Run 20261014-phase1-source-sigtime-pseudo-gradient-manysig-m3-r01，唯一 owner codex/root/source-pseudo-gradient-20261014。普通 N607 账户，CWD PROJECT/releases/cvs_source_pseudo_gradient_20261014_r01，入口 python -u -m experiments.cvs_source_pseudo_gradient.audit，本地发布入口 experiments.cvs_source_pseudo_gradient.publish --output local_artifacts/source_pseudo_gradient_delivery_r01。不可变 run PROJECT/runs/20261014-phase1-source-sigtime-pseudo-gradient-manysig-m3-r01，独占三 row 输出；全配置/权重路径见 experiment.json。

每 seed 两角色共66600 source packet forwards/262batch forwards；全三种子199800包/786batch。L25批×2scope×4向量×3参数=600原始向量/seed，三种子1800；不计作训练更新。所有 optimizer/head/model 更新为0。原始 PT/权重/日志/机器状态不进 Git，仅提交聚合结果与报告。

唯一独立 P0/P1 审查 PASS，限定范围未发现直接问题。局部检查已 PASS：7协议负例、同分母分解、matched trueCE、零接受、无错误null、实际网络三对象非零导数、joint160、参数/buffers/grad未变。发布只做白名单内一次归档SHA、本地/远端比对、远端编译和真实 checkpoint 零IQ no-query smoke，PASS 后自动运行，不额外设计审批/封印/重复数据验证。

停止规则：仅所属 run 的数据/来源/query/非有限/参数变更/输出碰撞/无法运行等系统技术失败，保留产物、写 failure、不自动重试。没有训练因此不新增目标测试；自动收回源结果并同 row 分析、登记、提交推送。BF 训练仍按原冻结方案自动 clean 预测及独立 truth-last 评分。

标准 N607 只读预检已于 2026-10-14 23:32 CST 通过；普通账户、项目路径和现有 GPU 状态已核实。本诊断只用 CPU，不分配 GPU。

完整结果见 [源域诊断分析](source_pseudo_gradient_analysis.md)。L 接受集合均无观察错误，错误梯度不可识别；V 的 joint160 一致性不支持简单可靠度门槛。没有新增目标分数。
