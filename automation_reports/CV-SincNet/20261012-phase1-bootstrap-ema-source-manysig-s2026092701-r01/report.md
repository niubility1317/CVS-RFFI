# 固定源集成向EMA移交：30轮源端预登记

本候选在目标评分之前决定探索，目标仍为clean三seed均值84%，未达成。比较 SELF_EMA 与 BOOTSTRAP_EMA，单学生seed2026092701、5个源RX整域留出，共10个从零模型，每个固定30x222=6660更新。原数据L_s20%不变，实际fitL5040，heldV5400，fitU45360；heldRX同时排除L、U、所有teacher输入。U原56700的完整truth-blind schema及物理ID先用现有契约核对，不解封任何U真值，不访问目标。原VALIDATED_ONCE数据复用，不增加数据验证链。

两arm用相同随机native、时频/PA/参考结构及cos头，无显式CFO身份标量；AdamW weight_decay0.0001、lr0.0002至0.000001的原cos30日程。前5轮纯L clean CE，label smoothing0；每个更新后原地EMA decay=min(0.99,1-1/(step+1))，参数和buffers写入保留版本更新以使推理cache失效。第6轮起U，固定confidence0.90、两个完整visit pass稳定、noise0.015、KL权重0.5在5轮线性升到满量。EMA保持eval、不求grad；软target detach。最终选择固定第30轮student，绝不按heldV挑EMA/epoch。没有星地训练增强、目标支持或测试时更新。

SELF仅当前学生EMA。BOOTSTRAP加载已经从零训练固定30轮的同fit4RX源CE成员0以及独立随机成员1/2，逐个检查实际source_contract、物理fitID、heldRX、config、完整预算和无未知继承/目标接触，再strictload。源teacher冻结对fitU生成softmax温度1等权软概率，不读取U类别。step1111 anchor权重0.8线性降至step4440的0，后10轮完全当前学生EMA。固定teacher不限制学生的后期类别判断。本handoff是待证假设，不先称有效；EMA依据[Mean Teacher原论文](https://arxiv.org/abs/1703.01780)。

每row固定后在heldV做一次预测/评分，保存ID/logits/混淆；预登记资格：BOOTSTRAP相对CE均值至少+1pp，相对SELF至少+0.5pp；各至少4/5RX非负；相对两者最差RX均不下降超过1pp。完整10row后才判定，不按途中source微小涨分改recipe。未合格不访问目标；合格则三primaryseed2026092701/02/03全源固定30轮，同机制各自teacher来源严格检查，逐row自动clean+六practical，全视图prediction固定后独立truth-last评分，目标不回流。源pilot显式只做源诊断。

成本：新增10学生300模型轮次/66600更新；BOOTSTRAP继承15原教师共450历史模型轮次且增加U缓存前向和EMA内存，完整披露。推理固定单学生177025参数；不把旧教师成本隐藏为从零全方法、不称同成本优越。一次代码/config Git提交、一次P0/P1、聚焦负测和一次真实checkpoint noquery smoke；一次不可覆盖run/release、N607preflight、归档SHA比对、remotecompile、启动PID/CWD/argv/GPU/loggrowth核对。每GPU最多3个总实验进程含其他owner，>=12000MB空闲。只按明确权限/输出碰撞/系统技术故障停止所属run，不按低分停。精确矩阵/命令/环境/CWD/路径/stoprule见experiment.json，无额外阻塞门槛。
