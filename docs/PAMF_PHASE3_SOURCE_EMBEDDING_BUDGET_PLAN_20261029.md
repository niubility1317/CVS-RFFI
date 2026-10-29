# Phase3 开放集特征训练预算对照

2026-10-29。独立候选 `20261029-phase3-source-embedding-budget-s2026092701-r01`。当前固定E51＋原4OE读出在原F参照known风险下未知宏拒识92.6667%，仍未通过95%与每RX约束。本轮检验222步源开放集训练是否不足；不将更长训练预先视为有效，也不按source novel选择中途checkpoint。

两行各自从原三视图完整六类PAMF E50严格加载并冷启动CENTER_UNITS AdamW，独立新增2220步、累计13320，E51至E60每轮222。禁止串接已看source分数的E51。constant lr1e-4、继承parent wd、betas.9/.999、eps1e-8、physical clip5。保留原native有标签KL／CRIC／As及六类默认头联合，CRIC.1、ordinary augmentation epoch13；As沿global11101..13320。supervised_only不是纯CE/PAMF-CE。U/teacher/EMA迭代关闭；V／novel／target不训练或早停。

uniform_oe仅加原source21公共池6134，domain iid均匀15再cell内iid记录，batch64、seed2026102907，λ.1 KL(uniform6||softmax6)。eval临时禁BN更新和dropout，但保留编码器与原身份头梯度；恢复训练mode和RNG，不改变匹配L随机流。每行L128×2220＝284160采样、1704960 L包视图、142080 As生成；OE另142080暴露。实际L2与OE2 Sinc／head梯度smoke无优化步且恢复state／buffer／RNG后立即训练。只保留固定末步checkpoint作为性能候选。

每行冻结后重新导出8原始分支及完整6类logits，L2smoke＋L6300／V27000／原4OE4200／novel3150＝40652新前向，不能复用旧模型表示。fit只由自己的L形成几何，consensus100D＋原4OE成员监督，linear101／MLP1633各固定200CPU步，普通L+OE均值方差、AdamW .001/.0001。source21训练OE与原4读出OE的累计标签成本分别记，未知TX私有身份不进入任何训练。

predict直接采用上一独立候选定义的原F6已知风险：总体及每sourceRX宏FRR≤5%、原F闭集宏准确率−新open宏准确率≤2pp，冻结每method最大合法单全局阈值，ties不拆；novel分数和标签不参与。阈值complete之后读取novel特征；九字段30150×7预测全部固定后独立truth-last评分。仅MLP可推进，完整两行后若双过优先supervised_only；95%未知（总体及每RX）、5%FRR、.5pp编码器损伤、2pp总损伤均保持。当前knownV与novel3重复source开发，不能当新独立确认或跨RX／星地／无标签适应／注册证据；正式RX24 U/Q仍不访问。

流水线trainGPU→exportGPU→fitCPU→predictCPU→scoreCPU，最多两行并行、每行一条GPUworker，合计每卡最多新增2。实际容量不足排队，不因foreign占用停止任何用户任务。immutable root、实际Git code/config、focused负测与真实checkpointsmoke、一次新候选P0/P1、一次N607 preflight／单归档比较／compile、实际PID/CWD/argv/GPU/log读回和全部评分按最小流程完成。只有技术错误允许精确本run停止；低拒识/旧类受损是科学负结果，完整报告、不改预算或选择性重跑。
