# PAMF 同记录星地视图与未知身份结构学习

上一轮源匿名邻接验证中，P1 互近邻同设备精度 1358/1674=81.1231%，P0/P1 top1 双向一致只有 25 对、22 对同设备，且某个旧类没有边；预登记 98% 和覆盖约束无解。本轮不放宽该规则、不读取 OE TX 或把邻接结果当作身份标签。

## 固定机制和两个对照

同一物理记录的 P0 星地 received 与 P1 冻结 DURA 恢复结果属于同一发射事件。这是本轮唯一的 OE 身份正对；不同 OE 记录之间不猜同类，不把所有其他 OE 视为不同设备。源 L 仍有合法六类标签，可构造同时跨 RX 和 day 的同设备正对。

精确复用完整四贡献与常量条件 C0 结构：时频、带原增益 PA、带原增益 reference、已限幅 SIG 各160，同次前向 concatenate640→64→64/LN/unit；原默认六类头及有界零初始化身份残差保留。两组从原 Tri-SIG-TIME SSL F0 E50/11100 独立初始化，不继续训练旧 C0/C1/M，不按目标成绩挑父模型。常量 conditioner 两组相同，不以新参考条件输入解释本轮效应。

临时 hook 只捕获同次 projector 最后 LayerNorm 的单位化前 raw64，不新增参数、不重复前向；标准差与协方差目标作用于 raw64，不能作用于受单位范数限制的 open64。模型实际活动参数315106、新增82113、native容器982835，未使用域参数667729，均与两组旧 C0 结构相同。

| 行 | 完整 L/KD/五类集合成员 | L 跨 RX/day 监督对比 | L 同记录 P0/P1 | OE 同记录 P0/P1 |
| --- | --- | --- | --- | --- |
| P_L 控制 | 相同 | 权重0.1 | 权重0.1 | 仅诊断，权重0 |
| P_LOE 唯一主 | 相同 | 权重0.1 | 权重0.1 | 权重0.1 |

两行都执行相同 P0/P1 前向和 OE 诊断，唯一实验差异是 OE 视图正则是否进入梯度。P_L 仍在原成员任务中使用 OE outside-six 监督，不能称无 OE 控制。L 跨域对比的温度固定0.1；正对只含同标签且 RX/day 均不同，分母包含这些正对及异标签，不把同类非跨域记录误作负样本。监督对象只为 L，不解封 OE/原 U_s/目标标签。

同记录 raw64 正则：`25*MSE(z0,z1)+25*std_penalty+cov_penalty`。std_penalty 为两视图 `relu(1-sqrt(sample_var+1e-4))` 的均值；cov_penalty 为两视图样本协方差非对角元素平方和/64。它是身份一致性与防塌缩假设，不保证剥离接收机或真实未知设备可分。不同于既有 raw fused160/P1 phase-gain/VICReg+uniform-six OE 负结果，本轮作用于开放 raw64、使用同记录真实 received/恢复视图，并保留原成员任务、额外合法跨 RX 身份对比；不把它改名为旧方法的成功。

## 训练和数据预算

每行2220 fresh CENTER_UNITS AdamW更新，lr1e-4、原weight_decay、物理clip5，累计13320。新模块/model seed2026092701，source sampler2026103017、slot2026103018，与旧M五类集合采样相同。完整原生 L128 六前向768行、CRIC0.1、ordinary/raw KL、As CE且As KL0、LS0.01保持；As64重放固定 P1，不生成新频道。另 raw P1 L128 KD(T2,w1)、support96/query96/OE64成员前向256行。新增 P0 为 native_L 前64及同 OE64，共128行；两组每步1280 F行、每组2841600。共有已知/未知视图对各64/步。真实原生Sinc/骨干/默认头联合更新，encoder/dropout/BN eval不切梯度、buffers不更新。

已知保护 anchor=完整L+KD；待保护附加目标=成员0.1+跨RX L对比0.1+L视图正则0.1+OE视图正则0或0.1。共享梯度冲突投影与范数上限沿旧实现，关系评分器独立。记录 OE 视图目标对 raw64 的梯度，区分实际施加与仅诊断；参数变化不能单凭weight_decay解释为有效使用。

仅训练源 L6300/OE6134，读取各自既有 G10 P0/P1，OE只有outside-six成员关系及RX/day，无真实TX/K。原Us56700只属于原SSL祖先，本增量不读。源接收机/日期、旧六类、F0和DURA祖先不变；源V27000仅冻结后known风险校准，source novel3150只冻结后独立评价。所有 Phase1 后新增输入均星地，不读clean或重新抽信道。重复曝光不增加物理记录；本轮记录实际采样到的唯一源索引数以区分池容量。

## 冻结预测和验收

每行独立固定2220终点后，L形成六原型、known V按原42组规则固定六个预测类别阈值，再完成 source novel3150 和 RX24-1 futureQ864全部预测，独立 CPU 才连接两truth。所有推理只用 P1，额外P0仅源训练；同P1原F0闭集为不可更换基准。target U384/正式RX24-18不读，没有适应、发现或匿名注册，AN/HOW N/A。开发Q和source novel历史暴露，不称新盲测。

唯一主 P_LOE::relation_max，不按控制或目标分数换主/选checkpoint/调阈值/重采样。source未知90不取消目标测试。主报逐TX宏 A_K、R_U、H，伴C_K和原F0closed−open分解；targets v3 CK≥90%、RU≥90%、原F0同输入闭集到AK损≤2pp，总体及预登记六scene。high郊区未知类缺覆盖，按原规则N/A，不能认证六景。完整无标签适应/未知发现/未来匿名注册仍ACTIVE。

本轮两组训练F共5683200真实行；eval每组新F37314、原F0864、G864真实行，总F5759556真实行。训练实际checkpoint烟测每组8合成F行（原native/open4+新physical4），eval每组新F2/原F02/G2，合F24/G4合成行；G真实1728。source池42584加Q864为43448既有物理记录。活动F+G582057、完整eval含native容器+原F0参考共1482779参数；没有剥离容器或B1时延证据。

## 发布边界

默认N607，最多两row并发、每卡最多新增2own，每次实际free≥8192MB；foreign不整卡排除、不停止/修改/重启。一个不可覆盖run/root、实际代码Git提交、聚焦负测与首步真实checkpoint noquery smoke、唯一直接P0/P1审查、最小预登记、一次preflight/单archive比较/remote compile、独立PID/CWD/argv/GPU/log增长、全部预测固定后的独立评分。健康行不中断，技术错误保留无自动重试；科学负结果不触发停/热改。用户已授权自主设计和分工，白名单外审批/签名/固定评审链NONBLOCKING。
