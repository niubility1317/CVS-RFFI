# 额外 PA 输入诊断后的独立训练因果检验

状态：DESIGN_NOT_RUN。此前整套 ordinary 的五fold规则FAIL保持不变，不回溯修改结果。新诊断在五个冻结N0头、held L6300上给出NO_PA−FULL准确率均值+0.600198pp、RAW正确集合保持率+1.126556pp，全部五RX为正；这只证明当前条件下的输入风险，不能证明训练收益。

现在只检验一个新候选：N2保留原ordinary全部其余设置，仅令实际native参数aug_enable_pa_normal=false。原p_pa和schedule保留，真正执行由no_pa控制；guard禁止simulate_pa调用。私有augmentation流每个call独立并恢复全局RNG，因此PA关闭不改变以后L/U采样或dropout随机流；实际全部L/U序列仍逐条比较。未知原TX硬件不反演，不引入其他PA/量化/剪裁或新标签特征。

首轮固定原顺序RX1，不按已知RX8负分换折，modelseed2026092701。一个新scratch总E20=13L+7U/4440updates，SIG_TIME/RAW_HZ/NoExplicitCFO/joint160/defaultcos、LS.01、AdamW/cos/RDrop/oldgate/ownEMA/U规则及原载入路径不变；默认头和骨干联合训练。只用RX3/4/6/8的L5040/U45360，seenV21600；RX1原V5400仅FINAL，L1260/U11340不训练。当前PA诊断曾只读heldL，不改变这些物理角色或将其转入训练。

复用已完成、同初始化规则/命名流/同预算的N0和N1源结果作为固定对照，不加载其权重。N2首先要求相对N0 held accuracy>=+1pp、macroF1不下降、seenV均值下降<=.5pp、健康与覆盖/实际L-U序列完整；还要求相对N1 accuracy和macroF1各不下降超过.5pp，以检验去PA是否保留旧有收益。只读诊断的差额不进入训练损失、强度、权重或晋级分数。

任何首轮条件失败即结束本NoPA训练候选，不改强度/seed/RX救援，不启动组合。首轮PASS仅允许另登记其余固定RX3/4/6/8各一个N2 scratchE20，再将所有五fold与已有对应N0比较：accuracy均值>=+1pp、>=4fold为正、worst>=−1pp、macroF1均值不降、每fold seenV均值损失<=.5pp。第一折N1保持条件不能替代最终五fold规则。

这不是按目标成绩删PA，也不把删除一个有害算子当作“新混合增强已成功”。本轮没有As/新G/对齐/头/分支或U机制变化，明确source-only，无目标预测。源证据成立后还需要独立固定full-source的Ag/As因子矩阵；每个正式训练row仍E20，自动原clean+6practical冻结预测和truth-last。lowurban>60%、clean净增和跨seed重现都尚未证明。

新成本为一个E20/4440更新/1136640 L网络前向，U/EMA/sourceeval/heldV5400/smoke2按实际计数。复用的RX1 N0/N1父对照各20epoch是历史40model-epoch成本，新模型不继承它们。既有200model-epoch五fold研发与207910前向PA诊断另外保留，不隐藏成本。

## 已实现并启动

20261016-phase1-ordinary-noPA-RX1-e20-s2026092701-m1-r01已实际发布运行VERIFIED；代码2b916b8892fce0dd1ae80ff9c8f1d649445b245b，新N2 scratchE20，真实CP noquery smoke、PID/CWD/argv/GPU和日志增长已独立核实。当前E2，尚无FINAL源成绩。一个限定P0/P1增量审查PASS，本地实际图512合成fwd/256aug计入成本，0local realIQ/CP/optimizer。只关闭PA的actualnativeflag已读回，U/defaulthead/预算/角色保持；完成后按原固定条件处理，不热改。
