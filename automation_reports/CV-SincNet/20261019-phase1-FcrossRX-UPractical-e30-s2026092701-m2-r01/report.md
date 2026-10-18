# U 学生 practical 视图预算重分配：E30 最小匹配实验

状态：PLANNED，尚未发布，无新性能结果。run 20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m2-r01。唯一 owner codex/root/practical-UPractical。

上一轮 nominal 质量分层已完成，源 hard 组 S−I −0.526661 pp，固定方案关闭；另一对话正在运行有界困难风险 M3，本轮不叠加其权重或 bank。历史 Uurban39 是额外 U forward＋qKL；P67 是 epochbank 资格＋Gaussian strong；P68 是第二 L As。本次精确组合未见已实训证据。

唯一假设：保持 L 星地训练和总学生前向预算不变，允许已由 raw EMA 教师赋予目标的更多 U 物理样本，以完整 practical 观测参与现有学生目标，能否获得超出原 As 的 urban 收益并控制 clean 代价。

|逻辑条件|训练来源／唯一差异|
|---|---|
|NOAS30|复用原 P43 B00 ownNoAs20＋10 完整冻结读出；独立 NoAs 训练轨迹|
|AS30|复用原 P43 B01 ownAs20＋10 完整冻结读出|
|MATCHED_NOAS30／N01|原 B00 ownNoAs20＋10；共同教师资格，原 NoAs L6＋Gaussian U；严格阶段 clean 控制|
|UTEACHER30／C01|原 B01 ownAs20＋10；三新行共同取消 student strong agreement，其余原 clean teacher 资格和 U 目标保留；学生 Gaussian std .015|
|UPRACTICAL30／S01|与 C01 同父、同数据、同更新／NN 预算；每 U batch 恰好四分之一槽替换为 complete practical(rawU)|

三新行各 total E30／6660 updates，其中本次 E10／2220；reset AdamW lr2e−4、wd1e−4、cosine10 min1e−6，既有 SIG_TIME_X3／统一 Sinc／真实 joint160／default cosine6×160 scale30 联合训练。N 严格加载原 P43 B00 ownNoAs20 student＋ownEMA，C/S 严格加载原 P43 B01 ownAs20 student＋ownEMA（reference commit 19aedc3604fed9c193237dd92366164178af0983），不采用目标选出的父状态。NOAS30 不能称为共享 As20 父的尾段对照。

数据权限：split392005，原全 L_s6300／U_s56700 TX hidden／V27000，源 RX1/3/4/6/8、day1/2/3、全6TX，IQ256、eq1、Fs25MHz；没有 RX 留出和 L_dev。保留 NoExplicitCFO 与必要内部补偿，不增加部署分支或另训头。复用原 VALIDATED_ONCE 数据 handles。

教师 clean 前向先产生停止梯度目标与 base_mask：按 RX/day domain confidence quantile .86、阈值 clip[.92,.97]；原 batch_neighbor(window2,minconf .8)，domain gate off。三行共同 pseudo_strong_agreement=False，最终资格完全来自该教师路径，不能因 urban 学生答错拒绝 CE。教师权重沿各自行程更新，因此资格公式匹配不等于两行未来选中集合完全相同。不叠加 epochbank 或新阈值。

学生先对整批调用原 Gaussian strong 一次，候选再将预定1/4槽覆盖为对相同 raw U 的完整 practical；不在 Gaussian 输出上再叠加 practical，不重抽另一包，不使用 L bank。B256 选64，末 batch124 选31，每尾段预计 U As141750 包。传播、噪声、硬件、质量、lock、补偿、AGC 保留完整内部依赖，U 用独立命名 RNG 和通道计数，不污染原 L 物理流。具体固定 selector 和命名流由实际 design/config 固定。

U CE 仍为 lambda .16，对 teacher-qualified 样本的 CE 求均值，分母 max(1,accepted_count)，不按 As 接受数另归一化，无 label smoothing。原 entropy .01 仍在全 U 槽取均值，包括未通过资格和候选 As；因此本次估计现有学生目标随观察类型改变的总效应，不冒称纯 CE 作用。U TX 真值完全不读，不报告虚构 pseudo accuracy。

L 路径按各自对应原 P43 配方保持6L／768 packets per update：N 的 selected64 仍是 raw、As=0；C/S 为原完整 As64。C/S 公式：.5RD ordinary128＋.5RD mixed128(ordinary64＋raw→As64)＋.25RD raw128＋.1 ORD128 CROSS_RX，名义 L 系数1.25，As .25，原 LS .01、sameIQ RDrop、SupCon及父 epoch13普通增强实际配置不变。不减 clean 份额，不加新损失。

实际矩阵为 N/C/S 三新行，原runID含m2保留为不可变初始名称，alias actual_m3；实际 experiment.json.rows 为矩阵证据。发布前加入 N 是必要匹配控制：历史 NOAS30 仍保留 student strong-agreement，不能作为同 U 规则的严格阶段控制。旧 NOAS30／AS30 只作历史参照，不影响新配方。

固定 source-only 判定：选最后一轮 FINAL30，不依据中间波动换 checkpoint。S−C 的 V1800 two-draw mid/low 等场景准确率均值至少＋0.5 pp，且 fullV clean S−MATCHED_NOAS30（新 N01） 至少−0.1 pp，才有未来 E50／扩种子资格。失败关闭本固定1/4配方，不扫比例／门槛／种子／父状态救跑。源分层只描述学习情况，不用间隔、接受率、梯度代替识别收益。两 draw 不是两个模型种子。

每 row 冻结后自动 source readout＋全部7固定视图预测＋独立 truth-last scorer，不因源失败取消正式收尾。owner 先取 source 并保存判定，才读取新正式成绩。阶段正式 clean 相对同预算 MATCHED_NOAS30（新 N01） 下降≤0.1个绝对百分点；最终需可重复 clean 正收益及星地增量，单 seed／小于0.1 pp 微增不足证明。

正式测试保持原 received IQ／SNR／channel／168000物理样本：clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban，RX0/2/5/7/9/10/11、day0..3。各行全部视图 prediction 固定后 separate exec scorer 连接 truth；不做目标状态标签投射或选择性重跑。报告全部 Acc／macroF1／worstRX／urbanmean／RXdayTX／修复新增／suburban自身clean差及10配对。五逻辑条件不是2×2因子，不计算虚构交互。

成本：三尾段共30modelE／6660更新，学生 U 和教师 U 各每行567000样本前向，L每行1704960；源读出12600／行，正式1176002／行，native validation／smoke以实际计数。两个原20父各按实际复用计算，C/S共享As20父，旧冻结对照不重复收费。候选额外141750次完整U模拟另计调用、包数、perf_counter墙钟、复制／记录成本；NN不增不等于计算零增。记录资格、pseudo类别×RX/day（无truth）、原门控反事实只以已有logits描述、输入路由、CE分母、entropy和完整训练日志。

环境：N607既有CVS-RFFI/PyTorch fullFP32，CPU2，普通非管理员账户。GPU动态队列，每GPU全部owner compute PIDs≤3、free≥12000MB，不干预健康旧任务。实际代码以唯一 Git commit／release_commit.txt 固定；发布后在launch_verification记录实际commit、进程与命令。CWD PROJECT/releases/cvs_FcrossRX_UPractical_E30_20261019_r01，精确启动 CVS-RFFI/python -u experiments/cvs_u_practical30/runner.py，row 命令见 experiment.json。runtime私有参数不进Git。

不可变输出 PROJECT/runs/20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m2-r01，每row独占 source／source_qualification／prediction／scoring／日志；源码包 PROJECT/releases/cvs_FcrossRX_UPractical_E30_20261019_r01。预期artifact见experiment.json。一次普通preflight、一次release archive比较、一次remote compile、首步真实父no-query与synthetic实际hook smoke PASS后直接训练；唯一独立P0/P1限使本次运行错误/非法/无法启动/无法合法预测的问题。

停止规则：只停止精确run-owned的协议/query越界、错误数据或父契约、默认头/数值异常、输出碰撞、无法合法prediction闭合等系统技术失败；保留全部产物，先核实PID/CWD/argv，不盲目重启。科学负结果不触发中途停止，不热改其他健康实验。其余旧契约附加seal/hash/审批均NONBLOCKING／REJECTED_EXTRA_GATE。

源 panel 复用既有合法 V1800×2 draws，保存的载频2.45GHz与训练／正式2.462GHz差异沿用并披露，不将source分层当正式物理状态或clean精确预测。各候选和控制使用同一面板。

冻结烟测预算为2356 NN：两个唯一真实父各2个零输入预测，共4；三行实际L路径各768；三行原生U教师／学生各8，共48。烟测只用合成IQ，无真实IQ／query／optimizer更新，合成传播128 L＋2 U另计。训练与读出三行预计13054692 NN，连同成功烟测预计13057048，以实际counter为准，技术失败另计。
