# 已冻结RX分类器：零训练源读出检验

原RX_HEAD固定混合62.4000%，相对Random61.4222%高0.9778pp且5/5正，相对native3仅+0.6222pp、2/5正，未过原源条件，没有target。这个源证据支持区分共享骨干与分类头平均的贡献，仍不能称新方法有效。原run及失败判定保留。

本run复用同一10个fixedfinal30父模型，0训练/优化/校准，逐个严格核对真实source_contract、物理fitL IDs排除heldRX、config、epoch30/6660、无未知继承/U/target/CFO、冻态不变后strictload。源数据VALIDATED_ONCE及原角色复用，fitL5040、heldV5400；只L用于匹配smoke、V用于冻态预测，U及目标关闭。

三读出在结果前固定：GLOBAL_ONLY只originalglobal softmaxT1；ENTROPY_GATE为0.5global+0.5专家概率按softmax(-4H)加权（H为每packet每专家六类entropy）；CONFIDENT_HEAD为0.5global+0.5最大maxprob专家，tie固定最小index。全部六类同规则、逐packet独立，不输入RX、querytruth/role/metadata、批次真实类别数、配额或全局重排；不改父模型。读出再与既有native3源冻结概率固定0.5/0.5，ID对齐后先保存预测及混合，再source评分。

3readouts×5RX、每row匹配RX与Random两parent，162000源读出概率及162000混合概率；完整15row后每读出严格资格为RX混合source均值>=native3+1pp及Random+0.5pp、各4/5RX非负、worst均非劣1pp；仅合格读出取RX source均值最高，完全并列GLOBAL/entropy/confident顺序。未合格不访问目标。合格才按此前已固定fullsource5头、随机native30、三primaryseed2701/02/03+各自合法native3复用进行正式测试；动态专家数使用同一读出数学公式、无新fit超参。逐row自动clean+六practical，全7views各168000完成后独立truth-last；84尚待完整三seed实际证明，现有基准已暴露，不称新盲测/卫星部署。

不隐藏历史训练：parent10模型66600更新/300轮及native3历史另计，本run只加预测/评分时间。一候选family P0/P1、一直接GPU预留issue定点复核、聚焦权限负测与匹配真实checkpoint noquerysmoke、实际code/config Git提交、单不可覆盖run/release/output、N607preflight/单archiveSHA/compile/PID-CWD-argv-GPU-loggrowth；初始GPU环境独立worker复用既有容量保护，<=3总进程含其他owner/free>=12000MB。普通账号唯一owner，按技术/权限/输出碰撞失败保留所属产物，不根据性能停或重跑。矩阵/命令/环境/CWD/路径/stoprule见experiment.json，无额外gate。
