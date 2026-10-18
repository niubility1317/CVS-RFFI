# P76：共享传播与环境内 TX 判别的匹配四格 E30

当前状态：PLANNED；没有启动真实训练，没有新性能结果。实际代码 commit 在发布前固定于 Git 和 release_commit.txt，起点为 3e82dbc9d9db95d8370028e6d6f627e89af9c4ce。

最有价值的基础仍是完整 practical residual 与 ordinary/raw 监督保留。P75 同父纯噪声重抽未过源判据，固定候选关闭；不据其正式微涨扩 E50。P20 已检验公共传播、P41/SAT192 与 mixedContrast 已检验 As 对比，均不能重新命名后原样启动。它们没有检验本轮的环境条件对比交互：P20 对比只看 ordinary；P41 混合集合或 mixed128 没有公共 H，也未限定同环境负对。

唯一假设：让已有 As 表示在相同外生传播背景下区分不同 TX，可比单独共享传播或单独 As 对比带来更多身份收益。不是复制全部 clean 特征，不要求所有门控一致，不新增部署头。

| 行 | 每组共同场景 | 共享 H | 额外 As 对比 |
|---|---|---|---|
| C00 | 是 | 否 | 关 |
| C10 | 是 | 是 | 关 |
| C01 | 是 | 否 | 开，0.05 |
| C11（唯一候选） | 是 | 是 | 开，0.05 |

四行从同一个 SAT18 AS_CE20 student/ownEMA 独立分叉，追加 E10，总 E30／6660 更新。共同父路径 PROJECT/runs/20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01/AS_CE-s2026092701/stage20/final_ssdg.pth。其 source-only／实际全源角色和固定 FINAL20 已验证，来源与 P75 相同，严格复用结论，不重新验证未变数据。父前20轮 As KL 已关闭，不是 P43 原 As20。P43 NoAs30／As30 仅作同预算整配方参照，旧预测评分复用，不冒称同 As 父的 NoAs 尾段。

实际固定 F＋跨 RX 同 TX、SIG_TIME_X3、统一 Sinc、joint160、默认 cosine30 头，正常联合更新。L6300/U56700隐藏TX/V27000，RX1/3/4/6/8，day1/2/3，6TX、split392005，无训练RX留出或L_dev。IQ2×256／fs25MHz、原RMS center=false，原normalize_input后完整residual。训练fc2.462GHz；固定正式received IQ／fc2.45GHz和六practical配置不变。实际导入 pinned adv3b02_xuc/code/leo_practical，不按顶层code猜实现。residual/post_sync/EQoff意味着G=1；reference_snr_db30、quality adaptation／phase tracking／IQ compensation／AGC启用、reflect边界保留，不反演录音未知TX效应。

group=(RX,day,eq)，不读TX，按cumulative epoch/step刷新；四格共用相同场景先验及group场景流。H因子只移植P20已验证的geometry/probs/state/z/ray_hz/ray_coef/f_doppler，重算f_total/f_hat。每包noise/phase/sync/tracking/equalizer原流和稳定session RX硬件保留，各自原生质量、补偿、AGC；不复制首包接收状态。白输入质量代理可能自然给同H相同lock，但实际波形功率／AGC仍不同。每格均构造personal64＋每group一个不process的environment，process64次，确保不把构造次数等同波形数。第二As IQ原样复制，只改变dropout，不再次加信道。

原 sixL768、ordinary/raw CE与KL、ORD128 SupCon实际0.1、U batch_neighbor规则保持；As CE总0.25（两pass各0.125）、总L CE1.25、As KL关。ON仅额外0.05 As对比，复用已有两个mixed forward的time_down/t2.pool maps，各自As64的时间均值/标准差、detached集合标准化、L2、温度0.2。正对同TX不同RX（day不限），负对不同TX且同RX/day/eq；同TX同RX和其他组负对不进这一个辅助分母。两层×两pass取均值，无新增NN／参数／head。无合法正负对只令该辅助项为0，不删除任何合法As CE。真实标签只来自L hook.labels；sig_i是包索引，不能当TX；U真值隐藏。

源晋级预先固定，只有C11为候选：全V clean相对原NoAs下降不超过0.1绝对百分点；固定2draw×mid/low均值相对max(C00,C10,C01,原As)至少+0.5百分点；交互C11−C10−C01+C00至少+0.5百分点。任何失败均关闭此固定方案，不扫权重、seed、分组条件或用正式分数救回。全部四行仍各自冻结后自动clean＋六practical预测，再独立exec truth-last评分。source decision先保存并由owner观察，再读取新正式指标；不能以未完成peer判SOURCE_PARTIAL。只有源PASS才可另行执行已授权总E50且需匹配控制，不自动扩seed。

新训练成本：4×E10=40新model epochs、8880更新；共同父20/4440只复用一次。各行预计native NN3162962、source12600、formal1176002；共享真实父权重/合成图smoke3076，共预计17409332新样本前向，本地NN0。实际教师／验证／更新／模拟构造／process／墙钟从完整日志与物理记录分别报告，预算结束不等于收敛。

执行环境：N607普通账号现有CVS-RFFI Python3.10/Torch2.1/NumPy2.2，fullFP32 CENTER_UNITS、CPU2；CWD PROJECT/releases/cvs_FcrossRX_propContrast_E30_20261019_r01。唯一owner=codex/root/prop-contrast76；容量锁下每GPU最多2个全owner实验进程，launch前一次preflight选空闲容量。不可覆盖输出 PROJECT/runs/20261019-phase1-FcrossRX-propContrast-e30-s2026092701-m4-r01，逐row独占source/prediction/scoring；旧实验和另一对话健康任务保持。

启动命令：

    CVS-RFFI/python -u experiments/cvs_prop_contrast76/runner.py

首步一次真实父权重/ownEMA＋四格synthetic六L graph smoke，PASS自动继续，不增加smoke许可。代码/配置Git固定后一次独立P0/P1、一次release归档SHA传输比较、远端compile；实际PID/CWD/argv/GPU/日志增长读回。技术错误只停止/保留自己的错误row，其他健康row收尾，不自动重试。错误角色/来源/defaulthead、非有限、越界预算、输出碰撞、错误private流或不能产生合法预测视为技术失败，低性能不触发停止。

预期交付：source实际模块/args/配置/parent初始化、training_sequence、全部epoch/step loss、prop_contrast_steps及physics（含合法L标签和环境pair覆盖）、final/冻结、固定V分层、source_matrix_decision、逐row七环境predictions、独立scores、analysis_r01完整曲线/分层/交互/F1/最差RX/修复新增错误/成本/PNG+PDF。新正式数据不供训练或选模。技术缺项和科学未达目标分别报告。

另一对话splitAdam r02实际两As行已因相同component-gradient检查退出，NoAs完成，源矩阵未闭合；这是技术失败，不是科学负结果。本轮只读核实，没有修改或重启该owner产物。

实际发布代码 32355ff0e6d73a7305e2fb8ce332651c9d1a7cbe，单archive传输和25文件远端compile VERIFIED，3076NN真实CP/四格图smoke PASS（source IQ0/target queryfalse/optimizer0）。合成标签与As parity相关，ON smoke额外As项为0，不据此称新aux已激活；实际source首批ON均61/64有效anchor、502正pair/238负pair，第1epoch平均加权As loss C01=0.0498677631、C11=0.0569158431，平均有效anchor62.225/64，确认真实训练使用新增项。OFF stats0为未计算而非无pair。四行GPU0/1/2/3的PID/CWD/argv/CUDA/实际GPU进程/日志增长已独立核实，均E21完成，角色EXACT_MATCH且target_contactfalse。尚无源域判据或正式性能结论。
