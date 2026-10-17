# 原接收先验的虚拟会话覆盖：全源 E20＋10

状态：PLANNED／未发布。唯一假设是：完整 practical 训练只复用 15 个模拟接收会话，增加同一原先验下的会话覆盖能改善 urban 身份学习。当前只由代码证实“固定15状态”，没有证实它是性能瓶颈。本实验直接训练和评分，不把模拟检查或源统计当作身份收益。

## 单变量与匹配参照

原 ordinaryAs source-final20 学生＋自身EMA严格加载，原source固定final选择，无未知上游或target选模；与既有 nominal E30 FULL_AS 共用同一个父。追加10轮／2220更新，累计E30／6660更新；重置AdamW lr.0002/wd.0001、cos10，与控制一致，不称scratch30或精确optimizerresume。父来源与完整角色匹配复用既有证明，输入未变，不重新构造或验证数据。

F＋跨RX同TX／三路SIG_TIME／统一Sinc／joint160／原默认cos头与骨干联合训练。全部源RX1/3/4/6/8、day1/2/3、L6300／U56700隐藏TX／V27000、split392005保持，无训练RX留出，不换5580/720契约。原U、教师、普通增强父实际E13冻结、SupCon累计E21..30=.1、4L前向、.5ordinaryRD+.5AsRD+.1SupCon均不改；每对RDrop复用同一个已生成IQ，As从raw-before-ordinary产生，全部有标签As监督。没有新loss／分支／头拟合／U规则／额外NN。

仅把原模拟器的完整Receiver抽样会话键改为virtual/e{21..30}/rx:day:eq，共150个；每个虚拟会话轮内稳定，不按TX／sig／标签设硬件。直接调用原receiver_for_session和原receiverseed；真实幅度范围±.2dB、正交误差±1°、RX oscillator std200Hz、估计误差std.05dB/.2°全部继承既有先验。传播／遮挡／几何／H／噪声seed与namespace不变，完整原residual／post_sync／NoEQ／referenceSNR30／reflect只执行一次，接收质量／补偿／AGC依赖保留，NoExplicitCFO及内部必要补偿保持。

真实记录的源RX与IQ不替换、不反演TX硬件、不新增PA/DAC/量化或剪裁。该操作只是 A_s 的模拟接收覆盖变化，不能据此称A_g有效或真实接收机替换。旧“校准刷新”只换估计误差且固定15组true硬件；其失败保持。本次完整原Receiver先验变化是另一个算子，不按旧目标差值选新幅度或刷新间隔。

## 固定判定与测试

唯一模型 RX_POPULATION-s2026092701。固定累计final30后自动clean＋六practical，各168000包；同row全部七视图prediction/logp固定后独立CPU truth-last，用CP内默认头直接判决。正式received IQ／测试SNR／信道与困难样本不改，既有nominal控制只复用产物，不重训或重测。报告全部RX/TX/day、macroF1／worstRX、修复／新增错误、同包clean与场景联合正确和各suburban绝对值／自身clean差。

固定反证条件：low urban相对nominal至少+.5pp，clean与mid urban不降，每suburban不低于nominal .25pp。任一失败关闭固定配方，不改幅度／会话键／间隔／seed／父／预算救跑，不自动扩seed。结果只作固定单seed结论，不能用于目标驱动候选排序／选模／晋级／选择性重跑。另完整报告clean相对匹配B00净正、low>60%、各suburban差≤2pp探索条件；不得降低clean缩差。此单A_s隔离不构成新的四因子或联合成功。

## 执行与产物

- run：`20261017-phase1-F-crossRX-RXpopulation-e20plus10-s2026092701-m1-r01`；owner：codex/root/F-RXpopulation-E30。
- CWD：PROJECT/releases/cvs_F_RXpopulation_E30_20261017_r01；命令：CVS-RFFI/bin/python -u experiments/cvs_rx_population30/runner.py。N607普通账户、CPU2、按所有owner每GPU最多3实验且空闲≥12GB选1GPU。
- 不可覆盖输出：PROJECT/runs/20261017-phase1-F-crossRX-RXpopulation-e20plus10-s2026092701-m1-r01；登记与配置见[experiment.json](experiment.json)和configs/cvs_RXpopulation_e30.json。实际Git HEAD作为代码身份，发布archive内release_commit.txt记录，一次archive本地/远端SHA比较和remotecompile，不加审批或hash链。
- 训练2220新更新、L1136640NN／U567000原规则、As284160；150会话draw，0新NN／额外滤波／完整信道预跑。父20／4440复用成本、教师／源eval／预测／smoke／墙钟实际分别记录，不称零物理成本。
- 预期真实CP noquery smoke、resolved／actualargs／actualmodules／ordinary、全量L/U序列与每步目标、finalCP／source冻结、七视图prediction／score、coverage、成本、曲线和同row报告。
- 仅源/协议不符、非有限、输出冲突、启动/系统技术失败停止run-owned流程并保留，不重启旧健康任务，不自动重试。

## 已完成的有限验证

本地编译PASS，36协议负例、120同latent配对（120原＋240候选/重复完整处理）、LOS/PARTIAL/BLOCKED覆盖、150虚拟会话重复性及原生调用精确回放PASS；0真实IQ/CP/NN/更新。合成最大相对IQ变化1.57454%，只用于说明算子实际生效，不是标签保持或识别提升证明。一次独立P0/P1审查PASS，0 P0/P1；N607只读预检PASS。尚未发布、训练或取得正式结果。
