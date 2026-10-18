# P73：完整 practical 的两随机子网 CE 聚合

状态：VERIFIED_RUNNING，两个新E10尾段各已完成4轮；尚无冻结后新性能结果。起点99e0030044cebc88568446a7418eb24302640ad1，实际代码/config提交cb5d8abdd7159ca5a3dad7bbf62f6b37a4c9d6fd，推送独立读回VERIFIED。逐row矩阵仅维护于experiment.json。

上一轮P72同父E30完整训练与all7已完成，源两个条件FAIL，新增raw-As共享mask关闭。P70/P71固定质量组/有界风险、既有As时机/权重/跨view对比/状态保留均有实际结果，不重新命名原样复试。另一对话“审计并设计 CVS 联合增强”独立三行UPractical已完成E30训练、正在源后正式收尾，本侧不重复、不干扰。

唯一假设：当前同一As IQ的两个独立Dropout输出分别计算CE后平均，要求每个随机子网都给出足够标签概率。固定两次前向，改为先平均softmax概率再CE，可能减少偶发子网失效的过度惩罚。它只改变As训练风险的聚合语义，未增加身份/物理信息，也不保证收益。数学为mean[CE(p0,q),CE(p1,q)]→CE((p0+p1)/2,q)，q仍原LS=.01。实现须以logaddexp(log_softmax(z0),log_softmax(z1))−log2稳定计算；不是平均logits、推理集成、CE降权/T2、不同完整信道、取消KL或修改mask。近期代码与登记窄去重未见该精确聚合已实训。

两个新行严格同一合法原P43 scratchAs20学生/ownEMA独立分叉，重置AdamW lr2e-4/wd1e-4/cos10/CENTER_UNITS，各追加10轮/2220更新，累计30轮/6660。C1为原meanCE，P1为probability-meanCE；原NoAs30及As30最终预测/评分只作冻结参照复用。NoAs来自自身NoAs20，不称四行同父；P1−C1是机制主效应。禁止加载P70/P71/P72已测试final作父模型。

固定F＋跨RX同TX SupCon、SIG_TIME_X3、joint160、默认cosine30头、900722参数/206state和正常联合更新，无新部署模块/分支/诊断头，保留NoExplicitCFO。L6300/U56700hidden/V27000，全部RX1/3/4/6/8、day1/2/3、split392005，不留出训练域或Ldev，不解封U真值。原U时间/置信度/学生strongagreement与熵目标、EMA保持，不叠加另一对话的门控改动。

完整原六L路径保持 .5 RDordinary128＋.5 RDmixed128＋.25 RDraw128＋.1 ORD128SupCon；总CE系数1.25，As CE=.25，As同IQ对称KL=.25。只修改mixed As64的CE聚合，普通64、primaryordinary128和raw128全部损失不变，既有六次L共768包网络前向、不重抽ordinary或As、不加新loss。两次Dropout保持原生独立与globalRNG调用序列；不使用P72私有mask。默认头仍只有一次既有学生推理。

每步记录同一实际logits上的两种CE、Jensen gap、两mask不一致率及既有质量4组，不额外网络前向或autograd；这些只是机制读数，不作为新的代理晋级门槛，也不能替代真实识别。梯度健康沿完整原step日志报告。

As复用P70之前生成的nominal142080观测，按原64*(step−1):64*step顺序各消费一次；逐step核对currentraw[smask]五字段物理key、y、domain、origin step/slot后消费，不加权/筛选/重画/重采。每包仍完整几何、遮挡、传播、噪声、接收状态及补偿残差，硬件按RX/day/eq session稳定，不由TX定义，训练/源验证/正式流分离，所有合法有标签As参与CE。

真实runtime固定pinned20261015_r02 experiments/adv3b02_xuc/code/leo_practical，不使用根code副本；打印并保存__file__、resolved config。输入float32 B×2×256/fs25MHz、原逐包RMS center=false之后从raw生成As；residual/post_sync/EQ关闭G=1/reference_snr_db30/reflect，接收质量适配/相位跟踪/IQ补偿/AGC保留。trainfc2.462GHz、固定正式fc2.45GHz沿原条件。正式received IQ、信道/SNR/困难样本不变；新增噪声比不是总SNR。

源判定仅两项：P1全V27000 clean−匹配NoAs30≥−.1绝对百分点；固定1800×2draw pooled midlow P1−C1≥+.5pp。任一FAIL关闭固定聚合候选，不扫平滑/温度/比例/随机种子，不扩E50或seed。sourceFAIL仍各行自动clean＋六practical预测，各行7预测固定后独立truth-last评分；owner先保存源决定再读正式分数，正式结果不调参/晋级/选择性重跑。source2draw不能称两个modelseed。阶段目标只clean基本不降，最终repeatablecleanpositive/lowurban>60未完成。

新增成本预计两个E10/4440更新：每行native3162962+source12600+formal1176002，加共享真实父student/ownEMA无query smoke4，总8703132新样本前向。父As20成本4797722NN/1810.249311秒复用、独立列明，不隐藏为免费或新增训练。池历史生成142080次/287.656秒已发生，新simulation0；CPU分析/本地toy无实际CVSNN。实际训练、池加载、source、formal及analysis墙钟分别记，不把NN相同称墙钟相同。

普通非特权N607 CVS-RFFI Python3.10.19/Torch2.1.0+cu121/NumPy2.2.5、FP32/CPU2。CWD=PROJECT/releases/cvs_FcrossRX_AsMarginal_E30_20261019_r01；命令CVS-RFFI/python -u experiments/cvs_as_marginal73/runner.py。immutable output=PROJECT/runs/20261019-phase1-FcrossRX-AsMarginal-e30-s2026092701-m2-r01，唯一owner codex/root/as-marginal73，每GPU最多2总进程含其他owner、free≥12000MB。先一次真实CP noquery smoke，PASS立即继续。

技术停止仅当前行输入/父源/物理配对/非有限/错误聚合或KL/调用或预算不符/输出冲突/非法prediction等确定错误；保留partial、不自动重试，不按低分停，不停止健康同伴或其他owner。执行最小流程的一次本地代码commit、聚焦负测、一次独立P0/P1、一次preflight/archive单SHA比较/remotecompile、启动后PID/CWD/argv/GPU/log增长、独立评分，不加审批/receipt/hash链。

预期产物：真实CPsmoke、resolvedconfig/模块/数据角色/严格初始化、完整step/epoch/marginal_steps/ordered_steps、frozenCP、固定源面板与先行决定、all7prediction与独立scores、逐row/seed accuracy宏F1最差RX/urban均值/配对净纠错、源状态质量/normalizedmargin/固定As20clean正确集合、覆盖/成本、训练和机制曲线PNG/PDF。真实IQ/权重/log/原始预测不入Git；正式报告保留全部负结果和缺项，预算耗尽不冒充收敛。


发布前实际验证：20协议负例、6概率聚合CPU检查、1复用实际P70hook的六前向接线toy、8CPU分析检查、21新增py语法PASS；一次独立P0/P1审查PASS，无未解决直接P0/P1。本地实际CVS NN/IQ/CP/信道模拟均0，真实父CP smoke4和GPU训练待launcher首步，无性能结果。


实际发布收尾：归档一次本地/远端比较及24py编译VERIFIED，真实父student/ownEMA严格noquery smoke4NN PASS。C1/P1在GPU0/1正常运行，PID/CWD/argv/GPU/log增长独立核实，当前各完成4/10新增epoch（累计E24），没有技术failure。实际模块、L6300/U56700hidden/V27000、parent、CE聚合模式与原AsCE/KL各.25已读回。launch_verification.json含聚合证据，原进程/配置读数保留local_artifacts/cvs_FcrossRX_AsMarginal_E30_20261019_r01，未访问正式性能；源决定与all7评分仍待训练冻结自动收尾，不宣称已完成实验或联合目标。
