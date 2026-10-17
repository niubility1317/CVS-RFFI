# 完整 clean 保留配方：独立 U 星地教学 E30

状态：LOCAL_VERIFIED，尚未启动。唯一假设：在完整原参考配方上，复用自身clean EMA的可信分布监督同包完整practical U，能优于同成本的额外raw U监督，并保住clean。P38固定源面板已显示teacher置信precision99.88%至100%、coverage89%至92.56%、low加权教学机会38%至40%；这些是源域资格，不是U真值精度或训练增益。旧SIG_TIME E50的SSL增量很小，不能直接复用其结论或叠加收益。

固定3组、1模型seed2026092701，全部独立从零开始，各自20+10累计E30/6660更新（总90modelE/19980更新）。B0为原参考L配方加一次rawU零损失学生前向；URAW以相同rawU学生前向计算额外软KL；UAS将该额外前向换成完整As(rawU)，其余规则相同。每个row的第一阶段13L+7U/cos20，第二阶段严格加载自身固定FINAL20及自身EMA、重置AdamW/cos10/0L+10U；原L在全部30epoch参与。未用旧目标暴露checkpoint初始化；真实旧E20只做兼容性smoke，随后丢弃。

L严格沿原完整参考：普通128两次、混合128两次；混合64ordinary+64完整As(raw)，有效CE监督ordinary.75/satellite.25；RDrop两次IQ相同、原SupCon仅普通pair。原ordinary冻结E13实际强度，保持原L信道逐包3场景名义采样与stream20261016421。F/SIG_TIME_X3/统一Sinc CENTER_25K与CENTER_UNITS/NoExplicitCFO/原joint160默认cos6×160 scale30，骨干和头联合训练；900722参数206state，不增加部署模块、头、诊断分类器、特征全对齐或新的Ag。

源角色精确复用split392005、7/63/30、L6300/U56700隐藏/V27000，RX1/3/4/6/8×day1/2/3×6TX；无训练RX留出、无L_dev、无V梯度。额外U复用原ownEMA out_w，0额外U教师前向；旧confidence/temporal/strong门控、旧U CE/entropy、EMA更新不变。原native在两个新钩子前均断言U标签=-1，新代码只读eq/rx/day/sig/base_index元数据，不读TX。

每个row全部17个Uepoch都执行额外963900学生包前向，含q=0包；B0附加loss为0。URAW/UAS用T1原scale30 softmax，q=clip((maxp_teacher-.95)/.05,0,1)，teacher stop-gradient，loss=batchmean(q KL)，系数0.16×min(1,(累计epoch−13)/5)。不按入选数重新归一化；不要求urban学生高置信；不从困难urban反向监督clean。该系数、ramp、tau不扫描。新增前向用private dropout stream20261018442+step并恢复CPU/CUDA RNG，三个row同流，避免改动后续原训练随机序列。

UAS调用原真实residual/post_sync/noEQ(G1)/reference30/reflect，fs25MHz/fc2.462GHz、质量适配/phase/IQ补偿和传播遮挡噪声依赖不拆散；新U stream20261018441与L/验证/测试分离，按epoch/step/physicalID刷新；硬件stream20261016423和原L一致，按RX/day/eq会话稳定，禁止TX定义session。dataset RMS后As自身unitRMS，无post-As RMS，不反演未知TX。保留三训练场景、全部LOS/PARTIAL/BLOCKED及锁定状态支持，不拒绝困难U。

首次U前，各row仅用固定合法V1800及既有2draw×3scene full源观测作一次自身teacher资格核查，14400NN（student12600/teacher1800），0更新/状态不变；全部条件固定为precision≥99%、coverage≥75%、每RXprecision≥98%、两draw mid/low q加权teacher正确/student错误机会≥10%。失败则本候选后续额外loss永久关闭，仍生成登记视图、完成预算及全部正式评价，不重新校准或救跑。后段只继承本row自身E20已有资格状态和EMA，不加载旧诊断教师。

最终源科学条件：UAS相对URAW在每个draw的mid/low均≥+1个百分点；fullVclean UAS≥B0−0.25个百分点且≥URAW−0.25个百分点，所有own teacher资格通过。未通过则不扩seed、不调整系数/扰动/阈值/epoch救跑。该门槛只评估固定C4，不代表整体Ag或联合目标完成。目标探索条件仍为clean净正收益、low>60%、suburban各场景距自身clean≤2个百分点，均不保证结果。

每row固定最终checkpoint后立即独立预测clean及既有全部6practical；每环境168000、全部7预测写完后才exec独立truth-last scorer。正式received IQ、测试SNR、信道、边界及样本集合不动。源矩阵判定仅dispatcher写，不读取formal；主agent在源判定固定前不读取正式成绩，不据目标结果选择或重跑。某row完成不等待整批才进测试。

执行CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01；环境：N607普通uid1000、CVS-RFFI Python/Torch2.6、FP32、CPU2threads。命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01/experiments/cvs_Uurban39/runner.py
```

唯一路径/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01，日志/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/logs及同名-launch，owner codex/root/Uurban39；每GPU最多3总进程且计入其他owner，不干预健康旧任务。停止仅限错误角色/来源、越权query/truth、输出碰撞、非有限loss/grad、无法执行或无法生成合法预测等系统技术失败；保留partial，不自动重启。不以低分或科学失败停止。

代码与配置提交身份待实际commit写入；Git先提交push并独立比远端，单release归档一次SHA比较及远端compile。launcher首步真实E20 checkpoint零query smoke，PASS立即继续real-F三组额外U合成梯度烟测；synthetic输出独立不占stage20/source路径。worker分配GPU后先作本次CUDA RNG问题定点回归，0NN/0更新。一次独立P0/P1审查发现3个未定义变量，均已定点修复并PASS；本地6项检查PASS/1项CUDA不可用SKIP、compilePASS。

成本计划：每row新30E，父成本是本row已计入的20E，不另借父预算；LNN3409920、原U学生与教师各963900、额外U学生963900、首次U源检查14400、冻结源读出12600、正式预测1176002及原native验证/初始smoke，实际NN和墙钟按完整日志逐阶段累计。总预算耗尽不等于收敛。预期产物见experiment.json：真实配置/导入文件、source角色、ownparent/EMA/optimizer、完整epoch/step/L-U-sequence、U信道元数据、覆盖、资格、默认头源预测、正式逐row/seed/RX/day/TX七环境scores、独立报告/分层图/训练曲线/成本/负结果。

## 实际发布状态

RUNNING／VERIFIED。执行代码提交 91bbfcdc026b48abfeaa576e14bc4b7f44d3d64a 已独立核实GitHub远端一致，一次release归档SHA比较与远端compile PASS。dispatcher PID776627；B0 PID776653/GPU0，URAW PID776734/GPU1，UAS PID776817/GPU2，实际CWD/argv/物理GPU/日志及自身run根均独立核对。3个GPU的私有RNG定点回归PASS。

真实旧E20兼容性CP零query smoke2NN PASS；真实F合成额外U路由48NN PASS，原Umask保持，CPU RNG恢复，URAW/UAS原默认头和Sinc梯度有限非零。合成使用独立目录，实际stage20/source未被占用；该证据不表示教学资格或训练性能通过。

最新源进度三个row均已完成epoch1／222更新，各自LNN113664、完整L As14208，额外U尚未启用（预计E14首次）。实际初始化均scratch、checkpoint_sources/ancestors为空、角色EXACT_MATCH；真实总参数900722、13L7U、原FP32策略；无traceback。worker日志从约6.4KB增长至约11.5KB，step与sequence文件持续增长。

真实导入leo_practical为不可变依赖release内experiments/adv3b02_xuc/code/leo_practical/channel.py与residual.py，并非仓库顶层code；DataAugmentation与SSDG.train_ssdg同属该adv3b02_xuc实现。具体__file__保存在各stage20/actual_modules.json。训练／测试未完成，首次自身teacher资格尚未执行，当前没有新的准确率、图或跨seed结论。

本地执行证据：E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/verified_running_r01.json 与 source_progress_r01.json。后续只读核对原PID/进度，健康任务继续；逐row冻结后自动进入既定七环境评价。

## 首次 U 与独立分析准备

三组首次自身teacher资格均PASS，源V覆盖B0=80.5556%、URAW/UAS=80.6667%，覆盖内precision均100%；该数字仅来自合法V1800，不是U真值精度。实际E14三个row均完成56700额外U学生前向，UAS完整As(U)也为56700，物理日志约28.9MB并继续增长。URAW首次U更新step2887的权重0.032、q均值0.61072、原mask未改、pseudo_truth_available=0、无非有限skip；仅说明机制实际执行，不说明识别改善。

独立分析程序analysis/cvs_Uurban_E30_results_20261018.py已实现并通过4项聚焦检查及编译。source模式只读已冻结源预测和完整日志，重算6660更新、原L目标分解、oldU掩码/hidden标志、额外963900U前向、Sinc物理步进、原始增强和采样序列及2draw×3scene源分层；final模式必须先有独立源判定，再检查全部七视图固定及独立评分后才连接truth。所有物理统计区分白功率代理、signal/added-noise与未知总SNR，U真值不解封。

该分析不修改正在运行的训练release，不增加NN或更新。门控逐包激活及逐分支梯度未在当前run保存，不从标量loss或准确率反推；L joint TX×state逐包侧表亦未保存，报告将如实区分已测的RX/day/scene/state聚合覆盖和缺项。源/正式图表尚未生成，不提供虚构图路径。

独立源分析等待进程已实际启动并核实：PID791020、start_ticks150569691、CUDA_VISIBLE_DEVICES空、CWD为单独analysis release，argv仅包含source模式及原三个worker PID，不传truth路径。分析代码身份8e9e15c093472bb722915117d3682745260d1394，独立GitHub读回一致、单analysis归档比较与远端compile PASS；训练release未修改。当前等待全3源冻结/源判定文件，stdout/stderr均0，尚未运行完整分析或生成图。最新已观测B0/URAW E19、UAS E16；UAS累计170100个完整额外U观测，3个原worker均活跃且无traceback。
