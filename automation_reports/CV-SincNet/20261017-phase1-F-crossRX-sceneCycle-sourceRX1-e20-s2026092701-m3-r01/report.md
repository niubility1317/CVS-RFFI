# 逐物理包六场景覆盖：三匹配 E20 预登记

状态：PLANNED，尚未启动。run：20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01；owner：codex/root/F-sceneCycle-E20-20261017。实际执行 commit 将由 release_commit.txt／resolved_config 固定。

本轮只检验 A_s 的场景组织，不是新 A_g。旧 sourceRX1／seed01 的全部4440步和568320次L场景选择已按真实seed/元数据/namespace精确复现。5040物理包均见过6scene，缺失为0；同包scene次数差中位11、最大27。因此不声称漏掉困难场景，只检验减少跨出现次数的不均衡能否提高识别。历史复算0IQ／CP／NN／更新。

固定底座 F＋跨RX同TX／三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认cosine30头正常联合训练／scratchE20／13L+7U。fitL5040/U45360/V21600，RX3/4/6/8、day1/2/3、原6TX；RX1的原L1260仅最终源读出。不混同完整L6300/Ltrain5580，U真值隐藏，旧ordinary/oldgate/ownEMA保持，不加载训练父模型或另训头。

B00/C_IID/B01三行相同初始化seed2026092701、20epoch/4440更新。共同完整raw.75+ordinary.25+原ordinary-only浅中层SupCon；额外.25R(raw/IID单As/逐包均衡单As)。每R为原平滑CE.01均值＋同一IQ两次网络随机前向的RDrop1KL。全部8L前向，总pairweight1.25，L均值按物理包/视图明确归一化；不稀释原base，不加loss/网络/部署分支，RDrop不称IQ增强。

C_IID逐点保持原fullpractical，按原train/epoch/step namespace均匀选1scene。B01仅把同一物理L第v次出现选为permutation[v%6]，每6次构造一个全部6scene的私有随机排列，seed=stable_seed(202610175001,physical_scene_cycle,pid,v//6)，pid为rx/day/eq/sig/base，不用TX/置信度/状态。保留原IID私有抽样工作；传播/noise/receiver session仍沿原seed和scene运行。同一scene同一packet同一步生成原完整IQ。每As都作用于raw而非ordinary；不连续施加两个信道，不删包，不强制锁定状态。全部合法L的As参与监督，无新增U或teacher视图。

每个row固定final20即自动做sourceRX1 clean+全部6practical，每scene1260，矩阵26460硬判决。source-eval委托原实现，不推进训练scheduler；正式target、receivedIQ、测试SNR、传播及困难样本保持。没有新target访问或按目标调参。

B01相对C_IID须：clean accuracy与macroF1不降、lowurban accuracy≥+1pp且macroF1不降、midurban不降、urban均值≥+1pp、每practical与seenV≥−.5pp。相对B00须：clean及F1不降、lowurban≥+5pp且F1不降、midurban不降、urban均值≥+3pp、每practical与seenV≥−.5pp。健康预算/原目标尺度和B01逐包次数差≤1同时核实；健康或覆盖均衡不替代识别。suburban绝对值和距自身clean逐一报告，≤2pp仍为探索目标，不把降低clean当作接近。

任一科学条件失败即关闭固定逐包六scene cycle；不换顺序seed/周期长度/边际概率/系数/RX/modelseed/预算救跑。通过也只支持独立源seed确认，不证明Ag、完整四格交互或联合目标。所有row正常完成E20，不因中间源成绩提前停止。

本地35输入/资格负例，4114合成NN/3合成联合更新/520合成fullpractical/1921调度选择PASS；0真实IQ/CP/target。B00损失/默认头梯度与原D_raw完全一致，随机As及sourceeval输入与原native逐点相同，手工物理路径4包逐点一致，同初始化/CPU/Python RNG/原Uargs一致，默认头更新且export160相同。一次独立P0/P1结果另附。

资源/成本：3scratchE20/60modelE/13320更新。每row4546560LNN含extra1136640；旧ordinary568320/row，C_IID/B01各568320fullAs，U/EMA/V/8820held与dispatcher真实CP2合成NN分别实际记录，生成/传输walltime单列，不拟合头。匹配旧B00sourcefold最终20权重仅作launcher第一步2合成包无query兼容smoke，不用于训练初始化；PASS立即继续。

命令：CVS-RFFI/bin/python -u experiments/cvs_scene_cycle20/runner.py。环境：普通N607已有CVS-RFFI Python，FP32 GPU worker、CPUdispatcher2threads/loader0。CWD：PROJECT/releases/cvs_F_sceneCycle_sourceRX1_e20_20261017_r01；不可覆盖output：PROJECT/runs/20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01；logs为其logs子目录。每GPU最多3个所有owner进程、最少12000MiBfree，使用共享capacity锁。一个owner；一次普通账户预检、一次archive传输对比及remotecompile。实际代码/配置只由一个Gitcommit标识，不新增文档/成员hash或审批。

技术停止仅限实际角色/来源/预算/输出/非有限值/无法启动/非法预测等系统问题：保留产物，取消pending，健康worker继续，不自动重试、不干预其他owner。预期产物为每row真实CP/config/roles/冻结/完整steps与L/U序列、逐step实际scene和visits、5040逐包scene计数、全部源七视图预测、独立全量复算/识别修复新错/差距/成本/报告/索引。原U真值和target始终不读。
