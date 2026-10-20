# 有界源频谱精度加权的复数恢复误差

状态：RUNNING / VERIFIED（controller资源等待，模型未开始训练）。用户目标仍为完整目标low urban≥70%、其余五practical接近原F clean85.113889%；“接近”暂按≤2个百分点，是Agent操作解释，目标未达成。本次矩阵、权重规则和预算在spatial/scratch新目标分数读取前固定。

源依据是已有完整source-only observability（86aa3179）：L6300/U56700在lag16/32/64/128的平均相干最高约.143，不能假定周期补全有效；oldUNet源low urban最低功率FFT四分位的有界分母复误差约.981。该旧模型诊断既不证明当前R03仍有同样缺陷，也不证明低能量频点就是TX指纹。当前pair_loss的logpower不保留相位，因此检验有限权重下的复误差分配，而非无限放大高频或依据目标涨跌调参。

|新固定行|复杂谱残差权重|对照|
|---|---|---|
|spectral_uniform|全部256频点为1|新匹配控制|
|spectral_whitened|同source L/U全局clean PSD的有界倒数，均值1|新加权比较|

两行wave均为IQNMSE+.1logpower+.1complexspectralNMSE；后者为orthonormal FFT(estimate-clean)的复数模方加权均值，除mean|complexclean|²（下界2e-4）。该下界与原I/Q均方1e-4一致，均匀权重由Parseval等于IQNMSE。whitened先对同一63000 source L/U clean全部样本平均FFT功率，再除全频平均功率，分母加.01；w=clip(lambda/(PSD+.01),.25,4)，lambda用固定64步二分使平均权重1。两行都在首个optimizer步骤前计算并固定全源谱，uniform输出全1；不按单包、类别、RX、环境或目标调整，0TX/V/target读取。

严格只加载原合法R03（methodb01b534f、coarse20/correction20）完整权重，2140840参数joint训练，不继承任何失败或refined/spatial/scratch权重。E10/2470、modelseed2026092701/dataseed2026101951、原R02 cache1..10、.15clean/.60residual/.25full、observed+clean、batch256物理、AdamW3e-5/warm247/cos3e-6、WD1e-4/clip1、FP32/noTF32保持。L63000/U567000优化曝光、1260000IQ；coarse/corr各累计30轮，60组件epoch、50串行继承pass。三原F50/defaulthead seeds01/02/03/method4df7b94d整体冻结，mean finalcos+.3T²KL(T2)/per_frame_cap不变，实际任务sum-gradient合并后除本批IQ数一次。无新身份/RX头，0分类头拟合、0L/U TX真值。

两个新行是匹配比较，仅谱权重不同。旧per_frame_cap没有额外.1complexspectral项，仅保留历史背景，不冒充匹配第三行。固定训练谱向量和PSD为私有训练损失状态，不进入恢复模型或推理；实际mean/min/max/sourcecount、逐轮spectral loss/原损失/cap/曝光/预算和成本保存。有限输出梯度限制不保证AdamW参数步或最终身份收益。

source V1800/固定F1/七视图只诊断，无资格/排序/早停/状态更新。两fixedfinalE10均自动完整168000 clean+六practical，三原默认头独立argmax，不投票。每行全部七视图IQ/ID/预测固定后，独立无CUDA CPU读取目标truth/配对clean，报告逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净纠错、距原clean差距与full256/prefix232/tail24 NMSE/logpower。推理只输入当前IQ，不读权重/PSD/F/clean/sourcecache、真实TX/RX/场景、其他帧或更新状态。源/波形检查不替代目标识别，不称三恢复器seed、新独立盲测、Phase2或在轨性能。

33项聚焦检查PASS，覆盖有界均值1/scale不变、source全人口权限、Parseval与能量floor/梯度、相位敏感、原R03 state/新checkpoint、谱状态和source行负例、actualhandle队列及truth-last前置拒绝。唯一一次独立P0/P1审查PASS，审查者独立复跑33项通过，未发现确定P0/P1。launcher首步实际R03/三F/四Ls no-query smoke，实际复谱loss和原头到IQ梯度、coarse/correction梯度、F冻结、完整初始state与checkpoint往返；0TX/V/target/optimizer，smoke权重只用四Ls、不是训练PSD；PASS立即资源队列。

新不可覆盖run/root/release，一次N607普通账号预检、单实际代码配置commit、一次归档本地/远端SHA比较和远端compile。CPUcontroller等spatial根终态且旧owned source/target真实handles退出，才在GPU6/7创建新worker，不只凭marker、不读分数放行；scratchGPU4/5和其他owner全部保留，总最多4ownGPU，每卡一个本owner，source≥18GB/target≥12GB、allowner<2。CPU2线程/本owner scorer最多1；新source scorer等scratch根终态，再共用本run score.lock，无循环。权限/代码/冲突、非有限/OOM、PSD/weight/预算/梯度/F冻结或预测失败保留产物并失败该行，无自动重试，健康行继续。

launch后实际PID/CWD/完整命令/GPU/日志增长独立核实。精确命令/CWD/路径/数据与父来源、矩阵、stop rule及产物见experiment.json。IQ/训练权重/训练PSD与向量/逐样本预测/原始日志/私有机器状态不提交。没有向另一用户对话发消息。

一次N607普通账号预检VERIFIED：新root/release不存在，原R03合法来源核实；当前spatialGPU6/7实际target和scratchGPU4/5健康任务全部保留，GPU6/7将在旧owned handles真实退出后使用，磁盘空间充足。无额外签名、数据重验证或审批流程。正式训练尚未启动。


## 实际发布、smoke 与资源队列

方法commit`ba2dbf4de934ae785330877166601a969ece65c6`已推送并独立核实远端OID；一次归档本地/远端SHA比较、远端compile/import PASS。launcher首步实际原R03/三F/四Ls no-query smoke PASS：两行完整初始state/IQ等于原R03，actual复谱loss与coarse/correction梯度有效，原默认头到IQ梯度、F无梯度和checkpoint往返有效；0TX/V/target/optimizer。smoke中的谱权重仅由四Ls FFTpower生成，不作为正式63000source训练谱。

CPUcontroller实际PID/CWD/完整命令LIVE，日志持续记录等待spatial旧owned GPU handles，读取目标分数0；此时两条新GPU模型worker尚未创建。已经独立核实旧spatial target仍LIVE并保持，scratchGPU4/5健康任务保持，未抢占/停止/热改。队列满足旧run终态且实际oldowned handles退出后才创建GPU6/7source；后续PID/GPU/训练日志增长会单独核实。当前没有新训练/源V/目标结果，不能据发布或smoke宣称恢复收益。
