# 全源 E30 冻结默认头：延迟／传播相位四格源归因

状态：**ANALYZED，source-only，0训练更新**。预定延迟重要贡献条件通过，传播相位条件失败。这个结论只适用于已知平坦核counterfactual family；不等于原色散full信道的唯一归因、正式性能提升或新Ag有效。全局研究目标仍ACTIVE。

关键发现：在相同核能量、噪声、RX硬件、接收状态、补偿与AGC下，将平坦核脉冲位置24改为0，B00源mid/low提高26.083333/19.777778pp，B01反而下降2.611111/1.805556pp。原平坦核中的As模型优势由29.777778/23.861111pp降至1.083333/2.277778pp。两个urban×两个draw全部达到预定≥5pp恢复及至少消除一半优势的条件。默认头确实表现出显著的观测时间位置适应差异。

## 固定模型、来源与源权限

执行commit 71f0c8f9fbdd530c51f54764f092b61b9e5f6f18，起点4be849bc060068c1bf72f55676ac8ec3eabf6968。B00/B01来自20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01各自scratch E30=20L10U/6660更新、FINAL30固定model state，非EMA；own_model在加载前核对完整角色/scratch/无祖先权重/源终态选择/无目标训练选模。F=SIG_TIME_X3、CENTER_25K/CENTER_UNITS、joint160/defaultcos30、NoExplicitCFO及原内部补偿，900722参数/206state，默认头仍为训练时共同更新的原头；本次冻结全部参数，未拟合任何头或诊断分类器。

完整原L6300/U56700hidden/V27000、7/63/30、split392005和全部5源RX保持；面板V1800=6TX×RX索引1/3/4/6/8×day1/2/3×20，同一物理ID与命名draw0/1，不留训练RX、不切L_dev、不解封U TX、不用V反向更新。P30已有panel函数直接读取合法源缓存；P24原source句柄和VALIDATED_ONCE复用，无新验证链。

真实2CP/4合成包no-query smoke第一步PASS，此前源IQ0/target0。P33缓存的clean/default logits与flat_H/fixed D00按CP路径、IDs、TX、RX、day、条件轴严格对齐；D00模型预测复用，不重复前向。实际leo_practical模块__file__与P24 actual_engine逐值相同，来自原不可变adv3b02_xuc/code。fs25MHz/256点、residual/post_sync、EQoff即G1、refSNR30、quality/phase/IQ补偿、AGC、reflect边界等原配置不改。所有正式测试IQ/SNR/信道及困难样本不变，本诊断不读取任何目标IQ/truth/正式成绩。

## 已执行四格控制与判据

D00=平坦核冲激位置24/原全局传播相位；D10=位置0/原相位；D01=位置24/零传播相位；D11=位置0/零相位。核能量和每个draw的同一新增噪声保留；full原lock、scale、频率估计、相位/IQ补偿、G1、AGC全固定。RX/day session跨TX稳定，seed大整数由Python直接从原记录读取及stable_seed复核，无JS浮点中转。只改变已知平坦核，不反演TX硬件，不还原未知未来样本。

独立源判据在新源读出前固定：delay或phase分别需在四个urban/draw单元中，B00单项恢复≥5pp，且As优势降低≥max(5pp,原D00优势的50%)。D11仅解释交互，不能用联合格替代独立条件。delay PASS、phase FAIL；不调阈值/脉冲位置/相位/seed救跑，不自动训练/扩seed。两个channel draw不构成跨modelseed复现。

下表均为源V1800×2draw合并，百分比。**其中60%以上的读数是人工源counterfactual，不能冒充正式low urban>60%。**

|源counterfactual|mid B00 %|mid B01 %|low B00 %|low B01 %|
|---|---:|---:|---:|---:|
|D00_delay24_originalphase|49.138889|78.916667|40.472222|64.333333|
|D10_delay0_originalphase|75.222222|76.305556|60.250000|62.527778|
|D01_delay24_zerophase|51.000000|83.250000|41.222222|68.416667|
|D11_delay0_zerophase|79.805556|81.972222|63.222222|65.222222|

|draw|scene|原D00 As优势 pp|B00去延迟恢复 pp|去延迟后优势减少 pp|B00去相位恢复 pp|去相位后优势减少 pp|
|---|---|---:|---:|---:|---:|---:|
|-1|practical_mid_urban|29.777778|26.083333|28.694444|1.861111|-2.472222|
|-1|practical_low_urban|23.861111|19.777778|21.583333|0.750000|-3.333333|
|0|practical_mid_urban|30.055556|27.055556|29.777778|1.500000|-3.000000|
|0|practical_low_urban|23.055556|19.888889|21.500000|1.444444|-2.944444|
|1|practical_mid_urban|29.500000|25.111111|27.611111|2.222222|-1.944444|
|1|practical_low_urban|24.666667|19.666667|21.666667|0.055556|-3.722222|

合并mid/low的B00 macroF1从D00 47.175670%/39.376116%升至D10 75.235110%/60.330560%；B01则从78.805226%/64.185254%到76.219150%/62.257196%。分类间隔统一logits/30、固定B00 clean正确集合保持率、同包修复/破坏、逐RX结果与CM全部保存，不以特征或梯度替代身份收益。

单独去传播相位使B00 mid/low仅+1.861111/+.750000pp，B01却+4.333333/+4.083333pp，模型优势未缩小。它不是As训练优势的主解释，但仍可能是As模型的剩余损失。这里的全局传播核相位与P33的残余载波/CFO/相位跟踪控制不同，不能混同。

## 科学边界与下一项最小实验

延迟24→0也将reflect受影响的前缀范围从24点变为0；本实验尚未把纯时间位置和前缀边界效应完全拆开。flat_H原控制已经去掉频率选择性，所以不能把90%以上的平坦核模型差距缩小，扩张为“full物理星地收益90%都是人工延迟”。D10是已知核counterfactual，不是对真实色散IQ的无损提前或可部署补偿。原full+6practical测试保持不变。

新证据支持优先检验**有界、无循环接缝的接收窗口时移Ag**，而不是继续增加噪声/CFO或对齐损失。它应作用于raw L IQ，保留稳定clean监督，经过完整原As，验证时间位置稳健性是否能改善clean跨RX并保护urban。这个候选尚未实现、登记或启动，不能称有效；原周期、LA16、MMSE/PDP与其他已关闭路线不因此复活。

下一最小训练矩阵拟独立scratch E30的B10/B11，与严格配置核实后可复用的matched B00/B01组成完整四因子；全源/默认头/U规则/4L前向匹配，按既定固定终态逐row七view+truth-last。具体时移边界、源标准与放弃条件须在该候选第一次训练前固定。如果没有实际源身份保持收益、clean保持失败或原full urban没有增量，就放弃该方法，不以本诊断PASS冒充训练候选PASS。

当前实际正式成绩仍为P32：B01 clean83.057143%/low53.394643%，B11 clean83.182738%/low53.339881%；仅1modelseed。原As较matched B00提升low21.639286pp、clean代价2.180357pp。未达到双赢、low>60%、所有suburban接近clean或跨seed目标。

## 验证与成本

5项synthetic检查PASS（kernel能量/相位/位置、源权限反例、统计符号、四单元源门槛、完整collect路径）；6合成native+6原flat+24新render共36次，D00重放误差0/噪声差0，0真实IQ/CP/NN。一次独立限定P0/P1静态审查PASS，local/remotecompile PASS，单archive local/remoteSHA比较VERIFIED。真实smoke首步PASS。

真实源7200 physical draws/28800受控renderer，D00逐包最大绝对误差0，四格噪声最大差0，核能量最大相对误差7.136155e−16，0新native ResidualChannel.process。新43204NN=每模型21602（2smoke+21600新格），D00 14400历史NN复用；0新epoch/训练更新/头拟合，模型state逐tensor不变，无.grad。父两模型历史60epochs/13320updates和P33诊断216004NN分别记账，非本次新增训练。全局目标预算仍用户最新scratch E30。

CPU2/CUDA空，总墙钟82.712022s（受控生成50.955183s、模型读出14.261480/13.440614s）。PID646381/start149070646的CWD/argv/CUDA独立匹配，stdout0→637bytes、stderr0，原进程自然退出、completion存在、无failure。未停止、重启或热改其他健康实验。没有新训练曲线；父模型完整曲线保留原P32报告。

## 代码、命令与实际产物

实现文件experiments/cvs_delay_phase_attribution34/{kernel,render,design,run,runner,stats,checks,publish}.py；__init__.py为包入口。精确配置/数据/六类seed/路径/owner/停止规则见experiment.json，原预登记见preregistration.md。命令与CWD：

```text
# CWD: /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_delay_phase_E30_20261017_r01
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_delay_phase_E30_20261017_r01/experiments/cvs_delay_phase_attribution34/runner.py
```

run=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-delay-phase-e30-s2026092701-m2-r01，release=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_delay_phase_E30_20261017_r01，原日志=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-diagnostic-FcrossRX-delay-phase-e30-s2026092701-m2-r01-launch，controlled_iq.npy及逐包control_records.jsonl.gz保留原远端root。已取回实际结果：[逐model/seed/draw/RX指标](E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017/results_r01/metrics.csv)、[同包效应](E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017/results_r01/paired_effects.csv)、[四格对比](E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017/results_r01/contrasts.csv)、[完整统计与源判定](E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017/results_r01/summary.json)、[生成重放证据](E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017/results_r01/observation_completion.json)、[完成证据](E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017/results_r01/completion.json)、[PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017/results_r01/source_delay_phase.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_delay_phase_E30_20261017/results_r01/source_delay_phase.pdf)。PNG已实际查看，2模型logits/元信息也已取回；IQ/CP/预测文件不提交Git。

Git网络钩子超时后会话重置，独立核实代码commit已生成、登记尚未执行、无残留Git进程，因此从该commit继续登记，没有重复commit或重复启动。直接GitHub TCP/TLS故障下本地提交保留，最终push/OID独立读回另行记录，不force/rewrite。REJECTED_EXTRA_GATE：brainstorming设计审批和通用全库测试前置不在项目最小白名单内；本次已授权诊断未增加设计审批、报告hash、memberhash或重复审查。
