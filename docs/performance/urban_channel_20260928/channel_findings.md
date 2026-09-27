# Practical urban信道与监督质量核验

日期：2026-09-28。范围为只读源码/已结束实验证据分析；未训练、重新评分、修改信道或选择目标最优模型。源码根目录`C=experiments/adv3b02_xuc/code`（本分支；行号对应修改前c694c57f快照）；配置目录为其相邻`configs`。后文`C/`均指这个已核实目录。

## 1.先纠正一个会影响方法设计的事实

六比例实验**没有把mid/low_urban输入DAOT教师**。虽然配置写`daot_hard_scenarios=practical_mid,practical_low_urban`，实际A1解析后教师只有2个视图。`C/scripts/train_rc4_practical.py:28`强制`daot_teacher_view_count==2`；L路径`C/SSDG/train_ssdg.py:2176`以及U路径`:2463`只有计数≥3才生成hard视图。实际教师是clean+practical_high，学生信道视图是practical_high。

因此，[旧六比例最终报告](evidence/legacy_ratio_analysis.md)中“DAOT student high与teacher mid/low_urban保持不变”必须理解并修订为：**DAOT学生high、教师clean+high不变；mid/low_urban只是未执行的hard配置字段。**不能据此推断DAOT强迫low_urban与clean一致导致失败。

RC4的U卫星身份监督同样关闭。配置`rc4_residual_ratio_high_50_50_20260920.json:206`为`rc4_satellite_hard_only=false`。`C/SSDG/train_ssdg.py:10422`构造mask时，fasttrust_rc4且该开关false产生全零mask；`:7240`至`:7246`仅该开关true时计算卫星hard CE。`rc4_lambda_satellite=.10`这个残留值不证明机制执行。六比例的urban直接身份监督主要来自E80后的**有标签拼接卫星CE**，不是RC4 urban伪标签，也不是DAOT困难teacher。

## 2.full、residual、noeq究竟表示什么

|术语|实际含义|不能作出的推论|
|---|---|---|
|full|显式合成时变多径、遮挡、载波/RX扰动、噪声，再模拟接收端补偿|full不等于完全未补偿|
|residual|短记录内冻结物理信道，构造H及可选均衡器G，一次应用等效GH；保留广义线性IQ镜像、残余频偏及噪声着色|residual不等于弱信道或自动均衡|
|noeq|关闭传播信道均衡；residual中G=1，仍保留H|noeq不等于关闭频偏补偿、IQ校准或AGC|

依据：`C/leo_practical/residual.py:1`至`:6`、`:40`、`:61`至`:66`、`:86`至`:108`；full实现`C/leo_practical/channel.py:395`起。六比例配置`:276`/`:277`明确residual与equalization=false；因此仍有完整传播损伤的冻结近似，并非“只剩很弱的均衡残差”。

可选EQ为129-tap、目标延迟64、20dB增益上限的FIR MMSE或正则ZF；估计来自真实模拟H加误差，不是从输入真实导频估计，见`C/leo_practical/equalization.py:40`至`:65`。不得将此包装为已实现实用盲均衡。receiver失锁时，即便配置开启EQ也不应用，见`residual.py:61`。

## 3.urban随仰角变化，不是随轨道高度切换

`C/leo_practical/channel.py:24`至`:31`定义high为45°至80°、mid为20°至45°、low为10°至30°；轨道高度默认600km（`:45`），六比例没有覆盖它。high/mid/low指**仰角**，不指三种轨道高度。

urban先验为`P_LOS=1−(90−θ)^2/7000`、`P_PARTIAL=.2*(1−P_LOS)`，余下为BLOCKED；suburban分母16600且partial比例.8（`:171`至`:177`）。按代码的区间均匀仰角解析积分，urban high/mid/low的平均BLOCKED概率约9.81%/38.38%/56.38%；low_suburban约5.94%。这些是模型先验计算，不是运行日志实测。低仰角城市确实会产生大量遮挡样本。

状态参数为LOS/PARTIAL/BLOCKED直达幅度均值0/−8/无直达，弥散功率−18/−18/−20dB，RMS delay20/50/100ns（`:60`至`:63`、`:444`至`:453`）。相同状态的这些工程参数没有因urban标签额外改变；urban主要通过状态混合比例与仰角/距离起作用。

AWGN底固定为相对参考单位输入的`10^(−30/10)`，不按每条衰落后信号重新定SNR（full`:469`至`:482`；residual`:50`至`:55`）。低仰角距离增大、遮挡削弱信号，噪声底不随之减小，所以困难信号的有效SNR会显著下降。AGC将信号和噪声一起放大，不能恢复SNR。输入本已有地面噪声，代码只知道新增噪声，不能把quality_snr称总SNR。

residual的质量指标使用输入白谱近似`input_power*sum(abs(H)^2)/noise_variance`，full使用实际合成后功率；彩色WiSig输入可能导致二者质量判定不同（`residual.py:52`至`:55`、`:148`至`:150`）。quality<5dB为失锁、5至15dB退化、≥15dB锁定，残余误差随质量放大至10倍（`channel.py:301`至`:310`）。这是一套工程误差模型，不是实测捕获概率。

## 4.训练与固定测试匹配程度

物理函数/route/EQ设置相同。预测从checkpoint args重建信道（`C/scripts/predict_phase1_truth_last.py:154`至`:173`）；目标视图seed固定392005，按opaque physical ID、scene与namespace确定，独立于batch分块（`C/cvsrffi/practical_adapter.py:31`至`:33`、`:74`至`:104`；`C/leo_practical/batch.py:75`至`:83`）。训练每个view调用使用新的generator种子，namespace含epoch与L/U，硬件按RX/day冻结；测试为`virtual_target_session0`的单一虚拟新增RX硬件。这是显式训练/测试随机化差别，不是逐样本真实卫星硬件测量。

六比例只改L拼接分支的场景条件比例：high或mid与low_urban组合，应用概率E1–40为.3、E41–90为.6、E91起.8；CE到E80才生效。75%困难是“已应用卫星样本里的条件比例”，不是训练总输入的75%。固定测试覆盖六Practical场景，训练并未显式覆盖每一种urban/suburban×仰角组合；六场景等权均值也不是训练分布等权风险。

旧报告固定E200六组low_urban约52.42%至53.71%，结果可描述困难瓶颈及类别代价，不能据此选择下一轮比例、弱化信道或选checkpoint。单一seed及重复epoch不能作为独立重复。项目协议`项目.md:75`要求V只读且不得更新EMA/prototype/normalization，`:103`禁止target反馈研发。后续诊断应使用合法source L与单一source V的固定信道视图。

## 5.信息损失与监督风险：事实与假设分开

已证实的设计事实：信道不主动校正TX/PA参数，仅新增传播和虚拟RX链路；EQ也只针对新增传播H。但这不是TX信息完整性的数学保证。BLOCKED时无直达、固定噪声底、随机多径零点、256点有限窗及反射边界都可能遮蔽原指纹。25Msps下256点仅10.24µs，fractional-delay滤波有24点公共数值延迟；residual冻结近似合理与否、反射边界占比、EQ129点记忆对短窗的影响都应在source上量化，不应只看无NaN。

有标签卫星CE在`C/SSDG/train_ssdg.py:9858`/`:9912`对选中卫星视图保留原TX监督，没有按receiver lock或可恢复性逐样本降权。这是潜在风险：极低信息样本仍被要求高置信识别。但不能从目标低分直接证明标签已“错误”或指纹已被完全删除；传播后物理身份仍是同一TX，问题是观测的可辨识性。

当前mean DAOT把reliability、importance置为1，recoverability也为1（L`:2221`至`:2242`；U`:2509`至`:2530`）。feature一致性没有教师consensus门控，而logit蒸馏才用consensus（`C/cvsrffi/daot_training.py:91`至`:102`；`C/cvsrffi/orbit_teacher.py:88`至`:120`）。这个事实目前针对clean/high两视图。如果将学生直接换成urban而不加质量区分，可能把严重受损视图强行拉向身份目标；这是待source验证的风险，不能写成当前六比例已发生的urban DAOT失败。

## 6.最小source-only优化设计与接口

第一步只做只读source诊断，不改变方法：固定同一批source V物理ID和信道seed，保留clean/high/urban视图，按LOS/PARTIAL/BLOCKED、lock、质量、仰角分层，记录教师正确率、错误一致率、同TX/异TX特征间距、logit margin、卫星CE、频谱衰减、有效能量/边界样本占比；使用V不反传，不创建新的永久V角色。不能用U隐藏TX做教师正确率，也不能用target混淆选择特定TX修复。

若source诊断支持扩大DAOT学生覆盖，最小改法是**保留原clean+high教师，单独配置学生信道视图**。现有名字`daot_student_scenario`实际同时控制student和teacher第二视图，不能只换字符串：

- L接口：`_compute_daot_labeled_step`（`train_ssdg.py:2121`）内`:2157`决定medium，`:2169`生成x_medium，`:2187`喂student，`:2199`把同tensor喂teacher。
- U接口：`_compute_daot_unlabeled_step`（`:2421`）内`:2444`决定medium，`:2456`生成，`:2474`喂student，`:2487`喂teacher；上游clean teacher复用保留。
- 增加显式`teacher_channel_scenario`和独立student选择策略/种子流，默认保持两者high，原配置逐步输出不变。两条路径都应缓存/复用固定teacher view与结果，不能为了urban学生无意重新采样teacher。学生场景频率由source设计预先冻结，不以历史target最高比例为默认。

现成可靠性helper**不能原样复用为“无算法变化”**：`C/cvsrffi/deployment_orbit.py:180`至`:183`依赖SNR、仰角、K_db；Practical adapter`:122`至`:126`没有输出K_db，缺项在旧helper变成0，且mean路径将结果覆盖成1。原recoverability来自teacher侧平均，不是新增urban student质量。直接开robust还会改teacher中心，不满足“保留clean+high原教师”的隔离要求。

较小且可解释的候选是保留教师聚合，单独给新增学生feature/logit一致性传入source定义的student可恢复性权重；先用已输出的quality/lock及source校准，不伪造K_db。仍需明确登记这是新的损失加权行为，不能宣传仅性能优化。还需注意`orbit_feature_loss`采用加权均值，统一缩小所有权重不会缩小总损失，只有样本相对权重改变；若目标是减小整批低质量监督，应单独设计coverage系数并明确比较，不混入现有loss-scale归一化造成伪门控。当前不指定阈值或比例，待source诊断决定。

最小验证应覆盖：①旧默认输出/损失不变；②L/U两入口teacher固定clean+high且urban只送student；③named RNG隔离、同seed可重放、不同batch分块固定测试不变；④零有效权重给出有限0 loss/0梯度，坏质量不能经clamp下限恢复成强监督；⑤metadata缺项显式处理而非默认为K=0；⑥U全程无TX读入，V不更新状态，目标不可回流；⑦实际激活阶段测forward数、信道耗时及峰值显存，确认没有额外不必要teacher调用。若复用现有有标签urban tensor，必须核对其物理ID、基础增强和seed语义，不能仅shape相同就当同一视图。

加速上优先保持物理分布和固定测试：缓存/批处理同配置常量和receiver参数、避免GPU↔CPU拷贝重复、复用已生成视图和已有teacher结果、identity-only教师/评估。不能把取消urban、跳过新增噪声、删掉quality失锁分支或缩小测试量当等价加速。residual已是冻结短记录加速近似；任何更快实现应比较逐记录物理元数据和IQ误差，而不是只比较最终accuracy。
