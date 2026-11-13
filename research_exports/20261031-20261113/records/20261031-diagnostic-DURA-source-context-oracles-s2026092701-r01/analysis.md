# 完整两阶段源域诊断：真实算子改善波形，未改善身份一致性

VERIFIED / ANALYZED，SOURCE_ONLY。主对照是纯repeat源final E30/7410；默认DURA仅描述。固定512个源训练池记录（128L/384U），clean输入和原缓存01/20全部覆盖。没有训练、TX/V/target读取或目标预测。CPU循环36.665214秒，3072次context输入、15360完整stage样本、13824新增F特征与1536复用defaultnativeF，另首步G/F各4条真实Ls检查。全部G/F状态不变、无梯度累积。六份完整私有NPZ的IQ/F向量及480个分组case已独立复算一致；没有挑选样本或子场景。

固定源研究门槛两项均失败，不启动rank32或source-action课程训练，不放宽。主对照替换为实际模拟算子，NMSE在缓存01/20分别下降11.452%/11.277%，但与clean F选择一致的记录净少2/5条，clean输入也少1条。这直接说明在该冻结G的联合计算中，物理算子更准确和波形误差更小，仍不足以保证固定F的身份判断更接近clean。它不证明真实算子没有价值，原因可能包含prior/variance与估计算子的共同适配。

rank32对实际source算子的两探针表示误差很小（总体约6.35e-8/5.25e-8，逐场景物理门槛均通过），所以本源模拟器的真实系数时间轨迹可被该固定基底很好表示。但是平滑预测系数后，最终IQ误差略增，Fclean选择仅+1/0，未通过事前双view至少+6/512且NMSE下降10%的门槛。真实算子可表示性不是预测算子准确性，不能据此宣传平滑网络会提高目标表现。

把主对照coarse替换成sourceclean时，F一致数+6/+5，但NMSE只略降；这是未参与晋级的描述性case，仍含部署时不可用的逐条clean。默认DURA同样替换后F一致数反而−34/−32、NMSE上升，说明coarse/prior可能是联合求解的潜在表示；不能把coarse单独不似clean视作模块失效，或直接将其改成clean监督。两模型结果不能拼成一个可部署方案。

|模型／源view|方案|平均NMSE|与clean F选择一致|相对native净变化|
|---|---|---:|---:|---:|
|repeat_E30/clean|native|0.119031%|512/512|+0|
|repeat_E30/clean|known_action|0.020448%|511/512|-1|
|repeat_E30/clean|known_coarse|0.070017%|511/512|-1|
|repeat_E30/clean|known_both|0.014522%|510/512|-2|
|repeat_E30/clean|rank32_action|0.118521%|512/512|+0|
|repeat_E30/1|native|2.026847%|488/512|+0|
|repeat_E30/1|known_action|1.794732%|486/512|-2|
|repeat_E30/1|known_coarse|2.017999%|494/512|+6|
|repeat_E30/1|known_both|1.915561%|495/512|+7|
|repeat_E30/1|rank32_action|2.031204%|489/512|+1|
|repeat_E30/20|native|1.911455%|497/512|+0|
|repeat_E30/20|known_action|1.695902%|492/512|-5|
|repeat_E30/20|known_coarse|1.886500%|502/512|+5|
|repeat_E30/20|known_both|1.791011%|500/512|+3|
|repeat_E30/20|rank32_action|1.915066%|497/512|+0|
|DURA_default/clean|native|0.003295%|512/512|+0|
|DURA_default/clean|known_action|0.000777%|512/512|+0|
|DURA_default/clean|known_coarse|0.003673%|511/512|-1|
|DURA_default/clean|known_both|0.000358%|512/512|+0|
|DURA_default/clean|rank32_action|0.002963%|512/512|+0|
|DURA_default/1|native|2.021283%|486/512|+0|
|DURA_default/1|known_action|1.945774%|477/512|-9|
|DURA_default/1|known_coarse|3.135636%|452/512|-34|
|DURA_default/1|known_both|2.790745%|453/512|-33|
|DURA_default/1|rank32_action|2.026606%|487/512|+1|
|DURA_default/20|native|1.943971%|498/512|+0|
|DURA_default/20|known_action|1.787570%|499/512|+1|
|DURA_default/20|known_coarse|2.813661%|466/512|-32|
|DURA_default/20|known_both|2.491676%|464/512|-34|
|DURA_default/20|rank32_action|1.948693%|497/512|-1|

使用原生stage_outputs处理全部两个prior和四步CG，五case固定为native/known_action/known_coarse/known_both/rank32_action，原生variance始终保持，替换coarse后不重新估计A；rank32只投影预测main/image的首32个DCT模式，kernel不变。真实A的rank32两白噪声探针只用于可表示性。默认native IQ重放与旧source-phase缓存最大绝对差0，在2e-5界内，因而复用旧defaultnativeF；主对照native新算，没有使用旧noVar native输出。原F50/defaulthead源clean引用的ID/位置/角色/RX/day/clean/场景严格配对。

这不是识别真实TX的准确率：Fclean本身可判断错误，512条是源训练池而非独立V或跨RX正式测试。参考替换拥有部署侧不可用的信息，且冻结其余模块后的误差不是任何新模型的最优风险。因此本结果既不能成为恢复极限，也不能外推成所有增强或端到端训练无效。它定位了“更低波形误差不自动等于更好识别”这一具体问题。

4项聚焦检查及唯一P0/P1审查通过；执行97da8a29fce23f7950bcfca9f525e4d091fa9757，单archive本远SHA一致、remotecompile通过、实际PID1273189/CWD/argv/CUDA空/日志301→548→927字节和自然结束已核实。没有新增GPU任务或修改其他owner。默认DURA和六景逐个距rawclean≤3pp、clean保持、同质量训练效率目标仍未完成。
