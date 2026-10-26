# NoSIG 有噪声困难伪标签：六行匹配源实验

**RUNNING，六行真实训练已独立核实。** 仅学生seed2026092701，尚无本批完整源结果或目标性能，原三种子实际目标未达成。

执行代码`b298bbc5afbb734e91c19c4c88ef39622a3ff6e8`；一次archive比对VERIFIED、远端compile PASS。冻结RawLabeler真实源checkpoint L2 smoke及完整U56700预测完成，F56702/G0、零optimizer；教师参数和全部注册buffer不变，U真值／target读取0。

得到2858个唯一noisy candidates，占U的5.0406%，来自原E30confidence<.95、R40confidence≥.95并排除RX1。它们不是真实认证标签；已知93.829947%只是源V代理，实际U precision未知。其余U仍原raw self-PL，候选包替换旧self标签而非重复施加两套监督。

六行实际E30/defaulthead/L128/U256联合反传smoke通过，L768/AsKL0、候选32条，三个第二视图行额外32包，默认头梯度有限非零，零optimizer。随后NoisyCE／NoisyScaledCE／NoisyGCE／NoisyGCE2Raw／NoisyGCEG／NoisyGCEHG分别在GPU0/3/4/5/6/7训练；PID/CWD/argv/GPU和训练记录增长独立核实，最近全为累计E32、非有限跳步0。

所有学生同父E30＋20至E50；同一批标签／mask，完整L及原默认头joint。CE、统一缩权CE与GCE(.7)用于分离标签／剂量／降权；Raw2、G、交替HG用于控制第二前向及物理视图。后两行使用原完成三fresh×三scene同G成对缓存，无新G训练、G后归一化或IQ混合。每模型新增4440步、U1134000曝光、候选57160曝光；第二视图行额外F57160包，实际逐view计数另报。教师与缓存成本单列。

每行final自动源V八视图，全部六行完成后再按固定原条件判断：较NoSIG-PL，RX1 G三景≥+.3pp、overall raw≥−.1pp且原困难池raw净正确>0。通过者先源raw，再heldG，再较低成本排序，在任何新target前冻结。RX1仅附加机制留出，基座/G已见，不称真正未见域。此处不以标签近乎无误作为前提，也不修改此前失败路线的条件。

[计划](../../../docs/goals/PAMF_NOISY_HARD_PL_PLAN_20261026.md)、[预登记](preregistration.md)、[启动证据](launch_evidence.json)、[独立审查与检查](local_verification.json)。

## 全部可用记录与扩种子输入准备

本次完整解析当时可用的12122条记录和2097行stdout，不是末尾抽样；live gzip尾部尚可缓冲，未宣称26640条最终记录完成。六行前1879个共同步骤的U曝光／noisy候选累计数逐步完全相同；L768、As KL0、跨RX权重.1、固定q/rho／统一缩权标志、零非有限和额外前向计数均符合配方。完整U epoch恰好曝光56700，2858个固定候选每epoch各一次。

所有六个原worker及controller在该审计时点仍存活，健康训练继续；尚无本批完整源性能，不提前排序／换配方。逐行当时全部完整epoch曲线见[可用曲线](available_epoch_curves.csv)，范围与计数见[完整可用记录证据](available_execution_evidence.json)。不同损失的数值不直接代表正确率优劣。

另外，对seed2026092702/03的原NoSIG E20权重做了真实CPU张量／来源检查：两者均无sequence_pool状态项、原头6×160、全floating state finite；同原L6300/U56700/V27000合同、scratch无祖先、固定E20/4440、final_only、原生U/EMA关闭但custom普通PL保留、无U真值／target／另训头。此检查新增IQ／模型前向／优化均0，尚未真实forward smoke或启动扩种子。它仅准备可用父输入，不代表三种子复现完成。见[剩余种子张量证据](remaining_prefix_tensor_evidence.json)。
