# 有界非恒定相位视图：双默认头完整源身份结果

**ANALYZED／VERIFIED，PASS_SOURCE_IDENTITY_ONLY。** 两份合规冻结默认头在四次全覆盖观察下均通过原源身份保护条件，可以继续检验训练价值；当前没有新 E20、heldRX1／target 成绩或 A_g／A_s 有效结论。

底座固定 F＋跨 RX 同 TX／三路 SIG_TIME／统一 Sinc／joint160／NoExplicitCFO／默认 cosine30 头正常联合训练，后续预算 E20。[原始预登记](preregistration.md)、[唯一矩阵／来源](experiment.json)、[候选说明](../../../analysis/cvs_bounded_nonaffine_phase_candidate_20261017.md)保留。执行 commit b59b0d92c67d6e8d14056e666cfb6f2d4a3f0859。

## 实际识别结果

全部只用原训练角色 fitL5040（RX3/4/6/8×day1/2/3×6TX，72格各70）。raw 结果接近100%是已见源训练样本检查，不是 clean 跨 RX 泛化成绩；四 draw 重复相同5040物理包，不当独立新数据。两模型为原合法 scratch E20 final20 B00，原 L5040/U45360/V21600、RX1 held、4440更新、final-only，完整来源、角色、native args和默认头已核对，本次不续训或另训头。

| 模型 seed | raw accuracy % | phase accuracy % | accuracy pp | macro-F1 pp | raw正确保留 % | 修复／新错 |
|---|---:|---:|---:|---:|---:|---|
| 2026092701 | 99.841270 | 99.821429 | -0.019841 | -0.019881 | 99.965223 | 3/7 |
| 2026092702 | 99.841270 | 99.811508 | -0.029762 | -0.029766 | 99.970191 | 0/6 |

| seed | draw1 accuracy pp | draw2 pp | draw3 pp | draw4 pp |
|---|---:|---:|---:|---:|
| 2026092701 | +0.000000 | -0.039683 | +0.000000 | -0.039683 |
| 2026092702 | -0.019841 | -0.039683 | +0.000000 | -0.059524 |

两个模型 pooled 和每 draw 的 accuracy／macro-F1≥raw−0.5pp、raw正确保留≥99%、每RX/TX/day accuracy≥raw−1pp，以及全观察 IQ 相对RMS变化 median≥0.01、max≤0.1，全部通过。每个RX/TX/day与72格的完整accuracy/F1/混淆矩阵、CE和原条件复算见聚合JSON。通过只支持这个固定观察在两份源模型上较少损坏默认头身份，不证明训练后一定有益。

## 算子与全部扰动

对每包全部256点，Float64 g[n] iidN(0,0.006²)，phi=cumsum(g)，投影去掉常数与固定样本索引线性项，再整路径按 min(1,0.1/maxabs) 收缩；原接收复数IQ（含既有噪声）乘 exp(j theta)，输出Float32。所有极端路径保留，没有逐点剪裁、删除、重抽或RX/TX/day/标签/置信度条件。没有PA、AWGN、multipath、DAC、CFO、常相位、RMS恢复、重采样或完整practical。原幅度和全包功率数学上保持；舍入误差实际记录。

sigma0.006来自实际旧普通增强未缩放上限；峰值0.1rad是本轮先验工程约束，使每点相对复数变化≤2sin(.05)<10%，不是由源/目标曲线选出的幅度。旧ordinary本就含低概率Wiener，因此这里只是独立、全覆盖、有界且去仿射的视图变体，不能称新的增强原理。有界缩放后的过程不是原始Wiener分布，也不是已识别的真实RX振荡器；数字旋转不等于替换物理RX。

| 全20160观察／模型，同命名路径 | median | q95 | max |
|---|---:|---:|---:|
| 相位 RMS rad | 0.022571150638 | 0.036858616354 | 0.057944886866 |
| 相位 peak rad | 0.057166898816 | 0.087579882811 | 0.100000000000 |
| 全包 IQ 相对 RMS 变化 | 0.022373112007 | 0.036630173402 | 0.058647748458 |
| 80:160参考窗 IQ 相对 RMS 变化 | 0.022989825903 | 0.043162613983 | 0.072427006556 |

每模型312/20160路径（1.547619%）触发整路径收缩，最小缩放0.7131059402；其余路径不缩放，没有丢掉任何尾部。全包相对功率变化绝对最大1.5599577e−8；相位均值绝对最大3.7703131e−17rad，线性项绝对最大2.7888985e−19rad/sample。Float64旋转前后逐点模长最大误差8.8817842e−16，Float32输出功率单独报告。所有参考窗都有非零功率。两个模型全部路径、顺序和320batch的输入变化统计逐项完全相同。

源相位曲率RX关联通过不等于这些新随机轨迹重现了真实RX差异。本算子不回放大相位源模板，也不声称覆盖所有RX相关变化。旧常相位E20、可训练feature-curvature、cyclecovariance和其他固定失败保持关闭。

## 完整复算、实际成本和执行

独立reader未导入候选score/qualify/phase/model。完整复算50400源logits的有限性/argmax、硬判决、标量logsumexp CE、accuracy、六类macro-F1/混淆矩阵、RX/TX/day、72格、每draw、raw正确集合和修复/新错、全部固定条件。8份压缩文件共94944260bytes完整读取，10321920个保存相位坐标全部检查，重算每包RMS/峰值/均值/线性项；全部收缩尺度和峰值、功率/RMS关系、IQ与参考变化统计和成本核对一致。没有只抽样分析，也没有加载source IQ或模型重算：独立范围限于保存的预测、路径和观察；原IQ和真实旋转输出没有传回，不能冒称再次从原IQ重做全部物理检查。

新run的raw metadata与此前孤立AWGN源检查一致，两个模型的raw全logits和hard预测也完全一致，最大logit差0。这仅验证本次raw控制和同一默认头，没有复跑旧增强或借用旧结果替代当前预测。

实际50402 NN＝50400源默认头＋首模型一次真实checkpoint合成2包no-query smoke。源相位40320包、另64合成相位；2checkpoint加载、2source cache加载、0新epoch/优化器/教师/另训头/ordinary/PA/As/target。本地28输入/配置负例＋18解析/科学反例、388合成相位另列，0sourceIQ/CP/NN/更新；旧NumPy桥未调用、不装包。原两参照历史40model-E/8880更新/9373356NN单列，不作为本次新增训练或未来初始化。

每次cache反序列化L6300与未用V/历史features，RX1原L1260在推断前排除；U只有opaque角色ID，没有U IQ/TX真值。未对U/V/heldRX1/target推断，不混同fullL6300或L_train5580/L_dev720。

一次独立P0/P1审查PASS，一份archive/一次传输SHA比较、普通账户preflight和remotecompile VERIFIED。真实CP smoke为首worker第一步，PASS立即继续。两worker均实际独立观察到PID/CWD/argv/CUDA隐藏，日志从live至终态增长；最终dispatcher与两worker自然退出、无技术failure或健康旧进程干预。没有持续live日志采样的夸大。两worker主体41.626755／42.328986秒，合计83.955741秒，总dispatcher墙钟未单独计时。分析和绘图0新IQ/NN/更新。PNG/PDF已生成，PNG实际查看。

聚合见[source_analysis.json](source_analysis.json)、[执行摘要](execution_summary.json)、[历史raw控制核对](raw_reference_comparison.json)。图：[源默认头身份与全部扰动分布](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_bounded_nonaffine_phase_source_20261017/final/figures/bounded_nonaffine_phase_source.png)；同目录PDF保留。原始IQ/权重/私有ID/完整logits/相位路径/日志/运行账户路径不入Git。

## 下一唯一假设与最小实验

本轮证实该固定有界相位视图可通过源身份保持条件；没有证实跨RX提取或practical适应。下一唯一假设：把这个视图用于监督训练，能在同预算F＋跨RX同TX中带来独立clean收益，而不是只靠增加前向或旧ordinary重复。

最小后续矩阵为三组从零E20：B00额外raw、C_ordinary额外同一份旧ordinary、B10额外phase(raw)。公共raw.75＋ordinary.25/SupCon/原U保持；三组都额外.25、总权重1.25、8个L前向，RDrop的两次前向使用同一已生成IQ。无As或组合视图，不继承源检查模型。不以source-fit识别率作为训练成功。

固定放弃条件：B10相对两个当前匹配控制都须source-heldRX clean≥+1pp、macro-F1不降、每practical/seenV≥−0.5pp，完整E20/4440更新与正常SupCon/U执行；任一失败就关闭sigma.006/peak.1全覆盖方案，不调幅、加损失、筛RX/包/seed或延预算救跑。通过后再独立seed确认，并继续完整契约Ag/As四格与交互项；这些训练尚未实现/登记/启动，不能把源guard PASS冒称E20已完成。

联合目标ACTIVE：正式clean净增益、lowurban>60%、suburban距自身clean≤2pp探索目标、稳定As、完整契约四格及跨seed均未完成。
