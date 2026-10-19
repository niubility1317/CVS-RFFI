# SIG_TIME 边界与内部等量平均：固定 checkpoint 源域归因

状态：SOURCE_ANALYZED / VERIFIED，固定源诊断与独立重计完成；事前条件FAIL，不启动训练或E50，总体目标未完成。

当前raw补充路径在冻结原F和共同更新两轮均没有形成超过普通As的源增量。这里不再加宽、换rank、换seed或调整旧loss。新问题是现有SIG_TIME聚合是否把边界局部波动放大为判别变化。实际代码三路均先tanh(local_projection(t2))，将32位置分为四个8位置窗口；一阶空间量等于u_end−u_start，空间与时间的反对称面积等于梯形积分u−(u_start+u_end)/2。后者恒等式只用于核验算子，不能把CNN相关噪声当独立白噪声。

本轮不训练、不新增模型或默认头，不访问V/U/target，不调整信道、不新增模拟。用B0 FINAL20和预定普通FULL_As_JOINT FINAL30两个合法源固定端点。都由原strict loader加载，数据角色及scratch父来源复用既有核验，不依据正式分数选择。

## 固定2×2局部算子与预算

每个原8×6窗口切为四个连续2位置区域。E把第1和第4区域各自替换为该区域均值（共4位置）；I把第2和第3区域各自替换为区域均值（同为4位置）。比较NATIVE、E、I、EI四条件，三路均按同一规则。位置数、6维、四窗、path_signature公式、原输出投影/归一化/增益、实际joint160及原默认W均保持。没有新学习参数或新分支，不把局部平均解释为物理信道反演。

E/I不是仅替换两个端点，也不是加权选择最有利窗口；其影响同时包含窗口内局部面积变化。等量I控制用于区分一般粗化与边界位置的作用。EI为固定交互条件，不作可选超参数候选。区间长度来自原四窗的再四等分，不扫长度、位置或seed。

复用R02 bank的合法L1800面板：raw、actual_first训练As混合面板、同L新mid/low各2draw，共10800/model；两模型21600完整编码器NN。真实CP首步使用零IQ2包/model共4NN/no query，PASS立即继续。总21604完整NN；每一实际源输入只跑一次编码器，额外三种科学条件头读出64800、原生logit复核21600；原native21600，总头决定108000。额外pool输出投影86400，首步smoke另有20次头决定。不将这些局部神经计算隐去或冒充额外接收样本。CPU2、GPU0、训练/拟合0、物理模拟0。

实际前向只增加只读capture：捕获原ReferenceCVS.features返回的base、三路原local_projection、原sequence_pool输出。要求base+native_pool与实际joint160逐位相等；原pool公式重构与原输出逐位相等；同原classify的native logits逐位相等。改变条件直接用base+改变后的原pool，再经同一W classify；不用减法近似重建base，不拟合任何分类头。结束恢复方法与hook，全部参数/持久buffer逐位不变。

## 固定解释和放弃条件

唯一主观察为FULL_As：E相对NATIVE的新mid/low四读出等权均值至少+0.5pp，E相对等量I至少+0.5pp，raw L1800 clean相对NATIVE不低于−0.2pp，三项同时满足才记作支持继续研究该边界假设。B0及实际训练As按场景作为解释，不能替换主观察；无V、未见RX、部署或可重复收益结论。

否则关闭这个固定四分区域平均的边界方案，不扫窗口/位置/父模型/seed救回。通过也不自动训练或E50；后续训练设计必须另行固定同预算NoAs和普通As。本轮没有新训练模型，因此不重复既有正式预测/评分。冻权重下的负结果不能否定训练后所有可能的SIG_TIME表示，也不宣称物理上限。

唯一owner codex/root/F-sig-boundary-source；输出PROJECT/runs/20261019-phase1-F-sigBoundary-source-s2026092701-r01，CWD=PROJECT/releases/cvs_F_sigBoundary_source_20261019_r01。普通账号既有Torch2.1/FP32/TF32off/CPU2，CUDA禁用。技术/协议/来源/接口/非有限/原生回放不等/持久状态变化则保留failure，不覆盖或自动重启。预期config/smoke/progress/每模型source logits与原生回放核验/results/complete或failure。最小发布只用代码commit、一次P0/P1、首步realCP smoke、单归档SHA/compile和一次PID/CWD/argv/log增长检查；没有额外seal或数据重验。

## 实现追踪

| ID | 来源 | 要求 | 目标 | 状态 | 验证 | 边界 |
|---|---|---|---|---|---|---|
| R01 | 附件三.3/七/九 | 从真实SIG_TIME端点机制提问 | 本报告/实际descriptors | verified | 原代码恒等式 | 非独立白噪声假设 |
| R02 | 本轮固定2×2 | E/I等量、EI交互、native严格保留 | operators/checks | verified | 合成张量/恒等式 | 非仅改两端点 |
| R03 | 目标三/七 | 实际base、pool、joint160和原W精确回放 | probe/checks | verified | 全模型与实际源 | 无新头/拟合 |
| R04 | 协议来源禁令 | 合法两个固定父及L角色 | runner/design | verified | 原strict loader/既有bank | 不读V/U/target |
| R05 | 最小工作流 | commit/review/smoke/发布/后态 | publish/runner | verified | 独立实际证据 | 不重复审批 |
| R06 | 目标八 | 完整源读出/配对/固定决定/失败保留 | report/analysis | verified | 全条件独立重计 | 不自动训练 |

最高风险是局部平均损伤了已学习的路径几何，或源端变化不转移到未见RX；本诊断不承诺性能改善。严格实现事前E/I算子，不能把它称为原物理噪声的精确分解。

本地接口修正记录：真实F包装类为SequenceReference，t2最终输出32位置，四窗各8、再四分区域各2。首次结构名称检查及随后尺寸检查在合成输入阶段失败，未访问任何实际CP/IQ/target；按真实路径修正，不是依据源成绩调区域长度。完整模型synthetic回放随后验证算子与实际forward一致。

本地10NN完整模型回放/5负测PASS；原输出与捕获、替代读出与直接前向逐位相等，方法/hook/模型/RNG恢复。此前两次合成接口失败和尺寸核验共8次尝试包前向，合计本地18次尝试NN，0真实CP/IQ/target。实际额外原生logit复核21600已单列计入头计算，不隐去局部神经成本。

唯一一次独立P0/P1审查PASS，范围见review.json；8模块、配置和0NN指标/源判定核对通过，0真实CP/IQ/target/remote，未改文件。代码/config commit 836c646c8de05a813f9c4e3baa2e350ff2b86f1e。命令 CVS-RFFI/python -u experiments/cvs_sig_boundary_source/runner.py；CWD=PROJECT/releases/cvs_F_sigBoundary_source_20261019_r01；普通账号既有Torch2.1/FP32/TF32off/CPU2/CUDA禁用。Git push失败于HTTPS连接，保留本地commit，未将网络交付增加为实验审批。首步真实CP4NN零IQ smoke PASS后立即进行固定源读出。

完整结果、源边界和关闭决定见final_report.md/results_summary.json。真实CP4NN/no source-queryIQ smoke、单归档/compile、PID/CWD/argv/CPU2/GPU0/log增长与自然终态见launch_evidence.json/terminal_outcome.json。
