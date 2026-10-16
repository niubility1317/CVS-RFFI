# 短窗频域相位测度：本地合成反证预登记

状态：PLANNED，正式矩阵未启动。run：20261017-diagnostic-F-crossRX-frequencyPhase-synthetic-m1-r01；owner：codex/root/F-frequency-phase-observability-20261017。矩阵／所有数值以experiment.json为唯一配置入口，实际执行commit由resolved_config.json记录。

本轮只检验短窗频域相位测度是否可辨识；无新IQ增强、模型或分类头。F＋跨RX同TX／三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认cosine头正常联合训练／E20／原U保持。当前源物理L/U/V、formal target与received IQ均不打开。真实长上下文入口及旧幅度谱FIR／时域非仿射相位／周期残差协方差／四周期抖动等固定失败保持关闭。

## 固定测度

输入为256点复IQ，模拟原packet RMS并转Complex64后以Complex128计算。只在网络既有80:160参考区域内，用lag20复内积角估计公共频偏并扣除；估计量不接收真实CFO，也不把它输出为身份特征，不改变现有模型内部补偿。对80点窗口投影到k=-6..-1,+1..+6、周期20的12个已知谐波。对中心k=-5..-2,+2..+5计算单位模A[k-1]A[k+1]conj(A[k])²，得到8个相位闭合观察。公共相位与线性频率相位在此抵消；不把抵消线性频率相位等同于消除时域CFO。

噪声代理为12谐波重构后的68自由度残差，并设1e-12相对数值floor；每个必要谐波功率／投影噪声比须≥25。普通情况任何包不可靠就将整个cohort记为不可辨识，不丢包后求成功率；DEGENERATE明确要求拒绝。每个合成TX/day/RX格70包的复均值先作RX圆参考，再减RX均值；cell resultant及RX圆参考resultant均须≥.8。对6TX×3day的18折同时排除一个TX和一个day，用其余5TX×2day预测每个RX相位；零RX对比和3种错误RX循环映射为控制。这是均值统计，不拟合身份或RX分类器。

## 完整最小矩阵与放弃条件

3种预定12音模板（flat、QPSK、chirp），7情况（RX_LTI、NO_RX、TX_CURVATURE、RX_GAIN_ONLY、TX_RX_INTERACTION、EDGE_ONLY、DEGENERATE），SNR20/30/40dB，每格4抽样。每次6×3×4×70=5040个合成包／256点，总252格／1270080包／325140480复观察点。data seed202610178201+templateindex×100000+SNRindex×1000+draw；7情况共用latent随机量，真实coefficients／CFO／响应仅生成与独立评分可见。所有合成因素都不映射真实物理TX/RX。

RX_LTI用a=[-.45,-.15,.15,.45]单位模全通响应；一半合成CFO在该响应前、一半在后，正例与零RX情况都保留随机公共相位、增益、时间偏移、RX/day采样钟差及propercomplex噪声。TX_CURVATURE只改变TX滤波；RX_GAIN_ONLY为实正幅度响应；交互情况按TX交替共轭RX相位，不含稳定共享主效应。EDGE_ONLY只改80:160之外以检查观察窗局部性，不把它当实际滤波瞬态证明；DEGENERATE缺失+3必要谐波。具体表达式／范围保存在配置与代码，合成SNR与范围不是实际源分布或未来Ag幅度。

正例每次必须全部可靠、profile相对RMSE≤25%，18折风险至少下降50%、至少15折改善、每RX风险不升，正确RX映射的风险改善比最佳错误映射至少高5个百分点。RX检测同时要求profile RMS≥.02rad。NO_RX／TX_CURVATURE／RX_GAIN_ONLY／EDGE_ONLY都须可辨识、共享RX profile RMS≤.01rad且不触发RX检测。纯交互须不触发检测、crossfit风险改善≤0且profile RMS≤.01rad；缺谐波退化每次拒绝比例≥99%。全部252格／63组各4次都须满足原条件；不挑模板／SNR／case／抽样或删包。

任一条件FAIL，完成全部格并关闭当前固定80:160／lag20／12谐波phaseclosure估计器；不改窗口、period、carrier、CFO方法、阈值、SNR、draw、seed、模板、case或提供oracle救跑，不读源IQ或启动E20。PASS也仅支持另行登记源观察验证，不证明RX硬件因果、标签保持、跨RX clean、As或联合目标。合成稳态频率响应及窗外扰动不覆盖真实捕获误帧或滤波瞬态；不能拿本结果替代这些缺失。

## 执行与验证

本地19输入／配置负例与8项代数检查PASS：公共旋转／增益／可补偿CFO不变、窗外修改不变、缺频拒绝、理想共享相位恢复及交互反例拒绝。原始合成生成144包／36864复点；6次测度调用432包／110592复点。均0真实IQ／checkpoint／NN／更新。正式含噪矩阵未执行；一次限定P0/P1审查另附，不新增重复审查。

本地普通Windows已有F:/App/miniconda3/python.exe／NumPy，CPU2线程，CUDA_VISIBLE_DEVICES为空；CWD为当前worktree。命令：F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_frequency_phase17.run --output local_artifacts/F_frequency_phase_observability_20261017/run。不可覆盖输出根local_artifacts/F_frequency_phase_observability_20261017/run；日志local_artifacts/F_frequency_phase_observability_20261017/run.stdout.log。只允许既定单一owner启动一次，技术失败保留全部输出、不自动重试或改动任何健康旧任务。

实际首项仅是输入／配置／输出位置核对；无模型，不加载真实checkpoint作无用smoke，无远端发布，因此N607预检、archive及remotecompile不适用，不建未消费未来流程。仅用一个实际代码／配置Gitcommit，一次本地PID/CWD/cmdline/CPU/loggrowth观察；全部252格完成后以不导入估计器的stdlib reader独立重算cell means对应的RX profile、18fold风险、错误映射、指标与63组判据，再更新同run报告／登记／Git。

REJECTED_EXTRA_GATE：generic brainstorming的逐阶段design/spec人工审批及旧契约多Agent、逐文件hash／seal规则不进入当前白名单；按用户本轮持续研究授权和ExclusiveMinimalWorkflow执行。当前不增加部署分支、损失或头，也不因基础检查通过宣称Ag或联合成功。
