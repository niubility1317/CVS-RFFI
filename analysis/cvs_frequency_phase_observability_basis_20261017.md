# 短窗频域相位：尚无源数据或增强有效性证据

2026-10-17。状态 SYNTHETIC_PASS／SOURCE_OBSERVABILITY_CLOSED。前项真实同包连续上下文入口已按现有构造和映射条件关闭；六scene均衡及旧幅度谱FIR、时域非仿射相位、周期残差相关路线保持关闭。当前没有新Ag算子、默认头实验、训练或目标访问。

## 依据与未知

[Analog Devices接收滤波说明](https://www.analog.com/en/resources/analog-dialogue/articles/2022/05/18/02/02/how-digitally-tunable-filters-enable-wideband-receiver-applications.html)说明频率相关群延迟可以改变宽带调制波形。这只支持考察接收观测中的频域相位，不能证明当前WiSig EQ1残留该分量，也不能证明phase-only随机化保持TX身份。[跨RX RFFI原文](https://arxiv.org/html/2411.03636v1)同时把滤波器列为TX指纹与RX污染来源；因此必须避免把所有相位响应都当接收机因素。本路线不采用该文的分支或附加解耦损失，不套用其性能数值。

实际旧experiments/cvs_spectral_iq16观察abs(FFT)^2的对数PSD/DCT；dsp.py以实指数生成频率增益，checks.py要求响应相位接近0。它检验的是幅度形状，不是非仿射频域相位。旧boundedphase是在时间点乘相位；新问题考察频率响应的相位变化，不能把旧源关联或身份保护结果直接迁移。当前未选择FIR/IIR、强度或边界方式，也不以更换旧Hann33／.5dB参数救跑。

## 唯一问题与最小检验

在现有256点合法L中，能否观测到跨TX/day仍稳定的RX关联非仿射频域相位，而不是TX滤波指纹、公共相位、整体时移、CFO或截断边界的假象？这是待检验推断，不是已有物理识别结论。

先完成一个固定有限窗口相位测度的代数与合成辨识检验，使用同样256点长度，显式区分公共旋转／线性频率相位（时间偏移）与时域CFO造成的频率平移；不得把去线性相位误称为去CFO。合成真值只作检验，不进入真实源估计器。若没有无需oracle且可区分这些因素的测度，就关闭该观察入口，不读源IQ或训练。测度与数值矩阵现已固定并登记：[252格合成预登记](../automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-frequencyPhase-synthetic-m1-r01/preregistration.md)。登记时19项输入负例与8项代数检查及一次限定P0/P1审查已通过；现完整含噪矩阵也已执行并独立复算，结果见文末。固定80:160／lag20自身CFO补偿／12谐波相位闭合量；3模板×7情况×3 SNR×4抽样，全部252格保留，任一固定标准失败即关闭此估计器，不重抽参数或种子救跑。

只有辨识基础成立，才另行固定fitL5040／RX3/4/6/8／day1/2/3／6TX的同时留出TX和day、错误RX映射控制等源统计；U/V/RX1/target不用于校准、选测度或选强度。统计不能以删掉不稳定RX、TX或包获取通过；若不具RX特异性、主要被TX／CFO／边界支配或任一必要源格不可辨识，则关闭固定依据问题。

即使源统计通过，也只能支持后续有界IQ算子与原默认头的身份保护检验；再通过后才固定匹配scratchE20。RX关联不等于纯硬件因果，数字变换不是更换真实RX，辅助量或身份保护不等于跨RX clean收益。Ag与As必须分别建立识别证据，组合与交互项独立验证。

模型及数据边界保持F＋跨RX同TX、三SIG_TIME、统一Sinc、joint160、NoExplicitCFO（必要内部补偿保留）、默认cosine头正常联合训练、E20、原隐藏U；不加部署分支或另训头。当前本项成本为0新IQ／CP／NN／更新，目标ACTIVE且尚无新Ag或稳定As证明。


## 完整合成结果

20261017-diagnostic-F-crossRX-frequencyPhase-synthetic-m1-r01 已完整ANALYZED／VERIFIED，252/252格及63/63组通过。共享RX相位最大相对RMSE3.999642%，最小风险改善98.092207%；TX曲率、RX幅度、交互、窗外边界全部不误检，缺频全部拒绝。全部保存均值／18折风险／63组独立复算一致，主体50.409012秒，0真实IQ/CP/NN/更新/target。只证实固定测度的合成基础，不证实真实RX残留、Ag或身份性能。下一步仅另行预登记fitL源观测；当前未读源或训练。完整报告见[同一行报告](../automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-frequencyPhase-synthetic-m1-r01/report.md)。


## 固定源观测登记

20261017-diagnostic-F-crossRX-frequencyPhase-fitL-manysig-m1-r01 已实现登记未启动，沿用原合成测度和数值条件；仅fitL5040全包，72格各70，任一必要包不可辨识则整体FAIL，不筛样本／不改阈值／不进E20。21协议负例和4科学用例及一次P0/P1通过。当前0真实源IQ已读／CP／NN／更新／增强，完整[源预登记](../automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-frequencyPhase-fitL-manysig-m1-r01/preregistration.md)保留。


频域相位源观测 20261017-diagnostic-F-crossRX-frequencyPhase-fitL-manysig-m1-r01 已完整ANALYZED／VERIFIED，FAIL_FIXED_SOURCE_PHASE_OBSERVABILITY_CLOSED：5040包全部观测，2492通过必要频点、2548未通过；RX3/4/6/8分别通过0/1248/1243/1。72格全留，22全通过／35全拒绝／15部分；共享RX折实际0，不筛包或只取RX4/6。全部5040观察质量与72格均值方差独立复算一致。1次cache／40320闭合量／1.396791秒，0CP/NN/更新/新视图/target。关闭当前源依据，不改测度/窗口/频点/质量/阈值/子集/seed救跑，不进增强/E20；不等于真实RX相位不存在。F＋跨RX同TX/E20/原U保持，联合目标ACTIVE。
