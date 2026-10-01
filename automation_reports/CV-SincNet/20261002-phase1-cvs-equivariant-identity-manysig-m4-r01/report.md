# CVS 全路径复相位等变记忆网络：源域预登记

状态PLANNED，尚未发布N607。目标为识别性能第一、参数轻量次要，以及有可检验约束的RFF physics aware。原物理L6300/V27000、U56700unused；普通身份CE、无增强、无域骨干；4个新scratch seed，E200×50、batch128、AdamW2e-4/wd1e-4/cosine1e-6，FP32无裁剪。202553个参数，原残差164225，增加约23.34%。所有参数在本地CE反传中获得梯度。

Sinc后的时间路径和received记忆路径均使用无bias复卷积、逐包各通道RMS及实数径向门控。三阶/五阶和延迟基底保留幅度相关复响应；功率、lag1/2复相关和位置功率读出形成不变表示。原rawFFT频谱、fusion和余弦头保留，未约束raw IQ旁路移除。全分类输入对公共常相位不变，仍保留相对相位/CFO响应。行为基底与滤波因果，逐包RMS/读出使用整包，整体不声明在线因果。不是TX系数辨识、任意信道或RX解耦。

冻结诊断沿用固定公共激励TX→信道→RX级联的5TX×6RX组合，再测试3个公共相位，报告全网logit和单位嵌入误差。TX/RX可形成相同received波形的反例保留。合成仅为机制诊断，不用于增强、排名或停止，不虚构身份准确率，不复现真实WiSig完整均衡器/噪声/瞬态。

源矩阵共8记录：4新＋4原residual_fusion源控制，只读原指标。原记录实际配置、完整物理角色、预算和scratch来源已实时核实，不加载权重、不读目标指标。固定四seed的0.5V+0.5最差源RX分数最高优先，完全并列后才比较V/最差RX、MAC/参数和固定顺序。详细文本日志、40000步与800轮结构化记录均保留。

新候选胜出后默认20261002-phase1-cvs-equivariant-clean-manysig-m24-r01：4新clean预测+20旧冻结预测复用，168000个原物理query、6TX、7RX，所有prediction固定后独立truth-last评分。原基线胜出时核实并保留历史测试，不重跑历史，不测未选新候选。目标成绩不反馈调参、选模、结构或重跑。clean为历史已暴露代理benchmark，范围不扩展到LEO/SFT/新类。

10项模型/数学检查、13项协议检查PASS；本地4个一次性CPU模型/12次CE更新PASS；模型/数学与源执行两项独立P0/P1审查PASS。本地实数Conv/Linear MAC为7199008/包，排除FFT与逐元素计算；CPU计时只作本地证据。远端Torch2.1烟测、实际GPU成本、真实源训练与新clean测试均N/A。未证明性能提升，真正RFF physics aware目标仍未完成。

[结构、数学与物理边界](../../../docs/CVS_EQUIVARIANT_IDENTITY_HYPOTHESIS_20261002.md) · [实时源控制核实](evidence/source_control_preflight.json) · [本地CPU烟测](evidence/local_cpu_smoke.json) · [本地profile](evidence/local_profile.json)。

## 远端启动独立核实

**VERIFIED / RUNNING**。不可变源release commit `555aaadea1ab5c40dab7f9d8a80a51b1cd7b5c76`，dispatcher PID `1247460`。独立读取4个worker的 `/proc` CWD/argv，均与release一致。远端CPU一次性4模型/12次CE更新PASS，未读取正式IQ、旧权重或target。

|seed|GPU|worker PID|已完成epoch|末轮CE|源V|最差源RX|
|---|---:|---:|---:|---:|---:|---:|
|2026092701|0|1247477|14|0.211953|93.5222%|84.4630%|
|2026092702|1|1247566|13|0.228014|94.2963%|86.7963%|
|2026092703|2|1247653|13|0.233416|91.6259%|84.2963%|
|2026092704|3|1247799|11|0.262908|92.5000%|79.3889%|

以上是读回时的中途进度，不是最终选模或性能提升结论。4行实际为202553参数、普通CE、L6300/V27000、每轮50步；全部参数末batch梯度被记录，源V只读。4行日志持续增长，健康训练不停止、重启或热改。真实GPU成本与完整物理结果待E200，新clean预测/评分为N/A。

条件clean执行链已补齐，重新核对4新+4旧源记录、scratch checkpoint完整payload与实际equivariant激活，只有实际source-selected新候选才允许query。57项条件clean检查PASS（既有52＋新5）；独立P0/P1审查PASS。完整源日志审计12项负测PASS，正式40000步审计尚待完成。后续collector/report控制面不修改本次远端源release。

[启动与实际进度](evidence/launch_readback.json) · [远端CPU烟测](evidence/remote_cpu_smoke.json)。下一步为E200完整日志与源指标审计、8记录冻结、新候选胜出后默认clean4新+20旧truth-last；原基线胜出则核实历史测试，未选新网络测试N/A。性能与真正RFF physics aware目标仍未完成。

## 本次控制面交付

完整源collector和分析实现独立P0/P1审查PASS（尚无正式E200完成数据），条件clean审查PASS；不改变正在运行的不可变源release。最新独立读回4行完成轮次为75, 74, 73, 72，日志增长、进程CWD/argv和actual config一致。等待自然训练完成后继续既定8源记录排名及默认测试分支，不重复启动。[当前读回](evidence/source_progress_readback.json) · [继续执行交接](evidence/current_handoff.json)。
