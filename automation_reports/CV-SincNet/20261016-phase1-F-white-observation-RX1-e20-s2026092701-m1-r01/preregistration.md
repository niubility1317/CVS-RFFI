# 预登记：F 有界接收观测噪声独立训练

状态REGISTERED_NOT_RUN。精确矩阵/命令/commit/env/CWD/paths/GPU/stop见experiment.json。

# F/E20：轻度有界观测噪声的独立 Ag 检验

上一轮 F raw＋全 ordinary 的首源折已按预定规则 FAIL，不调 raw 比例、不扩大该候选、不改用原 A1 主行。其关键证据是：full ordinary 与 raw 混合仍发生 TX 间错误迁移；移除 PA 的旧训练也失败。因此不能把 full ordinary 简单命名为身份保持泛化增强，也不能由冻结输入分数挑算子。

本轮独立假设：**只扰动接收观测的轻度噪声，可能抑制对源观测噪声细节的依赖，而不需要重新生成 TX 硬件非线性。** 这是待训练验证的假设。不是给 urban 降噪的 HQ，不复用 HQ 的 source70.97%作为训练收益，也不将噪声本身当成域泛化已有效。

唯一固定算子为 WHITE30：每个原 L 包采独立512维实高斯方向 z，n=z*sqrt(0.001*sum(x²)/sum(z²))，G=x+n。新增噪声能量严格按实际波形功率计，不用白输入代理；新增信号噪声比30dB（幅度L2约3.16%）。这是功率有界、期望各向同性的固定能量噪声，不是精确高斯AWGN，也不是原录音总SNR。30dB是预先固定的0.1%观测能量预算，位于旧普通增强已使用的20至36dB范围内；不读取目标、不扫描或依据结果调整强度/概率。全20轮固定，每step/view刷新独立方向，无标签、TX、RX身份种子。零功率包保留且不加噪。

不调用旧ordinary生成器，不加PA/DAC/IQ失衡/CFO/移位/多径/相位/量化/剪裁，不反演TX，不制造FIR延迟或边界。原IQ256全部观察区保留；原网络RMS、NoExplicitCFO和内部补偿保持。该阶段没有As，不把白噪声先叠到practical前；后续raw/G/As分别生成，组合顺序仍需独立消融。没有RX硬件参数随机化，接收噪声逐包刷新不等于硬件session变化。

## 最小真实训练

基座仍用户固定F三路/统一Sinc/joint160/defaultcos，全部参数按原流程联合训练，LS.01与RDrop1保持。新W30从零总E20=13L+7U，4440updates，L每曝光4前向，目标0.5RD(raw)+0.5RD(G)，每RD两前向复用同一IQ，0跨视图一致性。原U/ownEMA/旧门控/strong_noise_std/cos原样，不将新噪声用于U、V或测试。原始监督固定0.5。

只训练一个新源RX1模型，复用上一轮**A0 raw/raw**的匹配完整结果、配置、角色与实际L/U轨迹，不载入其任何模型、optimizer、EMA或统计。相同F初始化规则/seed2026092701/split392005：fitRX3/4/6/8 L5040/U45360/seenV21600，heldRX1 L/U不使用，held原V5400仅FINAL。A0正是相同四前向、同预算、同U原始监督控制；旧A1/A2仅保留历史，不作为晋级终点，不把本轮视为重新救援raw+wholeordinary。

唯一主比较 W30−A0：held accuracy≥+1pp，macroF1不下降，seenV均值损失≤0.5pp，有限完整预算与6TX×4RX×3day覆盖、全部实际L/U物理顺序相同。失败关闭WHITE30，不扫SNR/概率、换RX/seed/主终点；通过只允许另登记固定其余RX3/4/6/8的同算子匹配研究。五折后仍要求均值≥+1pp、至少4/5正、最差≥−1pp、mean macroF1≥0、每折seenV损失≤0.5pp，再独立固定Ag/As矩阵，不能从单fold进入target。

新成本20模型轮/4440更新/2273280 L前向；复用A0历史20轮/4440更新单列，0父权重成本。U/EMA/源eval/heldV/CPsmoke按实际计数。本轮source-only，无新target/星地成绩。正式联合模型仍须相同默认头、原clean+六practical、冻结prediction后truth-last，并证明clean正收益、lowurban>60%、suburban保持及跨seed；目标保持未完成。

本地实际图与唯一限定P0/P1审查均PASS，无直接缺陷；累计1024合成网络前向/1536噪声输入/2合成optimizer更新，两次各11负例，206初态张量与匹配F_A0构造相同。0realIQ/CP，未运行新真实训练。登记VALID、compilePASS；真实ownCP noquery smoke留在worker首步PASS立即继续，不增加额外审批。
