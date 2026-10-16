# 普通 AWGN／multipath 的源因果对照

2026-10-17。当前固定源矩阵已执行并完整复算，联合移除主候选未通过 raw 身份保护条件，已关闭。基线保持 F＋跨 RX 同 TX，三 SIG_TIME、统一 Sinc、joint160、NoExplicitCFO、原 cosine 默认头；本轮冻结两份原 E20 final20 模型作源审计，新增训练预算为 0。旧 RMS、PA bypass、shift bypass 和其他固定失败继续关闭。

真实调用已经否定“普通信道再接完整 As”的说法：既有 As 读取 raw，不读取 ordinary。本问题是普通监督视图中的身份损伤，不是降低 practical 测试难度，也不把移除旧损伤称作新增 A_g。

原生顺序为 anti-shortcut／移位与幅相频率变化／multipath／AWGN／PA。有效 E13：multipath 的全 batch 门概率 0.0451080601，2 至 4 taps、延时 0 至 1；AWGN 的全 batch 门概率 0.1002401336，各包 SNR 20 至 36 dB、噪声方差按当前波形功率；PA 的逐包概率 0.03508404675。ordinary 出口没有统一 RMS 恢复。

四格为 full、no_AWGN、no_MP、no_both。原函数在每格仍执行并消耗相同随机抽样，移除项仅返回其输入；每格从同一 batch 起始 Python／Torch RNG 重置，核对参数、原语前后 RNG 和 PA 强度，最后只推进一份 full 的随机状态。MP 的 taps／delays／组合核从原语起始状态用私有 generator 重放记录，不消耗增强流。原生函数在每格结束恢复，不编辑 DataAugmentation。

主候选只固定为 no_both；两个单项移除仅用于归因，不能在主候选失败后挑一个晋级。保留全部其余旧操作、PA、原始包尺度，不恢复 RMS、不按身份／RX／day／置信度分支。局部 kernel／noise 生成成本单独计数。

移除 MP 后，后续 AWGN 使用相同标准随机噪声和 SNR，但按新波形功率重算 sigma；PA 作用在改变后的输入上。因此四格测量整条链路的总效应，不是固定加性噪声／固定 PA 输出的纯多径直接效应。源四格的交互项也不是最终 A_g／A_s 联合四格的交互项。

仅 fitL5040，RX3/4/6/8×day1/2/3×6TX，72 格各 70 包；匹配两个原合法 source-only scratch final20 B00 的训练角色。原缓存 L6300、未用源 V／历史特征会反序列化，RX1 原 L1260 先排除；U 仅角色 ID、无 IQ 或 TX 标签。不用 heldRX1／U／V／target 推断，不混同完整 L6300 或 L_train5580／L_dev720。

四固定增强流 202610171701 至 04、独立 shuffle 流 202610171710 至 13，与此前源普通流相同，不挑抽样；不是实际训练重放。每模型 raw5040＋四抽样×四格×5040＝85680 源 NN，两模型 171360 源 NN＋一次真实 CP 合成 2 包 smoke＝171362 NN；源 ordinary 生成 161280，另 canonical 合成 ordinary 320，私有 kernel 原语重放按实际记数。0 新 epoch／优化器／教师／头拟合／target；两参照历史 40 model-E／8880 更新／9373356 NN 单列。

两模型主候选均须：相对 full accuracy≥+0.2 pp、macro-F1 不降，至少 3/4 抽样正增益且每抽样≥−0.2 pp；相对 raw accuracy／F1≥−0.5 pp，raw 正确保留≥99%；每 RX／TX／day 相对 raw≥−1 pp、相对 full≥−0.5 pp。全条件在源访问前固定。科学 FAIL 仍完成两模型所有格，再关闭固定 joint ordinary AWGN／MP removal，不选单项、PA 条件、系数、clip、抽样、RX 或 seed 救跑。PASS 也只是源身份依据，后续同预算同前向 E20 训练须另登记，不能预告泛化或联合收益。

本地 17 输入／配置负例、7 科学反例通过；480 合成 ordinary、72 个私有 kernel 随机标量，0 源 IQ／CP／NN／更新。确认 full 对原生输出／RNG 精确保持，PA 掩码和原语参数匹配、AWGN 功率依赖、无信道时输入精确保持及原函数恢复。旧 Torch NumPy 桥警告未被调用，不装包。一次独立 P0／P1 审查及仅预检缺字段的定向复审 PASS。

完整联合目标仍 ACTIVE：有效新增 A_g、稳定 A_s、完整训练契约 B00／B10／B01／B11 与交互项、跨 seed、正式 clean＋六 practical、low urban>60%、suburban 距自身 clean≤2 pp 探索目标均未完成。


普通AWGN/multipath源审计 20261017-diagnostic-F-crossRX-ordinaryChannel-fitL-manysig-m2-r01 完整ANALYZED/VERIFIED：FAIL_FIXED_JOINT_ORDINARY_CHANNEL_REMOVAL_RAW_IDENTITY_GUARD_CLOSED；171360源判决/320逻辑batch全部复算，171362NN/0更新/target。联合移除比full+1.522817/+1.488095pp，但比raw−5.868056/−5.962302pp、raw正确保留94.102742/94.023251%，固定raw保护FAIL，关闭不启动E20，不选单项fallback。每模型MP/AWGN/双门batch=6/17/0，interaction不证明无物理交互；full和raw全部logits与既有RMS源研究对应原视图完全相同。As本来读raw，无同视图普通信道再叠As。下一孤立轻度noise(raw)候选尚未实现登记启动，不能称新增Ag有效；联合目标ACTIVE。
