# 匹配 E30 No-As 参考

本参考在读取当前 As 三行正式目标成绩前固定。它补齐 clean 净收益所需的无新增星地增强基准，不新增研究机制，不按成绩晋级。

- Run：20261018-phase1-F-crossRX-matchedNoAs-e20plus10-s2026092701-m1-r01。参考对象：20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02。
- 数据：split392005，五源 RX [1,3,4,6,8]、三日，L6300/U56700 隐藏 TX/V27000；无 RX 留出、无 L_train/dev 划分，仅现有 256 点 IQ。复用既有 VALIDATED_ONCE 句柄。
- 模型：F＋跨 RX 同 TX SupCon；SIG_TIME 三路、统一 Sinc、joint160、默认 cosine 头联合训练。NoExplicitCFO。
- 初始化：原完整源域 REPEAT（No-As）固定 E20 student＋自身 EMA，再加 10 轮；重置 AdamW/cos10，固定最终轮。和 As 父模型的原始 seed、数据、预算与初始化规则匹配，父权重因增强处理而不同。不是从已失败候选继承。
- L 目标：0.5 ordinary RDrop＋0.5 同一已生成 ordinary IQ 的额外 RDrop＋0.1 SupCon。旧普通增强保留；无新增 Ag、无 As 生成。每个 RDrop 对内复用 IQ。U 和教师保持既有规则。
- 成本控制：每物理 L 四次前向，每步两次额外求导；MEASURE 不改梯度。2220 新更新、4440 额外 VJP；自身 E20 父成本和新增 E10、教师、评估、预测、墙钟分别报告。额外普通两前向补足 As 两前向的计算控制。总 L 损失质量 1；As 对照为 ordinary0.5＋As0.5。
- 结果：固定 E30 后默认头预测 clean＋六 practical，共 7×168000，全部预测固定后独立 truth-last；无额外头，无目标反馈，无自动扩种子或选择性重跑。

执行环境为 N607 普通账户，现有固定 F 依赖，FP32/CPU2。CWD 为 PROJECT/releases/cvs_F_matchedNoAs_E30_20261018_r01；命令为 CVS-RFFI/bin/python -u experiments/cvs_matched_noas30/runner.py。唯一 owner codex/root/F-matchedNoAs-E30，共享 GPU 每卡含其他 owner 最多 3 进程，空闲显存至少 12 GB。唯一不可覆盖输出为 PROJECT/runs/20261018-phase1-F-crossRX-matchedNoAs-e20plus10-s2026092701-m1-r01。实际代码身份为发布归档记录的 Git commit。

停止规则仅限输入/父来源/隐藏标签/非有限值/输出/预测等系统技术失败；保留产物和其他健康任务，不自动重启。无科学晋级门槛，因为这是必须报告的参考。

预期产物为实际配置、父来源/初始化、完整 L/U 序列及 source 日志、最终 checkpoint/source_frozen、七视图 prediction 完成、独立 metrics/correctness、同 row 成本和分析。

本地 16 个协议负例和 1024 个合成 F 前向、2 次合成更新通过；MEASURE 与原 No-As 梯度及更新逐位一致，RNG 一致，真实 IQ/权重/更新/目标访问均为 0。发布时执行唯一真实父 checkpoint 无 query smoke，PASS 后立即训练，不另设许可层。

最新状态：ANALYZED／VERIFIED。固定 E30、七视图预测及独立 truth-last 已完成。此参考本身不能证明 IQ Ag 有效，也不构成联合目标完成。

发布后早期状态（历史记录），VERIFIED：执行 commit 00cc9bad869d98bbca408858a4adec55887cf6c7。唯一发布归档传输比较、远端 compile 和合成检查通过；真实对应 REPEAT20 父 checkpoint 的 2 包零输入 no-query smoke PASS；独立核实原 dispatcher/worker PID、CWD、argv、GPU 和日志增长。已完整累计 E23（666 新更新），L 前向340992、As0/Ag0、五源 RX/U truth0。训练仍在进行，未完成预测或评分。

## 完整匹配结果

No-As clean85.245238%、mid urban37.279762%、low urban31.712500%。普通 As 为82.722619%、64.376786%、53.458333%；相对差为−2.522619pp／+27.097024pp／+21.745833pp。确认当前 As 独立 practical 适应收益，同时确认 clean 代价，完整联合目标未达到。固定梯度方案仍关闭，不用该对照挽救。

全部五源 RX/L6300/U56700-hidden/V27000、实际 2220 新更新／累计 E30；全部7×168000固定received IQ预测后独立评分。504CM／84分区／21新配对独立复算，额外分析0IQ/CP/NN/rawtruth/更新。No-As新增训练＋预测3770644NN；另有共享CP smoke2与开发合成2048NN，父E20训练3661082NN单列。

详见 [完整比较](final_analysis.md)、[分层指标](stratified_results.md)、[七场景图](figures/matched_view_results.png)。单一seed，无新增Ag、B10/B11或完整交互证明；不修改目标IQ、信道或困难样本，不追溯重跑已关闭路线。
