# 包内重复残差轻度收缩：固定源身份保护检验

状态：LOCAL_VERIFIED，尚未发布或读取真实源预测。联合目标仍 ACTIVE。

唯一假设是：对现有 256 点 IQ 的固定四周期窗口，保留同包周期均值，只将非周期部分轻度收缩，可以保留原默认头的 TX 身份决策。该必要条件通过后，仍须另做匹配 E30 训练才能检验 clean 泛化与 practical 收益。这里不把残差称为纯 RX 噪声，不宣称已经移除接收机响应。

[WiSig 原论文 IV-B](https://arxiv.org/pdf/2112.15363)说明均衡后重采样至 25 Msps 并提取 256 点；[Non-HT PPDU 文档](https://www.mathworks.com/help/wlan/gs/non-ht-ppdu-structure.html)给出 L-STF 的十个短周期总长 8 微秒。这给出名义 20 点周期的动机，不能保证每个当前窗口严格周期化。用户已明确没有可靠的原始长上下文映射，本候选不寻找或构造额外 IQ。

算子固定窗口 [80,160)、四个 20 点周期。P 为同一物理包四周期均值，D 为每周期相同的 sin(pi*t/19)^2 权重，首尾权重精确为 0。x'=x−λD(I−P)x，λ 仅比较 0 与 .1。DP=PD，因此精确算术中周期均值保持、总能量不增加，变化范数不超过输入范数的 10%。窗口外及每周期首末点逐位不变。无额外 RMS 归一化、循环拼接、补零、窗外输入、CFO 估计、RX/TX 拟合、PA/IQ 失衡/量化/剪裁或随机新噪声。该代数性质不能证明身份保持；非周期 TX 误差也可能被削弱。

区别于已关闭路线：cyclecov/周期投影是跨 TX/day 共享 RX 统计归因，本候选不拟合或借用其失败坐标；周期重排改变次序，本候选保持次序；酉相位保持能量并改变相位，本候选收缩非周期分量；同 TX 相干混合借用其他真实包，本候选只读自己一包。旧路线保持关闭，不更改旧阈值、窗口、频点或强度救跑。本轮也不重复另一对话的 E30 As 扩种子。

唯一登记和实际配置见 [experiment.json](experiment.json) 与 [算子配置](../../../configs/cvs_repeat_shrink_source_20261018.json)。原 fullsource REPEAT/CONCAT 的 source-final20 默认头仅作冻结读出，模型 seed2026092701，源轮数预先固定，未按 target 排名。加载前复用原完整来源核对，并实际检查全部物理角色与 scratch-final20 身份，无额外祖先/EMA/teacher。两模型已有 E20/4440 更新成本分别保留；本轮新训练为 0。F＋跨 RX 同 TX、SIG_TIME 三路、统一 Sinc、joint160、默认 6×160 cosine/scale30、NoExplicitCFO 与内部补偿均保留，不另训头。

原 L6300/U56700 TX 隐藏/V27000、RX[1,3,4,6,8]、day[1,2,3]、六 TX、split392005 均不变。只从原 V 各 TX×RX×day 格按 opaque ID 固定取前 20 包，共 1800；全部 90 格参与，无置信过滤、无留出训练 RX。V 仅推理与评分，不更新任何持久状态。原 source_cache 容器中的未使用特征随反序列化加载后忽略，U 不读取。原 L/U/V 物理角色复用 VALIDATED_ONCE，不重建数据。

最小矩阵：2 个原默认头 × raw/一次生成的旧 ordinary × λ0/.1 ×1800＝14400 源预测，另 2 CP×2 合成 no-query smoke＝4 NN，总 14404 NN。old ordinary 按原实际 args/E13 日程及私有 seed20261018101 生成一次，固定 Python/Torch RNG 并恢复，全部模型/强度共用；其 PA/CFO 等开关据实保存，不能把这套共有增强称为已验证的新 Ag。RDrop 训练未在本轮执行。7200 次逐包算子调用中 3600 次非零，1800 次 ordinary，0 backward/optimizer/teacher/头拟合/target；CPU 数值计算与 GPU 前向成本分别记录。

必要反证全部固定：每默认头×base 的 .1 相对 0 准确率不得低于 −.5 个百分点，原正确保持率至少 99.5%，每 RX/每 TX 准确率差不得低于 −1 个百分点。数值条件：能量相对增加≤1e−6、周期均值变化相对输入范数≤1e−6、总变化≤.100001，窗口外与各周期首末点完全不变。任一失败即关闭固定 .1 强度/[80,160)/period20/taper，不换强度、窗口、周期、taper、seed、父权重或预算救跑。PASS 仅为源保护，不是 DG/As/联合成功或自动扩种子/target。

聚焦本地检验：21 个协议/输入负例 PASS；18 个合成输入、独立 80×80 稠密投影对照、谱范围、周期均值/能量/边界、零强度、包排列与复线性验证 PASS；FP64 最大稠密误差 4.45e−16，FP32 9.79e−8。0 真实 IQ/CP/NN/更新。真实 CP no-query smoke 是实际 worker 首个模型动作，PASS 立即继续。按项目白名单只做候选一次 P0/P1；不增加设计审批/报告签名/额外数据验证。

发布 CWD PROJECT/releases/cvs_F_repeat_shrink_source_20261018_r01，命令 CVS-RFFI/bin/python -u experiments/cvs_repeat_shrink18/runner.py。本地发布入口 F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_repeat_shrink18.publish。一个 Git commit 固定代码/配置，执行 commit 写入 release_commit 与运行产物；不另要求设计/报告哈希。唯一不可覆盖输出 PROJECT/runs/20261018-diagnostic-F-repeatShrink-sourceV-s2026092701-m1-r01，owner codex/root/F-repeat-shrink-source。既有普通 N607 账户、CPU2/一个共享容量 GPU，全 owner 每 GPU 最多 3 实验进程且至少 12 GB 空闲。一次普通预检、一份 archive/一次传输 SHA 比对、一次远端编译、一次所属 PID/CWD/argv/GPU/日志增长核验；健康旧任务不干预。

技术停止仅角色/父来源/非有限/形状/输出冲突/执行失败，保存 failure 和全部原产物，无自动重试。低准确率为科学失败。预期 real_checkpoint_smoke、resolved_config、私有全部源 logits/ID、prediction_complete、全部 840 混淆矩阵和配对/条件、completion/dispatcher_complete；独立复算不再读 IQ/CP 或拟合头。源 V 标签允许读出，不冒称 target truth-last。测试 IQ/SNR/信道和困难样本均未改动，尚无新 target 结果。

当前缺项：有效新 Ag、clean 净正、As 跨种子保持、匹配四格/交互项、low urban>60% 和 suburban 各场景接近自身 clean（本路线探索目标≤2pp）。不能用保护检查、辅助量或单种子局部变化代替真实身份收益。
