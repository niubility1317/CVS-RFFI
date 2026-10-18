# F 默认头输入敏感度：源 L 排序与源 V 等量扰动反证

状态：已实现、待发布；没有新训练结果，没有 A_g 或 A_s 效能结论。Run：20261018-diagnostic-F-identitySensitivity-sourceLtoV-s2026092701-m1-r01。

用户最新约束保持 F＋跨 RX 同 TX 对比学习、E30、全部源 RX1/3/4/6/8、day1/2/3、原始 L6300/U56700 隐藏 TX/V27000、split392005，以及现有 256 点 IQ。没有原录制或更长上下文映射；本轮不提出切片扩展。诊断不留出训练 RX，不建立 L5580/dev720。

## 唯一假设与范围

固定默认头在合法源 L 上的输入方向敏感度，能预测在独立源 V 上较少损害身份判断的时间位置。先检验这个依据，不把非零梯度、源保护或较小扰动本身解释为跨 RX 泛化收益。模型冻结仅服务本次只读诊断；未来训练仍按原流程联合训练默认头和骨干，不冻结骨干另训头。

[Cross-domain, Scalable, and Interpretable RF Device Fingerprinting](https://www.eng.auburn.edu/~szm0001/papers/infocom24.pdf) 提供保留身份重要 IQ 区域的动机，但使用目标 support、LIME 线性替代模型和少样本原型适配。本轮不采用这些权限或结构，也不把论文结果当作当前 Phase1 泛化证据。本轮用原默认头直接输入导数，不拟合额外分类器。

已经核对实际 CROSS_RX SupCon：同 TX 同 RX 从分母排除，不存在把这种已知同身份样本作为负样本的问题，无需新实验。场景内质量 CE 权重与已关闭的可恢复性课程在风险重加权上重叠，未作为新路线发布。已失败的固定频率掩码、循环重排、局部周期收缩等结果保留；本轮有独立位置排序、等扰动量和反证条件，不以更名重开旧配方。

## 输入、读出与矩阵

读出为原完整源训练的 REPEAT（无 As）和 CONCAT（普通拼接 As）final20 model/default cosine6×160、scale30，来自 20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02；原始 epoch20、4440 次更新、scratch 继承和固定 final_only 在加载前核实。不能使用目标挑选子权重。保留 F/SIG_TIME×3、统一 Sinc、joint160、NoExplicitCFO 和必要内部逐包补偿。检查点只读，不建立教师、优化器或新头。

复用原 source_cache.pt 和既有 source_contract、native_role_comparison。只访问 labeled、validation 容器。L：90 个 TX×RX×day 单元各取 opaque ID 排序前 2 包，共 180 包。V：每单元前 20 包，共 1800 包。U 容器和 TX 真值不访问。完整 L/U/V 训练契约不变；两个确定性子集只用于诊断。

1. 每个真实 checkpoint 的首次模型动作是 2 包确定性非零合成 IQ 的无 query smoke。REPEAT 同时检查输入导数是否可获得，不能把导数存在当作成功。
2. REPEAT 在 L180 的 true-class 减最高其他类 cosine logit、除以 30 的 margin 上，仅对输入求导；模型参数全部 requires_grad=False，结束逐项验证状态未变、参数无梯度。
3. 256 点分为 16 个不重叠 16 点区域。区域内窗为 sin²(πt/15)，首尾精确为 0。每包每区域计算绝对梯度方向量：|〈∇margin,w_b x〉|/||w_b x||；再在包内对 16 个值归一化。零梯度包保留，使用均匀分数并记录数量。每个 L 单元取两包均值，区域分数取 90 单元最大值。固定最小 4 区域和最大 4 区域，分数相同按索引稳定排序；同一张图用于所有身份和 RX。
4. 映射先由 L 冻结。V 全部 1800 包在 raw 和 16 区域×正负两符号的 33 视图上，以两原始默认头完整读出，共 66 组。V 标签仅评分，不选位置、幅度、父权重或阈值。

## 等量轻度变化

变化是同 I/Q 的实数正增益：x'=x+s·α_ib·w_b·x，s∈{-1,+1}，每点增益介于 0.9 至 1.1。令 E_ib=||w_b x_i||、N_i=||x_i||，δ_i=min(0.01,0.1·min_b E_ib/max(N_i,10^-24))，α_ib=δ_i N_i/max(E_ib,10^-24)。同一包的全部区域/符号具有相同相对 IQ L2 变化量 δ_i，排除低能量位置的改动更小这一混淆。零能量或弱区域不删包，必要时全部变换为恒等，明确记录。

不新增噪声、深衰落、CFO、相位、PA、剪裁、量化、时间移位或额外 RMS；窗口外及窗端点不变。数字变换已接收 IQ 不代表替换真实接收机。本轮不调用 practical，不改变任何测试信道或 received IQ。

## 固定判据和放弃条件

对每个默认头、低敏感度 4 区域、两个符号，准确率相对 raw 不低于 -0.5 个百分点，原正确包保留率至少 99.5%，各 RX 与各 TX 准确率变化不低于 -1 个百分点。除身份保护外，还要求排序有可识别的信息：每个头全部 32 扰动视图累计新增错误至少 20 次；低 4 区域的新错误率不超过全部区域均匀率的 0.75，且不超过高 4 区域率的 0.5。分母按每个头原正确包数×视图数明确计，不用错误数直接比较不同视图数。

L 上高 4 平均分至少为低 4 的 1.5 倍且非零。全部数值检查满足增益界、窗外不变、实际相对 IQ 步长≤0.01、与共同步长绝对差≤10^-6。导数不可获得、所有位置几乎无可见身份影响，或任何固定条件失败，均关闭本次梯度规则/单元最大分数/16点网格/0.1增益/0.01步长/两父权重/seed；不扫描区域、网格、幅度、分数、父权重或 seed 补救。

通过只支持源域扰动排序依据；不能自动启动 E30、扩种子或目标测试。A_g clean 净收益、A_s practical 收益、组合交互和跨种子复现均未由本轮检验。

## 发布、成本与产物

代码/配置身份：本包一次实际 Git commit；发布 HEAD 记录在 release_commit.txt、submit 和最终报告，不要求额外成员 SHA。登记：experiment.json。命令：CVS-RFFI/bin/python -u experiments/cvs_identity_sensitivity18/runner.py。环境：N607 普通账号、既有固定 fullsource F runtime、FP32、2 个 CPU 线程；CWD：PROJECT/releases/cvs_F_identity_sensitivity_source_20261018_r01。唯一 launch owner：codex/root/F-identity-sensitivity-source。一个 CPU dispatcher 等待共享容量 GPU（计所有 owner，每 GPU 最多 3 进程且空闲至少 12 GB），一个 worker 读两个头。不可变输出：PROJECT/runs/20261018-diagnostic-F-identitySensitivity-sourceLtoV-s2026092701-m1-r01。

预算：2 次 CP 加载、1 次完整原 source cache 反序列化（不访问 U 容器）、1980 个独立源 IQ 包实际进入模型；smoke4＋L180＋V1800×33×2=118984 次包前向；输入求导最多 182 包、3 次 backward tensor 调用（其中 smoke2，L180）。57600 次 IQ 变换生成，在两头之间共享；新增参数更新、headfit、教师、普通增强生成、As 生成、target 均为 0。实际成本和 wall time 分别记录，不能冒称 E30 模型结果。

发布前执行窄范围负例/数值测试和一次独立 P0/P1 检查；一份 archive/一次本地远端 SHA 比较/一次远端 compile。smoke 是 worker 首步并立即继续，不设 token。启动后独立核实 PID/CWD/命令/GPU/日志增长。只停止本 run 的系统性技术失败；数据权限、parent、非有限数、输出冲突或实际预算不符时保留产物、失败不自动重跑；科学判据失败正常完成并关闭固定假设。健康旧任务不动。

预期私有产物：real_checkpoint_smoke.json、source_L_ranking.json、resolved_config.json、predictions.json、prediction_complete.json、worker 全日志。独立 CPU scorer 只在全部固定预测完成后读取保存的源标签/导数/logits，输出 result.json、completion.json、各 RX/TX/day/90 单元混淆矩阵和成对修复/新增错误。报告只入库汇总，不提交物理 ID、IQ、模型、logits 或本地/远端私有运行信息。最后对所有保存读出独立重算，不用局部日志替代完成证据。
