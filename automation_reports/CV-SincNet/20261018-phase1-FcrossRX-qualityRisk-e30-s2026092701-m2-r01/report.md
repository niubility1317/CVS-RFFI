# 完整 practical：按实际退化条件分层的有限风险 E30

PLANNED，两个匹配尾段实现中，尚无训练或性能结果。唯一假设是：保持同一个有限 complete-As proposal bank 的名义风险和 As 总系数，通过每批覆盖四个实际质量组，改善低质量身份识别。先检验抽样组织，不加入困难风险强化、U 路由、公共传播、Dropout 或新损失。

## 固定矩阵与风险

新 I01（BANK_IID30）与 S01（BANK_STRAT30）均从原 P43 B01 ownAs20 FINAL student 与 ownEMA 独立追加 10 轮，2220 更新，总 E30/6660。旧 AS30 为同一 As20 的原在线尾段，旧 NOAS30 来自其自身 NoAs20 对应轨迹；两者冻结读出复用。不能将 NoAs 写成共享 As 父。逻辑四行、clean＋六 practical，各 view168000；独立重计预计 504 CM/84 partitions。

这里的 raw 是现有 native 已中心裁剪并 RMS 归一化的 256 点 IQ，不是完整录制波形；进入完整原引擎时保留其原 normalize_input，缓存后的接收 IQ 不再二次归一化。

共享 bank 仅用完整 L6300，每个物理样本 4 replicas×10 个累计 E21 至 E30 合法 context，共 252000 次完整原引擎传播。IQ float32(2,256)为 516096000 bytes，metadata 另计。一次生成后原样复用；不按预测、质量、遮挡或锁定拒绝重抽。全部场景 IID 原先验，完整 propagation→N0/noise→quality/lock/tracking→compensation/AGC 依赖保留。signal_to_added_noise_db 是新增噪声比值，不是真实总 SNR。

四组固定为 <3、[3,8)、[8,15)、≥15 dB；p_g=N_g/N_bank 只来自训练 bank，不能用源 V 面板或目标统计代替。I01 均匀有放回抽 bank64，w=1；S01 每组均匀有放回抽 16，w=4p_g。两个 CE 和原同 IQ symmetric KL 全部按同一逐包权重计算：mixed=.5/128×(ordinary64 项之和＋bank64 加权项之和)。S01 每批 As 权重和恒为 64；不截权、不逐批自归一化。任何非空质量组都保留；预定 bank 若存在空组则技术失败保留产物，不临时改 bin 或删除难包。

I 与 S 对同一有限经验风险无偏（在 clipping/AdamW 之前的梯度期望）；不能声称严格等于连续原在线 As 风险，也不声称整条 U/优化轨迹相同。旧 live AS30 用于单列 bank/global sampling 与有限缓存复用的影响。原 L128 的 selected64 位置保持交替；bank 自己的合法 L 标签、RX/day/domain/PID 一同替换 mixed 槽，ordinary64 补集使用当前 L 标签。primary ordinary128、raw128 与 ORD128 SupCon 使用当前 L，保持 6L/768NN、总 L1.25/As.25/globalLS.01。没有 BatchNorm；mixed bank 不进入 ordinary SupCon 的跨样本标准化。

## 数据、模型、调用与来源

源 RX[1,3,4,6,8]/days[1,2,3]，完整 L6300/U56700 隐藏 TX/V27000，split392005、eq1、IQ256/Fs25MHz；无 RX 留出，无 L_train/L_dev。沿用 VALIDATED_ONCE 的实际角色与已核实父来源，只查当前最小数据句柄。V 不参与训练、bank、EMA 或任何状态更新。正式 RX[0,2,5,7,9,10,11]/days[0,1,2,3]、固定全部 received IQ/SNR/信道/困难样本保持。

模型保持 F 三路 SIG_TIME、统一 Sinc、joint160、默认 cosine6×160 scale30 随骨干联合训练，无诊断头/部署分支/替换头。保留 NoExplicitCFO 及必要内部补偿。普通增强及原 U weak/strong/资格/teacher/EMA .999 和 lambda 全部保留；bank 仅替换 L-As 的采样来源，不读取 U 真值。RDrop 两次前向始终复用同一个已生成 IQ。

父 run20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01；实际 releasecommit19aedc3604fed9c193237dd92366164178af0983。严格原 B01 own20 scratch4440/source 固定轮次，own20 args/frozen13、channel累计+20/+4440、SupCon+20、原 resetAdamW lr2e-4/WD1e-4/cos10/min1e-6/nativephase0/10。不使用 AS_CE20、LRaw、AsLS、NoU 等变体父。

历史去重：旧 recoverable 课程按 mid/low、locked/degraded、3至15 dB 选择有限提案，改变名义风险且没有 importance 校正，源失败关闭。本项不重新启动该恢复性筛选。sceneBalance 改场景、LHS 改阴影组织、Asweight 改总体份额、TX×RX AsAlloc 改当前 batch 的分配，均有已有结果；本项不扫描这些参数。限定检索未发现同四质量箱＋importance 保持有限风险的 L 训练，不宣称全库穷尽。连续 E30 修复 As 现已完成且源原判据失败，附件中技术未完成描述为旧状态。另一对话 optimizer-state 保留/双独立 As 已执行，不重复。

## 冻结、评价与放弃

固定 FINAL30。预登记的下一阶段科学资格只用源：固定 V1800、两 draw、mid/low urban 等场景平均的 <8dB accuracy，S−I≥+.5 个百分点；≥8dB accuracy，S−I≥−.2 个百分点；fullV27000 clean，S−旧 liveAs≥−.1 个百分点。三项合取；源分组是当前来源的诊断，不精确预测未见 RX clean。未通过则关闭这一固定分层方案，不扫 bin/配额/权重/父/seed，不启动 M3/E50。源通过才可另行登记有界困难风险 M3；本 run 不自动追加。

科学负结果不停止正式收尾。每 row 冻结全部 7 prediction 后独立 exec truth-last scorer；保存源决定再读取本轮新正式结果，正式不反馈参数、seed、候选晋级或选择性重跑。已有暴露 benchmark 的背景保留，不称新独立盲测。正式阶段条件为 clean 相对匹配 NOAS30 下降≤0.1 个绝对百分点；n1 或 <.1pp 微增不能宣称稳定双提升。

报告 all7 accuracy/macroF1/最差 RX、urban3 均值、RX/day/TX、修复/新增错误、各 suburban 与自身 clean 差距；分别列 S−I（分层增量）、I−liveAs（缓存/global sampling）、S−liveAs（总机制变化）、As−NoAs（基础适应）。源实际 addednoise 四组/state/lock 的 count/accuracy/F1/已有 logits 真类间隔均如实报告；正式没有逐包状态标签时，不将源状态映射到目标作因果归因。

## 成本、环境与发布

新两条各 10modelE/2220updates，共20modelE/4440。预计每 tail native3162962NN=学生train2271960＋eval324002＋teacher567000，source12600/formal1176002，共4351564；共享一次真实首步 smoke770（2真实父读出＋768实际6L hook），总新增8703898NN，以实数为准。原20与旧30读出单列复用，不重复收费。bank252000次完整传播、simulation墙钟、516MB＋metadata/IO/copy、采样与逐包 CE/KL 计算另外记录，不把零新增 NN 说成零成本。完整2220步日志和全bank/实际曝光分开保留。

实际 code/config commit 待生成。唯一 owner codex/root/practical-qualityRisk；不可变 output=PROJECT/runs/20261018-phase1-FcrossRX-qualityRisk-e30-s2026092701-m2-r01；CWD=PROJECT/releases/cvs_FcrossRX_qualityRisk_E30_20261018_r01；N607 普通账号、现有 CVS-RFFI/PyTorch fullFP32、CPU2、无安装。发布 verified native Python -m experiments.cvs_quality_risk30.publish；worker CVS-RFFI/python -u experiments/cvs_quality_risk30/runner.py --mode worker --row {I01,S01}-s2026092701。private runtime 不入 Git。每 GPU 全 owner compute≤3/free≥12000MB，动态队列，不动健康任务。

聚焦 protocol-negative/权重公式检查、一次独立 P0/P1 合并审查；launcher 首步真实 CP no-query smoke PASS 直接训练。一次 preflight、单 archive 比较/远端 compile、一次 post PID/CWD/cmdline/GPU/log 增长。只因 owned 系统性技术、角色、父、头、数值、输出或无法合法预测失败停止，保留产物，不科学负结果中止。无额外审批、签名、逐文件哈希链。

预期 proposal_bank/{complete.json,iq.npy,metadata.npz,physics.jsonl.gz} 及每 row source/{bank_steps.jsonl.gz,bank_exposure.npz,result.json}，完整 IQ/metadata（远端保留，不入 Git）、每 source CP/frozen/config/actualargs/initialization/EMA/optimizer/reference与quality-risk日志、source qualification/all7 prediction/scoring/complete、0NN source/formal/fulltraining/cost/complete；源、final 汇总分别一次 archive 取回，同 row 分析/登记/Git 收尾。
