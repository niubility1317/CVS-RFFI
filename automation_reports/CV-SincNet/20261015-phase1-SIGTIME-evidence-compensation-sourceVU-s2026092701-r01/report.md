# 有停止条件的身份主路径证据补偿：监督机会普查

当前状态 **ANALYZED / VERIFIED，固定普查未通过**。以下保留运行前固定方案，完整结果在文末。84% 默认头 clean 三种子目标仍未完成。SIG_TIME、固定 CENTER25K、COMPACT_LINEAR、LONG_SILU 历史正收益分别保留，不叠加不同配方数值。最近固定 PA 完整交换和源 RX 平均风险位移投影均为源 gate 0/8，原 E20 loss-gap 也未得到稳定正收益。

部署特征仍是 `z=((a+b)+c)+s`。同次前向直接捕获四个原分支：身份主路径 a、PA 增量 b、reference 增量 c、SIG_TIME 增量 s；不从 joint 相减推测。两个辅助视图分别保留一半 b 或 c，另一物理增量保持完整。所有辅助视图均 detach PA/reference 两端点、原 W 和三个融合 gain，仅 a/sRaw 接收辅助梯度；共享参数经 a/s 仍可能有梯度，不能说整个 PA 模块冻结。SELF 保留原加法顺序，须逐位还原实际 joint。

固定 beta=.5、pi=.9、coef=.1：`h(p)=relu(log(.9)-logp)`；EVIDENCE 用 `.1/(2B)*sum q*(h(PAweak)+h(REFweak))`，SELF 用两个同样 SELF 项与同分母。B 为完整 batch，不能由接受量或活跃量归一化。U 的身份目标和 q 完整复用原可信 teacher 链；不要求挑战后学生先预测一致，也不添加新 confidence gate。主 CE 对原默认头与骨干的正常联合训练保持；本次冻结普查不训练任何参数。

固定 BASE_CE 第一 seed 2026092701 的 final20，加载前核实实际源角色、scratch-only 祖先与 final-only source 选择。teacher 是这个学生的 deepcopy，未继承 EMA。两模型均 eval：这测固定起始证据分布，不能宣称与实际训练 train-dropout／当前 EMA 门控逐样本一致。

完整统计 V27000 和 U56700。U 原 native shuffle loader、batch256、末批124、222批保持，采样不读取 TX 元数据、不改 PAIRS。V 用固定随机序列 batch256（106批），只在所有视图／pseudo／q 已确定后将真实标签接入计数；不用于梯度或状态更新。种子为2026101501，V sampler 为2026101502。完整原门控为 rx/day quantile .86、阈值 [.92,.97]、批内物理邻近 window2/minconf .8、强视图Gaussian .015 一致，domain gate 关闭。随机批里的相邻物理样本稀少可能限制合法机会，科学失败后不偷偷改采样。

除完整／弱PA／弱reference的目标概率、停止活跃率和物理规模，还分别报告每个 RX/day、各 RX，以及接受子集和辅助活跃子集的源 V precision/errors。不能用总体高 teacher precision 替代困难活跃子集 precision；源 V precision 也不能代替隐藏 U precision，U 的真实精度和错误数均为 null。仅保存标量汇总，不保存逐包特征或 IQ。

固定 source gate：U 至少567个 EVIDENCE活跃样本（全U1%），其中至少100个相对SELF新增活跃；V EVIDENCE活跃至少100、其身份目标 precision >=.98；每个源 RX 至少10个活跃且 precision >=.95。该规则只筛查机会规模／源代理可靠性，不证明泛化收益。失败关闭这个固定入口，不扫描 pi、beta、门控或 sampler；通过后才另行预登记真实梯度、同状态 AdamW 和独立 source-L 的有限步诊断。

预算83700独立源物理包、167404 packet主网络前向（包含4个zero-IQ smoke），每次student前向三个原头readout，挑战没有额外IQ前向。实际 optimizer／临时optimizer／梯度／EMA／头拟合／target 均为0。入口真实 checkpoint 无 query smoke PASS 后立即继续全量普查。来源、角色、ID、状态不一致、非有限或技术失败保存产物并停止本 run；不修改其他健康任务。

run `20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01`，owner `codex/root/compensation-census-20261015`。实际 Git HEAD 发布时写入 `release_commit.txt`，父运行执行版本 `6693e4480ef990bc23c5021ff1440b547fe5fcaf`。普通 N607 账户、CVS-RFFI Python、CPU2线程／GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_compensation_sourceVU_20261015_r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_compensation_sourceVU_20261015_r01/experiments/cvs_evidence_compensation20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_compensation_sourceVU_20261015_r01/resolved_request.json`。唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01/source_diagnostic`；预期全量 V/U JSON、summary、独立进程／日志／产物读回和同row报告。完整记录见 [experiment.json](experiment.json)。

挑战特征和可靠弱视图监督已有 RSC、FixMatch 基础。本机制的研究点是实际四项部署结构下保留物理贡献、定向训练身份主路径，以及有停止条件的可信U补偿任务；具体创新性和效果仍待验证。[RSC](https://arxiv.org/abs/2007.02454)，[FixMatch](https://arxiv.org/abs/2001.07685)。


## 完整源结果

**ANALYZED / VERIFIED**，固定 source gate 未通过。执行 `ee303b067ebd614986bc5cd59e2d634935979394`，归档传输、远端编译、真实 checkpoint 无 query smoke、进程／CWD／命令／GPU／日志增长及独立终态读回均核实；进程自然退出、无 failure。完整两源角色和全部 RX/day 均保留，未根据中间统计更改规则。

| 源角色 | 全量 | 原 q 接受 | SELF 活跃 | PA减弱活跃 | ref减弱活跃 | EVIDENCE 活跃 | 相对 SELF 新增 | 接受目标精度 | 活跃目标精度 | 活跃错误 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V_proxy | 27000 | 880 | 0 | 3 | 2 | 4 | 4 | 100.0000% | 100.0000% | 0 |
| U_s | 56700 | 934 | 0 | 2 | 1 | 3 | 3 | 隐藏／不可推断 | 隐藏／不可推断 | 隐藏 |

各原门控 mask 总量：

- V_proxy：confidence=25468, temporal=907, agreement=26923, base=880, accepted=880, structural=971, all=27000。
- U_s：confidence=53475, temporal=951, agreement=56499, base=934, accepted=934, structural=1015, all=56700。

固定不通过条件：U_EVIDENCE_active、U_new_vs_SELF、V_EVIDENCE_active、V_RX1_active、V_RX1_precision、V_RX3_active、V_RX4_active、V_RX6_active、V_RX6_precision、V_RX8_active。

V 表中的 precision 是教师 pseudo 身份与真实源 V 身份的一致率，所有视图／原 q／活跃决策完成后才计算。困难活跃子集与总体接受子集分开；高总体精度不保证困难子集可靠。仅4个活跃V包正确，不足以支持稳定的98%精度声明；RX1/6没有活跃包，其precision为null。U 的 trueprecision/error 一直为 null，源 V 不能证明 U 正确率。SELF 和两挑战视图目标概率是同一个教师 pseudo，不是挑战后重新标注。

实际预算83700独立源物理包、167404 packet 主网络前向（含4zero-IQ smoke）、106个V批和222个U批，CPU2线程／GPU0，source进程运行约116.29 s。参数／buffer／梯度／eval flags 和普查外 RNG 不变；真实／临时 optimizer、EMA、头拟合、target 均为0。冻结 eval／BASE_CE-copy 教师不能代表训练中 train-dropout／当前 EMA 的逐样本 mask，也没有中期状态证据。默认头结构未改，无新预测／评分或 clean性能数字。

按预登记关闭**这个原可信 U 入口、固定 beta=.5/pi=.9 的证据补偿版本**，不提高 pi、放大 beta、改门控或重排 sampler 来满足阈值。这个结果不证明整个训练期、所有物理挑战、源 L 真标签训练或其他结构均无效；它明确当前固定状态／入口不足以支持计划的下一梯度诊断。保留实现与负结果，不发布该原配置的 E20+20 主训练。后续研究仍需从合法 L 真实身份产生不同的监督内容或检验源域训练条件，而不是将本 census 失败归为算法有效或宽泛的泛化上限。

SIG_TIME、固定 CENTER25K、COMPACT_LINEAR、LONG_SILU 既往正收益继续分别保留；不同预算／配方不能叠加或用目标最高分挑选父权重。84% 默认头 clean 三seed目标仍 active、未达成。

全42行角色／RX／RX-day统计见 [机会分布](source_opportunity.csv)，全部批与概率标量见 [完整证据](result_evidence.json)，运行核验见 [终态证据](launch_evidence.json)。不可变原输出和原日志保留。

下一独立路线已写为[跨RX类间特征差方向学习](../../../analysis/phase1_crossrx_identity_difference_20261015.md)：用源L真实身份构造同RX类间差，再跨RX比较方向；保留真实四项部署特征，辅助梯度定向a/s，使用有限方向停止。此路线尚未实现或发布源run，不冒充本普查晋级，不依赖提高本次beta/pi/原q。有监督对比／类条件DG已有文献基础，具体新增区别和效果仍待独立实验检验。
