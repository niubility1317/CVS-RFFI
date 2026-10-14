# 有停止条件的身份主路径证据补偿：监督机会普查

状态 PLANNED，尚无真实源结果。先检验额外任务是否有足够可靠的学习对象，再决定是否执行梯度／有限步诊断。84% 默认头 clean 三种子目标仍未完成。SIG_TIME、固定 CENTER25K、COMPACT_LINEAR、LONG_SILU 历史正收益分别保留，不叠加不同配方数值。最近固定 PA 完整交换和源 RX 平均风险位移投影均为源 gate 0/8，原 E20 loss-gap 也未得到稳定正收益。

部署特征仍是 `z=((a+b)+c)+s`。同次前向直接捕获四个原分支：身份主路径 a、PA 增量 b、reference 增量 c、SIG_TIME 增量 s；不从 joint 相减推测。两个辅助视图分别保留一半 b 或 c，另一物理增量保持完整。所有辅助视图均 detach PA/reference 两端点、原 W 和三个融合 gain，仅 a/sRaw 接收辅助梯度；共享参数经 a/s 仍可能有梯度，不能说整个 PA 模块冻结。SELF 保留原加法顺序，须逐位还原实际 joint。

固定 beta=.5、pi=.9、coef=.1：`h(p)=relu(log(.9)-logp)`；EVIDENCE 用 `.1/(2B)*sum q*(h(PAweak)+h(REFweak))`，SELF 用两个同样 SELF 项与同分母。B 为完整 batch，不能由接受量或活跃量归一化。U 的身份目标和 q 完整复用原可信 teacher 链；不要求挑战后学生先预测一致，也不添加新 confidence gate。主 CE 对原默认头与骨干的正常联合训练保持；本次冻结普查不训练任何参数。

固定 BASE_CE 第一 seed 2026092701 的 final20，加载前核实实际源角色、scratch-only 祖先与 final-only source 选择。teacher 是这个学生的 deepcopy，未继承 EMA。两模型均 eval：这测固定起始证据分布，不能宣称与实际训练 train-dropout／当前 EMA 门控逐样本一致。

完整统计 V27000 和 U56700。U 原 native shuffle loader、batch256、末批124、222批保持，采样不读取 TX 元数据、不改 PAIRS。V 用固定随机序列 batch256（106批），只在所有视图／pseudo／q 已确定后将真实标签接入计数；不用于梯度或状态更新。种子为2026101501，V sampler 为2026101502。完整原门控为 rx/day quantile .86、阈值 [.92,.97]、批内物理邻近 window2/minconf .8、强视图Gaussian .015 一致，domain gate 关闭。随机批里的相邻物理样本稀少可能限制合法机会，科学失败后不偷偷改采样。

除完整／弱PA／弱reference的目标概率、停止活跃率和物理规模，还分别报告每个 RX/day、各 RX，以及接受子集和辅助活跃子集的源 V precision/errors。不能用总体高 teacher precision 替代困难活跃子集 precision；源 V precision 也不能代替隐藏 U precision，U 的真实精度和错误数均为 null。仅保存标量汇总，不保存逐包特征或 IQ。

固定 source gate：U 至少567个 EVIDENCE活跃样本（全U1%），其中至少100个相对SELF新增活跃；V EVIDENCE活跃至少100、其身份目标 precision >=.98；每个源 RX 至少10个活跃且 precision >=.95。该规则只筛查机会规模／源代理可靠性，不证明泛化收益。失败关闭这个固定入口，不扫描 pi、beta、门控或 sampler；通过后才另行预登记真实梯度、同状态 AdamW 和独立 source-L 的有限步诊断。

预算83700独立源物理包、167404 packet主网络前向（包含4个zero-IQ smoke），每次student前向三个原头readout，挑战没有额外IQ前向。实际 optimizer／临时optimizer／梯度／EMA／头拟合／target 均为0。入口真实 checkpoint 无 query smoke PASS 后立即继续全量普查。来源、角色、ID、状态不一致、非有限或技术失败保存产物并停止本 run；不修改其他健康任务。

run `20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01`，owner `codex/root/compensation-census-20261015`。实际 Git HEAD 发布时写入 `release_commit.txt`，父运行执行版本 `6693e4480ef990bc23c5021ff1440b547fe5fcaf`。普通 N607 账户、CVS-RFFI Python、CPU2线程／GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_compensation_sourceVU_20261015_r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_compensation_sourceVU_20261015_r01/experiments/cvs_evidence_compensation20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_compensation_sourceVU_20261015_r01/resolved_request.json`。唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01/source_diagnostic`；预期全量 V/U JSON、summary、独立进程／日志／产物读回和同row报告。完整记录见 [experiment.json](experiment.json)。

挑战特征和可靠弱视图监督已有 RSC、FixMatch 基础。本机制的研究点是实际四项部署结构下保留物理贡献、定向训练身份主路径，以及有停止条件的可信U补偿任务；具体创新性和效果仍待验证。[RSC](https://arxiv.org/abs/2007.02454)，[FixMatch](https://arxiv.org/abs/2001.07685)。
