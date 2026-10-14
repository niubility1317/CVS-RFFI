# 源 RX 身份风险约束下的 AdamW 真实特征位移

本轮检验一个独立训练机制：默认头继续由原主 CE 联合学习，额外约束只修正实际部署特征参数的 AdamW 位移。最近 PA 完整交换源诊断 0/8 达标，原 E20 loss-gap 三种子对照亦未确认正收益，均按既定规则关闭。该源证据提示必须区分非零梯度、真实位移、特征变化和身份收益，不能用放大损失替代不同训练方向。

RAW 是当前 L90 的原 CE→clip5→AdamW。GRAD_CONE 先将同一 CE 的 feature 下降梯度投影到五个源 RX 风险半空间，再用原 AdamW。UPDATE_CONE 则投影 RAW 真正的 FP32 参数前后差。目标是最小欧氏修正，使 `g_RX·delta<=0`；欧氏距离仅定义在当前固定 CENTER25K 坐标，不声称单位不变。W、PA/reference/SIG 三增益不属于投影范围；原主 CE 仍保留其梯度。

投影类方法已有 PCGrad 和 CAGrad。这里的具体研究区别是约束**真实 AdamW 提议位移**，而不是假设梯度满足条件后，动量／裁剪／weight decay 的实际更新也会满足条件。不能宣称投影概念原创或已证明泛化收益。[PCGrad](https://arxiv.org/abs/2001.06782)，[CAGrad](https://arxiv.org/abs/2110.14048)。

固定矩阵为自有 BASE_CE 第一 seed 2026092701 的 INITIAL 与 FINAL20，各四个 L90（ordinal0/17/35/52），比较三控制。风险 L180 是各90cell的 ordinal66/67，独立评价 L180 是 ordinal68/69，与当前360包和彼此物理ID不交。它们均属于已训练过的合法源 L，未改变数据契约或持久角色；V、U IQ、U真值、target均不读取。每 RX 的风险／评价包为36，6TX×3day均衡。

风险 CE 使用真实四项 joint（base、PA、reference、SIG_TIME），原 W／三 gain detached。静态 feature 范围为部署 encoder 原参数，保留浅层、中层、PA、参考响应、原归一化等，排除 W／gain；不以当前非零梯度选择参数。INITIAL 明确是原配方、空矩、LR2e-4的假想一步，FINAL20 使用实际保存状态及 LR1e-6。两者是 CPU eval-mode 诊断，不冒充原 CUDA 训练同时发生的步骤，也没有中期证据。

FP64 解算只用5×5 Gram、最多32个 active set和KKT检查。UPDATE 的参数加法重新转换至实际dtype，逐 RX 记录理想风险、实际风险和 `sum(abs(g)*abs(actualDelta-idealDelta))` 舍入界；不使用全模型最大绝对值扩大容差。后续源网络评价恢复更新前 W 和全部增益，以直接比较 joint160 的有限步变化。零位移始终可行，因此必须同时核验保留位移、独立 held-L CE 和实际 feature 变化，不能把“被压成零”算收益。

预登记 source gate：八单元至少两个满足 RAW 至少一个 RX 正一阶风险、UPDATE 相对 RAW 实际位移差比例 >.01、实际位移保留 ≥.5、独立 held-L 最差 RX 的 CE 增量减少 >1e-6、mean CE 相对 RAW 增加 ≤1e-6、held feature RMS 差 >1e-7、至少一个浅／中层原参数实际不同，以及实际风险在 solver＋逐RX舍入界内。同时至少两个单元 UPDATE 相对 GRAD 实际位移差比例 >.01。门槛只判学习空间和有限步方向，不是 DG 性能、V选模或原创性证明。失败时关闭该固定变体，不扫描阈值、权重或 step scale。

实际预算：720 unique L／6300 metadata／10444 packet 主网络前向／16临时CPU AdamW／0真实训练、头拟合、EMA、U/V/target。任何来源／数据角色／状态不匹配、非有限、无合法 solver解或技术错误均保存产物并停止本 run，不影响其他 owner；科学失败不重试。

唯一 run `20261015-phase1-SIGTIME-RX-update-projection-sourceL-s2026092701-r01`，owner `codex/root/RX-update-source-20261015`。发布时实际 Git HEAD 写入 `release_commit.txt`；父 E20 执行版本 `6693e4480ef990bc23c5021ff1440b547fe5fcaf`，父本轮源码版本以当前 Git archive 固定。N607 普通账户，CVS-RFFI Python，CPU2线程／GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01/experiments/cvs_rx_update_projection20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01/resolved_request.json`。唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-RX-update-projection-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_update_projection_sourceL_20261015_r01/source_probe.stdout.log`。完整矩阵／数据／checkpoint来源／技术停止条件／预期产物见 [experiment.json](experiment.json)。入口两份真实 checkpoint 无 query smoke PASS 后立即继续固定源诊断。

状态 PLANNED，无实际源结果。只有固定源判定通过后才另行固定新 E20 或 E20+20 训练；原默认头、隐藏 U、匹配纯 CE、冻结后 clean truth-last流程保持。SIG_TIME、固定 CENTER25K、COMPACT_LINEAR 和 LONG_SILU 历史正收益独立保留，不能相加不同预算／结构的数值。84% 默认头 clean 三种子目标仍 active、未达成。
