# 单视图完整 PAMF：三个 REPL 改进候选

用户要求独立推进本方向，不再向另一个对话发信息；对照只复用已完成产物。当前只训练三个新候选，不重训监督、普通 PL 或原 REPL。两个旧模型仅做冻结 source V 推理，不访问目标或更新参数。

## 固定新矩阵和预算

单学生 seed2026092701，原F0 L6300/U56700/V27000，RX[1,3,4,6,8]、day[1,2,3]、原六TX、原VALIDATED_ONCE/物理ID。每候选从已经完成且已核实的m201单SIG_TIME普通PL E20/4440出发，原生+10+20至E50/11100，13L＋37LU；各新增6660更新/30轮，三行合计19980更新/90模型轮，原E20成本另计。不加载NoSIG/X3转换权重、停止控制的partial或按目标选择的F50。

| 新候选 | 风格混合 | U 外推 |
| --- | --- | --- |
| REPL_NoStyle | 0 | 原整条同伪类跨RX特征差，alpha.25 |
| REPL_NoStyleNoRXE | 0 | 无 |
| REPL_RXOffset | 0 | 真实L支持且去L类别均值方向的RX偏移，alpha.25 |

三行保留完整PAMF时频/PA/参考响应/单SIG_TIME/原6×160头joint，L768前向、AsKL0、ordinary/raw RDrop、LS.01、原浅中层跨RX监督及Sinc CENTER_25K/CENTER_UNITS和分段优化设置。U保持原REPL当前学生eval硬伪标签、threshold.95、weight6、六学生前向、浅中层机制、joint160对比.05、完整L梯度保护normcap1，无EMA教师/新头。hard_pl_no_uas=True沿用原所选REPL：U不加星地As，原L As保持。报告真实接受/曝光，不把接受率叫U正确率。

## 新 RX 方向

此前完整源几何的两表征已过预定必要条件，并独立复算2.22e-16；它不是分类提升。RXOffset只复用其中已完成E20的raw L6300归一化特征和可见真实标签初始化6类×5RX均值，每单元210条；不读V/Gmean/U构建统计。

每步捕获原第一L前向的真实CompositeReference joint160，不新增骨干前向；detach后按实际当前L标签/RX更新出现单元，固定EMA decay.9，未出现单元保留已有L均值。均匀平均类别形成RX偏移，再投影掉当前L类别均值差子空间。U使用`normalize(h_i)+.25*(b_perp[rx_i]−b_perp[rx_j])`，j仍为原共享计划的accepted/samepseudo/differentRX配对；方向完全detach，不向U donor特征反传。原有效配对ordinary CE .5只重分配为ordinary.25＋offset.25，保留CE归一化1.25，不丢未配对样本；增加两次原默认头调用，无新增U骨干或推理参数。

L统计与学生在E30同row同契约保存/续接，不能第二段重置到E20。最终L统计6660次、852480次当前L曝光，初始化6300另计；U/V/target不更新统计。原默认头和骨干仍联合训练，推理无需这些统计。

## 实际 smoke、执行和资源

代码/config由单个Git commit固定并写runtime.code_commit。CWD PROJECT/releases/20261026-phase1-PAMF-SIG1-REPL-RXOffset-source-release-r01；既有N607 Phase1 Python/Torch2.1，FP32/TF32off/CPU2，无安装。命令：PYTHON -u -m experiments.pamf_channel_pl.sig1_repl_offset_source dispatch --config RELEASE/runtime.json。唯一owner codex-pamf-sig1-repl-offset-source-20261026，独立不可覆盖run/output，各命令路径见experiment.json。

先真实已完成E20/defaulthead、L128/U256联合backward，0optimizer；逐行验证L768/AsKL0/U6、style实际0、真实U梯度到singleSIG/原head/body有限非零；RXOffset还验证L统计更新与非零RX位移、两次原头调用。使用独立candidate_smoke标志，避免原hard_pl_smoke强制启用style。PASS立即继续，不能只保存参数配置当执行证明。

全部物理GPU0至7可用，派发前核实全局CUDA＋pending<2且free≥10000MiB，计入其他owner；不静态占卡、不更改其他健康工作。一次archive比较/远端compile及独立PID/CWD/argv/GPU/log增长。首个技术/输入/来源/非有限/shape/统计续接/smoke/预算/输出故障停止新派发，健康已运行行继续，保留全部产物，不自动重试；科学低分不触发技术停止。

## 自动源评估与冻结选择

每行final自动原V27000八视图raw/G(raw)/三H/三G(H)，逐包logits固定后源TX/RX/day/overall统计，F/G所有参数与buffers保持冻结。G固定同一原F0 G-LU/DURA，不训练G。原完成F21-X3和m201 REPL_SIG_TIME E50作只读source参照，不训练、不重复其目标预测。X3与单视图差异如实标注，不能宣称严格架构匹配的SSL因果效果。

三新行和两参照全部source结果完成后才选择。每指标相对两参照的最大值：raw≥−.1pp，H3≥−.1pp，G3≥+.3pp。以精确正确数实现：raw V27000最多少27条，三H合计81000最多少81条，三G合计81000至少多243条。每项都必须通过，避免浮点边界误判。通过者先G3正确数最高，再raw、H3；完全平分按低计算量顺序NoStyleNoRXE、NoStyle、RXOffset。无通过则不扩种子，不降低门槛或选择性重跑。

source PASS不代替目标成功。后续同一冻结候选扩三个学生seed，使用各自已完成单视图E20和自己的L统计，原目标F0所有目标RX日期168000包/seed、raw和同G clean＋六practical全部预测后独立truth-last。原绝对目标保持rawclean≥85.981746%、raw六景>72.096197%、同G六景>78.074636%，不混入cross_rx126000口径，也不把三个变体当三个seed。
