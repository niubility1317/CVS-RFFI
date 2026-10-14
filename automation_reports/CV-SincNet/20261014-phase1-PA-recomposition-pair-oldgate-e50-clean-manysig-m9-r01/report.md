# 同身份跨 RX 的 PA 证据重组

本轮检验 CVS 的 PA 实际增量能否成为可跨接收机重组的身份证据。新目标由真实 L 身份标签决定，避免复制教师的全部关系/响应几何。特征插值已有 Manifold Mixup（https://proceedings.mlr.press/v97/verma19a.html）；待验证的机制在于 CVS 的可分离增量、同身份跨 RX 配对及梯度边界，不宣称混合特征本身新颖。

等 U 时序 R02 首次真实阶段测量已完成：MID/SPREAD 当时 ownEMA 在完整 V27000 准确率约96.8%，旧 gate 接受 coverage83.4%/precision99.9956%；真实同 AdamW 状态的一步 U 增量约为总更新范数的1.8%至1.9%，totalshadow误差1.49e−8。这里只是源 V 代理和一步测量，不解封 U、不给出实际 U 精度、不证明整体泛化改善。本 PA 配方 α=.2/λ=.1 已在上述数据及 U target 结果前固定，不选择胜出的日历，独立保留原tail33+17。

## 机制与固定对照

真实部署 `joint160=a+b+c`，其中 b 是已带实际 PA gain 的增量，c 是实际 reference projection 增量。只替换 b：`z̃_i=a_i+b_i+c_i+α(b_j−b_i)`，并对称训练 `z̃_j=a_j+b_j+c_j−α(b_j−b_i)`。同 TX、同 day、不同物理包，保持原a/c，α=.2固定。原现有 cosine W/scale30/normalize eps1e−4 计算真实 L CE；辅助 W 与 PA/ref gains detach，内部 features 有梯度。主 CE 正常联合训练原默认头和 gains，原 train/loss_tx_labeled 不包含辅助；不冻结、替换或重训头。

| 条件 | donor | 辅助系数 |
|---|---|---:|
| ZERO | 同成本计算全部诊断 | 0 |
| SAME_RX | 同 TX/day、同 RX | .1 |
| CROSS_RX（预定主条件） | 同 TX/day、跨 RX | .1 |

仅用每步已有 L128 前向。每 TX/day 至少2个合格 RX cell，每 cell 至少2个不同物理包；各合格 cell 私有随机选相同最小包数。共同 cohort 一对一配对，SAME cell内循环，CROSS cell间循环，每包各作一次anchor/donor，双向端点曝光完全匹配。剔除重复、不足和不平衡部分，记录实际coverage/cell counts；不为覆盖率改变身份/物理规则，不假设不同包同波形或物理硬件反事实。

## 原训练、预算与学习信号

全部新scratch50，33标签+17原pseudo阶段，L始终参与50轮；每轮222步，共11100真实AdamW/EMA更新，L1420800/U963900曝光。固定seed2026092701/02/03，共9模型/450模型轮次/99900更新。原默认CVS架构/844754参数/joint160/ResidualPhysicalHead6×160、NoExplicitCFO与内部补偿、PAIR、ownEMA=.999、旧confidence/temporal/strong gate、noise=.015/Dropout、λU=.16、entropy=.01、L smoothing=.01、AdamW lr=.0002/wd=.0001/clip5/cosine最低.000001均保留。不叠加R/E、TIME、Sinc、U日历、新伪标签链或训练信道增强。

前1110步warmup，新机制此后每4步，共2498calls/模型。新增参数0，额外L/U IQ前向0，额外实际优化更新0。所有conditions仍主 CE 正常训练完整L。在1112/3556/7328/10880四个真实步骤，记录aux/main-L特征梯度、方向、W/gain辅助梯度None；用真实总损失与总损失减去PAaux，复制同当前AdamW参数/moments/step/lr/decay，各自clip5模拟两步。原U/entropy留在base，真实仍更新1次，temporary total对照实际更新，记录各部署featuregroup数值更新与PAaux增量。8临时步骤/模型单列；一步增量不等于累计因果收益。Globalclip/moments可间接改变主W更新，不声称W训练轨迹不变。

三组191初始state和CPU RNG相同，真实 native构建→NoCFO→安装与预测builder一致，EMA在bound训练callback前deepcopy；原CUDA非确定性保留，不声称完整前缀逐位一致。数学、真实CVS与实际native L/coreloss→callback→probe→backward→clip→AdamW本地接线PASS，ZERO loss/gradient/update difference为0，active aux/Lnorm约.1168，actual-shadow误差0。样本是合成源形状128包，不能代替实际运行信号或泛化效果。

## 源资格与自动默认头测试

首seed三个row完整final50，final训练LCE≤.5、共同cohort平均coverage≥.25；两个active条件四点aux/Lfeature normratio均值≥1e−5；相对ZERO的源Vmean下降≤1pp、worstRX≤2pp。只据这些固定source-only条件放行另外6row，不微调α/λ/日历，不按target排序/停止/重跑。已启动健康row完成原预算。

每row固定final50后立即自动默认头clean168000包预测，所有prediction完成后freshCPU independenttruthlast评分。全9row1512000决定/162CM复算，报告CROSS−ZERO、SAME−ZERO、CROSS−SAME逐seed差/均值/SD/正差数、MacroF1/worstRX/day、源曲线/coverage/分支实际更新及负结果。只有CROSS稳定优于SAME与ZERO才支持跨RX机制贡献，不能用合成梯度或某个最高seed替代证据。固定主条件CROSS的三seedcleanmean≥84%是尚未达成的研究目标。

## 最小发布

- 数据：原VALIDATED_ONCE ManySig source contract/physical IDs/capsule/split复用，L6300/U56700隐藏TX/V27000，source RX1/3/4/6/8、day1/2/3、6TX，query clean168000。V仅原源健康/诊断，无梯度、阈值拟合或新头；Phase2不在范围。
- 初始化：9new scratch，无checkpoint/teacher/optimizer继承。首步no-query smoke仅严格读已核实同契约历史default50 student/ownEMA后零输入；不继承训练。数学与运行时observer/counterfactual实际import一起执行，不以compile代替依赖导入。
- run：`20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01`，唯一owner `codex/root/PA-recomposition-20261014`。
- 代码：不可变release的`release_commit.txt`，提交推送并独立核对remoteOID。
- 环境：N607普通账号、`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_PA_recomposition50_20261014_r01`；命令 `python -u -m experiments.cvs_pa_recomposition.dispatch`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-PA-recomposition-pair-oldgate-e50-clean-manysig-m9-r01`，独立不可变root。
- GPU：每GPU最多3总实验进程，计入其他owner，至少12000MBfree。不得热改、停止、重启健康旧run。
- 技术停止：本run的role/physical/来源错误、输出碰撞、非有限active训练、固定预算错误、PAcapture或W/gain边界错误、totalshadow不匹配、非法预测或未处理CUDA/data/artifact错误。两row同prediction前错误等预定系统异常仅处置绑定runowned进程，保留所有outputs，无未经修复的自动重试；target低分不触发停止。
- 产物：preflight、实际source args/契约/初始化/完整step与50epoch日志、2498PA调用/cohort/CE/margins、4actual gradient/AdamW/parameterupdates、finalcheckpoint/freeze、W独立featureexport、逐row默认head prediction/score/completion、source-only扩展决定、queue/analysis/finalreport。

唯一独立P0/P1审查PASS，无阻断问题。当前未发布PA、无真实训练/性能证据；等U/source阶段验证完成后按固定方案执行，既有结构交互和正收益路线独立保留。84%目标active，不能用发布或source信号代替验收。
