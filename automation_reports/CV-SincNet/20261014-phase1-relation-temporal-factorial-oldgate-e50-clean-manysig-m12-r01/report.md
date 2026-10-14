# 已验证时序结构与关系响应：匹配组合

本轮落实用户“此前有明确正收益的路线不能落下”的要求。前期 COMPACT_LINEAR 与 LONG_SILU 的三种子独立结果完整，当前关系首种子的预定 source-only 条件已在 target 汇总前通过。既有目标文档据此预先保留了逐项匹配组合。本轮只检验这两种结构与原关系＋响应目标是否协同；Sinc 居中坐标保持独立，不把所有结构一次叠加。

该组合是用户要求的已暴露基准后续探索，不是新的独立盲测或已经成立的机制收益。新关系矩阵与原健康进程保持原计划。当前报告无新目标结果，不根据首种子或后续目标结果选择父模型、改 R/E 权重、τ、α 或停止健康训练。

## 固定四条件

| 条件 | 时间路径 | 辅助训练目标 |
|---|---|---|
| COMPACT_ZERO | 原 COMPACT_LINEAR，t1/t2/t3 核/膨胀=(3,1)/(5,1)/(5,1)，分支线性 | 计算关系诊断但训练权重精确为 0 |
| COMPACT_RESP | 同 COMPACT_ZERO | 跨 RX 关系＋PA/ref 签名增量响应 |
| LONG_ZERO | 原 LONG_SILU，t1/t2/t3=(3,1)/(5,1)/(5,2)，分支 SiLU | 计算关系诊断但训练权重精确为 0 |
| LONG_RESP | 同 LONG_ZERO | 同 COMPACT_RESP 的关系＋响应 |

每条件固定 seed2026092701/02/03，各独立 scratch50、33 标签＋17 原伪标签阶段、222 步/轮、11100 更新；总计 12 新模型/600 模型轮次/133200 更新。先运行首种子四行，完整源条件通过后自动放行其余 8 行。固定 primary 为 LONG_RESP，clean 三种子均值 84% 仍为未达到的最终研究目标；所有对照及负结果保留。

机制完全沿用原 R/E：τ=.2、α=.5、λR=λE=1；ownEMA teacher、PAIR 元数据排列、旧门控、原 strong noise=.015 与 Dropout、原全 U entropy=.01、hard pseudo CE λU=.16、原优化器和 cosine。所有合法 U 学习连续个体关系，不新增 TX 伪类投票/阈值链。每 query120参照/每 TX20，day7/7/6；L180银行每222步刷新，共17次/3060额外 L 包前向/模型（总36720）。四组同银行和观测成本，ZERO 的加权辅助梯度精确为0。

PA、frequency、reference 原结构与真实 joint160 保持。所有模型905426个总参数，机制新增参数0；时序结构相对原网络增加60672个参数。默认 ResidualPhysicalHead6×160/scale30 正常随骨干联合训练。辅助目标不经类别 W，仅 auxiliary 重组 PA/ref 增益 stopgrad，内部特征仍有梯度；主 CE 与原增益正常训练。推理不用参照库，无另训身份/RX/分支分类器或 Phase2。

本地四个模型的200个初始 state tensor 和构建后的 CPU RNG 逐项相同；同结构的 ZERO/RESP 初始函数相同。紧凑/长覆盖和线性/SiLU 函数不同，不声称结构初始函数相同。原 `cudnn_deterministic=False` 保留；相同初始张量不证明训练轨迹逐位相同，实际 native policy 为 TF32False、benchmarkFalse、highest。

## 比较与源条件

预定配对为 COMPACT_RESP−COMPACT_ZERO、LONG_RESP−LONG_ZERO、LONG_ZERO−COMPACT_ZERO、LONG_RESP−COMPACT_RESP。额外固定交互为 `(LONG_RESP−LONG_ZERO)−(COMPACT_RESP−COMPACT_ZERO)`，完整三seed报告均值、SD、逐seed差及最差 RX/day。不把本轮结构因素拆成覆盖与 SiLU 各自独立因果贡献；该拆分已在原时间四因素矩阵报告。

首种子四个模型完整 final50，实际 final L_CE≤.5。COMPACT_RESP 相对 COMPACT_ZERO、LONG_ZERO 相对 COMPACT_ZERO、LONG_RESP 相对 LONG_ZERO 的源 V 均值下降≤1个百分点、最差源 RX 下降≤2个百分点；两个 RESP 条件的四个真实 feature auxiliary/L gradient ratio 平均≥1e−5、非零关系包比例≥.5。仅这些预定源条件决定 pending seeds 扩展，target scores 不进入 source decision，健康 firstseed 不因低 target 值重启或停止。

逐 row 冻结 final50 后自动默认头 clean168000包预测，prediction 固定后 freshCPU independent truth-last score。全矩阵2016000决定/216混淆矩阵完整复算。源浅层/PA/ref梯度和更新、3774辅助调用/963900 U观察、bank年龄、关系/响应分散、非有限/hidden truth状态全部保留，不用接线成功代替泛化提升。

## 发布与产物

- 数据：既有 VALIDATED_ONCE ManySig source contract、capsule/split/角色物理 IDs 完全复用，L6300/U56700 隐藏 TX/V27000，源 RX1/3/4/6/8、day1/2/3、6 TX；NoExplicitCFO 身份输入、内部补偿保留。
- 初始化：12 个模型从零，无加载权重/teacher/optimizer 来源。首步 no-query smoke 仅严格读取已验证同契约 COMPACT_LINEAR/LONG_SILU 的历史 final50 student/ownEMA，不继承到训练。
- run：`20261014-phase1-relation-temporal-factorial-oldgate-e50-clean-manysig-m12-r01`；唯一 owner：`codex/root/relation-structure-20261014`。
- 代码身份：不可变 release 内 `release_commit.txt` 的 Git commit。
- 环境：N607 普通账号，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_relation_structure50_20261014_r01`；命令 `python -u -m experiments.cvs_relation_structure50.dispatch`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-relation-temporal-factorial-oldgate-e50-clean-manysig-m12-r01`，不可变独立根。
- GPU：计入其他 owner，每 GPU 最多3总实验进程、≥12000MB free；不修改、停止或重启健康原任务。
- 技术止损：本 run 来源/物理角色不一致、输出碰撞、非有限 active training、源预算不可能、无法生成合法预测或未处理 CUDA/data/artifact 错误，保留产物，无自动重试，只处置 exact run-owned 状态。源科学标准只控制未启动 seed，不使用 target 止损。
- 产物：preflight、完整 source args/initialization/init state/step日志、R/E调用和实际参数更新、source frozen/finalcheckpoint、no-fit diagnostics、逐rowprediction/score/completion、source-only扩展决定、queue和总analysis/finalreport。

数学负例与真实 native auxiliary callback 已核实 mode 正确，ZERO梯度0、RESP真实t1/ref非零、W/gains辅助梯度None、hidden U TX negative PASS。合成梯度 fixture 将初始0的 ref gain设.1，仅用于本地接线；实际训练初值和冻结 checkpoint 不做此修改。唯一独立P0/P1审查、首步真实checkpoint no-query smoke、一次archive传输比较/compile和postlaunch进程读回由最小发布流程完成。当前没有新实验性能结论，84%目标继续 active。

唯一P0/P1审查发现 shared train 要求 model.install；已加入真实结构安装入口，并核实 native build_baseline_model→NoCFO→install 的200个初始state tensor逐项等于推理build。该问题的一次定向复审PASS，其余直接P0/P1无阻断。当前只待发布首步真实checkpoint no-query smoke，未宣称泛化有效。

## 真实发布读回

状态RUNNING/VERIFIED，执行commit `0af34c5345cba3488a5e4a5d429b2a26447fd09d`。首步真实历史temporal student/ownEMA strict no-query smoke与source L180probe梯度PASS，archive单次local/remote比较和remotecompile PASS。4个首seed worker PID/CWD/argv/GPU/log增长已独立核实，当前第15至18轮，真实各905426参数/default6×160/机制参数0。后8row仅按固定source规则扩展；无新target成绩，无性能成功结论。原关系12row已完整闭合，低分及CUDA非确定性限制保留，不热改任何健康任务。

## 完整三种子与结构内机制结果

ANALYZED / VERIFIED：全部12模型独立scratch50冻结、默认头clean预测与freshCPU truth-last评分完成，2016000固定决定、216CM及配对/交互独立复算，全部进程自然退出，无缺row。

| 条件 | clean均值 | SD（百分点） | MacroF1 | 逐seed最差RX均值 | 逐seed最差day均值 |
|---|---:|---:|---:|---:|---:|
| COMPACT_ZERO | 81.4137% | 0.5624 | 0.808958 | 72.1583% | 77.9746% |
| COMPACT_RESP | 80.6952% | 0.3494 | 0.800758 | 70.7250% | 77.4413% |
| LONG_ZERO | 81.4754% | 0.6113 | 0.810046 | 71.4361% | 77.2754% |
| LONG_RESP | 81.0359% | 0.9905 | 0.804747 | 70.5417% | 77.5738% |

| 配对 | 均值差（百分点） | seed01/02/03（百分点） | 正差数 |
|---|---:|---|---:|
| COMPACT_RESP−COMPACT_ZERO | -0.7185 | -0.9327 / -0.2786 / -0.9440 | 0/3 |
| LONG_RESP−LONG_ZERO | -0.4395 | -0.4143 / -0.9440 / +0.0399 | 1/3 |
| LONG_ZERO−COMPACT_ZERO | +0.0617 | +0.7375 / -0.0685 / -0.4839 | 1/3 |
| LONG_RESP−COMPACT_RESP | +0.3407 | +1.2560 / -0.7339 / +0.5000 | 2/3 |

固定交互均值+0.2790个百分点，SD=0.8504个百分点，逐seed=+0.5185 / -0.6655 / +0.9839，正差2/3。

**关系响应在两个匹配结构内均未确认收益。** COMPACT内−0.7185个百分点、0/3正；LONG内−0.4395个百分点、1/3正。正交互只说明LONG条件下损害较小，不能解释为关系机制有效或两者协同提升。LONG_ZERO对COMPACT_ZERO仅＋0.0617个百分点、1/3正，本轮PAIR/关系诊断上下文不确认以前长覆盖＋SiLU的稳定额外收益；历史独立正结果保留，不把配方不同的结果相加。

预定主条件LONG_RESP三seed81.0359%，距84%2.9641个百分点。COMPACT_ZERO81.4137%、LONG_ZERO81.4754%；不能以高于旧原架构的组合均值来证明关系目标贡献。PA重组α/λ与默认tail配方已在本结果前固定，不按本矩阵target调整或选择parent。

全量600模型轮次/133200更新/45288辅助调用/11566800隐藏U观察、36720额外合法L anchor前向已核实；每模型50×222完整source records，0非有限跳步/0U truth/0headfit/0stdout fatal。48真实梯度点及48AdamW更新齐全，ZEROaux梯度0，6active模型的t1/t2/t3/PA/ref实际收到非零aux梯度，W/两个增益aux梯度None。实际训练L/U对照方向是不同批次的参数梯度，不冒充此前同包有标签CE方向诊断。

完整默认head、strict source/final预算、W无关纯特征export和featureparameter/buffer保持的no-fit诊断均已完成。source_epoch_means.csv保留600轮曲线，final_seed_metrics.csv保留全部12结果。统计复算发现跨Python stdev末位5e−19差异，使用1e−12比例容差，所有整数CM计数精确一致；不是数据或性能差异。

原source-only资格先于target汇总，完整firstseed后按原规则扩展，未热改训练或挑分重跑。四结构初始200state匹配、同结构初始函数相同，但原CUDA非确定性不承诺逐位共同前缀，全部限制保留。Sinc、COMPACT/LONG历史正收益独立记录，关系＋响应不再作为有证实收益的主线。下一动作是既已固定的等U时序与PA身份标签重组；语义间隔后置、不叠加当前两项。84%研究目标active，当前组合矩阵完整闭合。
