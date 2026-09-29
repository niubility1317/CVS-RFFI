# D92-BranchLocalRidge重复基准结果与资源报告

2026-09-29。对象：`D92-BranchLocalRidge-v1`；运行提交：`de10068cd1168ab064c835bf341ddb0f63e1231c`。本报告只分析已完成产物，不训练、选模、调整参数或重执行。**在既有已评分目标数据的重复基准中，K=1、5、10、20的合并旧类、新类及H均值均高于D92、BranchRidge和BranchInteraction；分层结果并非全部改善，K=1相对两个Branch基线的增益较小。**这不是独立泛化确认、统计显著性或逐任务共同改善结论，也不自动宣告优化目标完成。

证据状态为`VERIFIED_TRIPLE_BASELINE_ANALYSIS`，见[完整分析JSON][analysis]和[机器汇总报告][report]。用户最新指示为“先不用独立验证，继续提升”：独立验证按用户要求暂缓；后续研发由主Agent协调，保持query-blind、仅依据合法support，不向方法研发回流本报告的query结果。

## 1. 覆盖、来源与统计口径

本次LocalRidge的rx3/rx1两个cohort均完成，分别有3600/1200次拟合，合计4800次，其中真实K=1为1200次。四个model seed为2026092701至2026092704，五个support seed为2026092711至2026092715。接收机为19-1、8-14、8-7、20-19；产物中的场景标识为`practical_high`、`practical_mid`、`practical_low_urban`，本报告保留原标识，不把它们改称其他场景族。

旧类固定6类，新增类数为0、2、5、10、20，总注册类数C=6+Nnew，每类K个物理support，总support数N=(6+Nnew)×K；K=1并不表示整个任务只有一个support。

| 新增类数 | 总注册类数 | K=1总support | K=5总support | K=10总support | K=20总support |
|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 6 | 30 | 60 | 120 |
| 2 | 8 | 8 | 40 | 80 | 160 |
| 5 | 11 | 11 | 55 | 110 | 220 |
| 10 | 16 | 16 | 80 | 160 | 320 |
| 20 | 26 | 26 | 130 | 260 | 520 |

每个K×新增类数有240个配对cell。每个K的旧新联合任务为960个cell，含old-only的全部任务为1200个cell。rx3/rx1按实际cell贡献3:1，不等权平均两个cohort。H先在每个cell内按旧类、新类准确率计算调和均值，再对cell等权平均；不是先汇总旧、新准确率再计算H。新增0类时新类与H为N/A。

候选、BranchRidge、BranchInteraction的两组cohort共六个运行产物完成并绑定后才进行比较。[来源与逐字段一致性审计][analysis]确认：每条运行配对中，rx3的3600条原D92及36条DG记录、rx1的1200条原D92及12条DG记录完全相同；两个Branch参考分别原样复用3600/1200条记录。[rx3算术审计][rx3-audit]与[rx1算术审计][rx1-audit]分别核对7236/2412条评分记录，合计9648条，范围含混淆矩阵派生的准确率、旧/新/H、Macro-F1、分组F1、类别下限及query计数。9648不是独立样本量，也不是候选拟合次数。

本报告完整解析三基线、合并及两个cohort的90份CSV，共2052行；完整读取分析与拟合审计元数据。未读取原始IQ、truth、特征值或checkpoint，未访问远端。记录与算术一致性审计不等于重新执行数值拟合，也不替代原输入协议及来源证据。

## 2. 方法含义与每K结果

LocalRidge保留原五个特征块，在固定交互空间中使用径向核。带宽tau和trace比例gamma只由当前合法训练support确定，固定lambda=1，进行一次解析ridge求解，不运行优化器。736维显式输入与123616维隐式交互空间不表示新增独立物理样本；query只使用已固定状态逐样本面对全部注册类。描述依据为[冻结实现审查][p0]，本报告不提出方法修改。

下表为旧新联合任务绝对值，单位%，顺序为旧类/新类/H；保留三位小数，原精度见[分析JSON][analysis]及对应CSV。

| K | LocalRidge | D92 | BranchRidge | BranchInteraction |
|---:|---|---|---|---|
| 1 | 42.064 / 29.536 / 33.139 | 37.109 / 23.672 / 26.547 | 41.997 / 29.089 / 32.671 | 41.912 / 29.145 / 32.681 |
| 5 | 60.038 / 46.126 / 51.524 | 44.103 / 35.316 / 38.419 | 57.626 / 42.154 / 47.795 | 58.771 / 43.960 / 49.497 |
| 10 | 64.749 / 53.051 / 57.868 | 53.482 / 42.215 / 46.536 | 62.797 / 48.318 / 53.905 | 63.720 / 50.635 / 55.825 |
| 20 | 68.833 / 58.445 / 62.840 | 60.244 / 46.303 / 51.715 | 66.523 / 52.804 / 58.295 | 67.560 / 55.545 / 60.488 |

配对差值单位为百分点（pp），顺序仍为Δ旧/Δ新/ΔH。

| K | 相对D92 | 相对BranchRidge | 相对BranchInteraction |
|---:|---|---|---|
| 1 | +4.955 / +5.864 / +6.592 | +0.068 / +0.446 / +0.468 | +0.152 / +0.391 / +0.458 |
| 5 | +15.935 / +10.810 / +13.104 | +2.411 / +3.971 / +3.729 | +1.267 / +2.166 / +2.026 |
| 10 | +11.267 / +10.836 / +11.332 | +1.952 / +4.734 / +3.964 | +1.028 / +2.416 / +2.043 |
| 20 | +8.590 / +12.142 / +11.125 | +2.311 / +5.641 / +4.545 | +1.273 / +2.901 / +2.352 |

预定每K条件为新类与H严格提高，全部cell（含old-only）的旧类下降不超过1个百分点。三个对照的四个K均通过该条件，且联合任务旧/新/H均值三项严格提高。含old-only的LocalRidge旧类均值依K为43.841%、61.771%、66.326%、70.250%；相对D92差值为+4.571、+15.275、+10.561、+7.864pp，相对BranchRidge为+0.059、+2.288、+1.837、+2.118pp，相对BranchInteraction为+0.124、+1.154、+0.935、+1.164pp。通过预定均值条件不意味着全部分层均为正，也不增设这种门槛。

## 3. 新增类数、seed、接收机与cohort分层

所有K×新增类数的候选绝对值如下；完整三基线配对差值见[机器报告的20格全表][report]及[分析JSON][analysis]。

| K | 新增0类：旧 | 新增2类：旧/新/H | 新增5类：旧/新/H | 新增10类：旧/新/H | 新增20类：旧/新/H |
|---:|---:|---|---|---|---|
| 1 | 50.949 | 47.630 / 30.160 / 33.988 | 43.850 / 31.422 / 35.200 | 40.387 / 29.893 / 33.476 | 36.391 / 26.668 / 29.891 |
| 5 | 68.704 | 65.296 / 44.931 / 52.144 | 61.405 / 48.611 / 53.715 | 58.356 / 46.665 / 51.456 | 55.093 / 44.296 / 48.781 |
| 10 | 72.637 | 69.713 / 52.215 / 58.964 | 65.528 / 54.686 / 59.239 | 63.035 / 53.353 / 57.539 | 60.720 / 51.951 / 55.731 |
| 20 | 75.914 | 72.984 / 57.694 / 63.759 | 69.382 / 59.875 / 63.998 | 67.451 / 58.558 / 62.478 | 65.516 / 57.653 / 61.126 |

合并K×新增类数下，相对D92三项没有负差值；相对BranchRidge的K1、新增20类旧类为−0.050926pp，新类仅+0.045139pp、H仅+0.002209pp；相对BranchInteraction的K1、新增2类旧类为−0.002315pp。K1、新增20类相对BranchInteraction的新类/H也只有+0.038194/+0.095959pp。这些数值体现收益很小，不应按四舍五入后的正数宣称稳定优势。

下表完整统计合并分层中的负差值个数；每格顺序为旧/新/H。新增0类只参与旧类计数，新类/H为N/A，不算负数。各行层级相互重叠，不能相加为独立失败任务数。

| 分层 | 可用层数（旧/新/H） | 相对D92负层数 | 相对BranchRidge负层数 | 相对BranchInteraction负层数 |
|---|---|---|---|---|
| K×新增类数 | 20 / 16 / 16 | 0 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 |
| model seed×K | 16 / 16 / 16 | 0 / 0 / 0 | 1 / 0 / 0 | 0 / 0 / 0 |
| RX×场景×K | 48 / 48 / 48 | 1 / 1 / 0 | 8 / 3 / 3 | 11 / 4 / 4 |

RX×场景×K的最小差值及出处如下。来源：[D92分层CSV][d92-rx]、[BranchRidge分层CSV][branch-rx]、[BranchInteraction分层CSV][interaction-rx]。

| 对照 | 最小Δ旧（pp）及分层 | 最小Δ新（pp）及分层 | 最小ΔH（pp）及分层 |
|---|---|---|---|
| D92 | −0.312500；20-19 / practical_low_urban / K1 | −2.047917；19-1 / practical_low_urban / K1 | +2.064003；19-1 / practical_low_urban / K1 |
| BranchRidge | −0.743056；20-19 / practical_mid / K1 | −1.056250；8-7 / practical_low_urban / K1 | −0.585933；8-7 / practical_low_urban / K1 |
| BranchInteraction | −0.604167；8-14 / practical_high / K20 | −0.531250；8-7 / practical_low_urban / K1 | −0.380069；8-7 / practical_low_urban / K1 |

合并model seed×K中，唯一负差值是相对BranchRidge的seed=2026092701、K1旧类−0.041667pp。拆到cohort后仍有额外负层：rx1相对BranchRidge的16个seed×K中旧类有4层为负，相对BranchInteraction有6层为负，新类/H均没有负层；其中K10也出现旧类下降，不能笼统归为全在K1。rx1最小旧类差值分别为−0.796296pp和−0.574074pp，均在seed=2026092701、K1。rx3的seed×K相对三基线三项均为正。依据：[rx1 BranchRidge seed CSV][branch-seed]、[rx1 BranchInteraction seed CSV][interaction-seed]。

cohort×K中，rx3相对三个对照三项均为正；rx1的K1旧类相对BranchRidge/BranchInteraction为−0.263889/−0.240741pp，而新类/H分别为+1.019444/+0.971200pp和+0.961111/+0.883208pp。更细的rx1、K1、新增20类三项均下降：相对BranchRidge为−0.768519/−0.194444/−0.362305pp，相对BranchInteraction为−0.555556/−0.083333/−0.207594pp。rx3、K1、新增5类相对BranchRidge新类也有−0.003704pp的微小下降。两个cohort的完整新增类分层分别保留于[rx1 BranchRidge CSV][branch-rx1-new]、[rx1 BranchInteraction CSV][interaction-rx1-new]和[rx3 BranchRidge CSV][branch-rx3-new]。

以上是完整指定聚合层级的检查，不是每个原始cell均改善的证明。多个模型、support抽样及不同方法复用相同target/query，不能把这些相关结果当作独立重复，未计算或宣称统计显著性。

## 4. 全部拟合记录与实际资源开销

[完整fit_audit.json][fit]覆盖全部4800次拟合：4800次完整fit、4800次分解、0次fold fit、0个optimizer step；每次为float64 Cholesky加两次三角求解，无jitter。本轮另有9600次用于有效自由度诊断的三角求解，不能隐去诊断开销。没有退化减少分解或零带宽情况。记录的最大梯度范数为4.4151×10⁻¹³，最大正规方程相对残差为7.1817×10⁻¹⁷；这些是求解记录的数值一致性指标，不是query识别质量指标。

| 实际成本项目 | 全记录结果 | 口径 |
|---|---:|---|
| 拟合耗时 | 923.787秒 | 各进程各episode记录之和 |
| fit调用耗时 | 926.661秒 | 与上一行重叠，不能相加 |
| query评分耗时 | 4345.239秒 | 逐query推理记录之和 |
| prediction写入耗时 | 22.986秒 | 各记录之和 |
| 总episode进程工作时间 | 5299.532秒 | 不等于真实墙钟时间 |
| 联合墙钟跨度 | 934.200秒 | 两cohort时间范围，含编排、基线核验与独立评分 |
| 两cohort墙钟时间之和 | 1427.302秒 | rx3为934.200秒，rx1为493.102秒；存在并行重叠 |
| 缓存读取时间之和 | 0.535秒 | 本轮复用已有缓存 |
| 本轮特征提取/模型前向 | 0秒 / 0次 | 不代表从原始IQ部署推理没有成本 |
| 最大观测单进程RSS | 566239232字节 | 非同时运行所有进程总内存，非GPU峰值 |
| 单episode数值头/持久状态 | 35744至3178272字节 | 均值728657.6字节；非完整模型大小 |

Gram构造累计907.664秒、解析solve累计3.769秒、有效自由度诊断累计4.691秒，属于嵌套或相邻内部计时口径，不与上表机械求和。最大单episode训练特征数组为3061760字节，Gram数组为2163200字节，系数为108160字节。数值头按`8*N*(736+C+2)+16+8*(tau is not null)+8*(gamma is not null)`记录；全部episode状态快照累加3497556480字节，不是并发占用，也不是必须持久保存的部署容量。

本轮query用于fit的行数为0，source用于fit的行数为0；新增source payload与新增地面统计量均为0字节。复用的received特征数组合计236980224字节、特征文件244741200字节、registry数组7727616字节，均为已有存储量，不能写成此次新增传输量。四个唯一既有模型文件合计63971616字节，指完整训练checkpoint包，不是测得的最小推理包；本轮没有加载checkpoint，新增模型传输记录为0字节，`model_already_deployed=null`。因此“模型是否已部署”和实际完整部署传输成本仍未知，不能由缓存复用或已有文件大小推出零部署成本。

## 5. 声明范围与交接

本报告可确认冻结LocalRidge在这些既有重复基准中的每K均值改善及预定均值保护条件通过，并完整披露负分层与实际记录成本。它不能证明未曝光数据上的独立泛化、逐任务改善、统计显著性、真实卫星验证或零部署成本。既有结果保留，不据此选择性重跑；此次分析不改变冻结方法、数据协议、科学判据或实验授权。

独立验证状态：`DEFERRED_AT_USER_REQUEST`。继续提升属于用户已明确要求的后续工作，由主Agent另行协调query-blind、support-only研发；本结果报告本身不自动晋级、不标记整个目标完成。

[analysis]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/analysis.json
[report]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/report.md
[fit]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/fit_audit.json
[rx3-audit]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/arithmetic_audit.json
[rx1-audit]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx1-m4-r01/results/arithmetic_audit.json
[p0]: E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt/docs/D92_BRANCH_LOCAL_RIDGE_P0_20260929.md
[d92-rx]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/vs_d92/combined_rx4/per_rx_scene_k_delta.csv
[branch-rx]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/vs_branch_ridge/combined_rx4/per_rx_scene_k_delta.csv
[interaction-rx]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/vs_branch_interaction/combined_rx4/per_rx_scene_k_delta.csv
[branch-seed]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/vs_branch_ridge/rx1/per_seed_k_delta.csv
[interaction-seed]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/vs_branch_interaction/rx1/per_seed_k_delta.csv
[branch-rx1-new]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/vs_branch_ridge/rx1/per_k_new_delta.csv
[interaction-rx1-new]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/vs_branch_interaction/rx1/per_k_new_delta.csv
[branch-rx3-new]: E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-ridge-repeat-rx3-m4-r01/results/triple_baseline/vs_branch_ridge/rx3/per_k_new_delta.csv
