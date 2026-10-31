# MILKR 宏 H 提升目标与当前证据

用户目标：固定 **Tri-SIG-TIME SSL** 基座，优化 MILKR，使宏 H 比匹配原版提高至少 5 个百分点；可用适当小矩阵，覆盖各 K/N。状态：**ACTIVE，未达成**。不得把实验技术完成写成科学目标完成。

## 当前最新结果：R14七RX确认

- S0：Q0较原版+4.1100pp；Q1相对当前Q0增量+0.9372pp；均值验收False，每格验收False。
- S1：Q0较原版+33.9102pp；Q1相对当前Q0增量+17.5136pp；均值验收True，每格验收False。

确认六RX和合并七RX均值共同达标：False；每格严格口径：False。全K1/5/10/20×N2/5/10/20、P01/MIX01、单seed2026092701，七RX560新/560原版B/C评分行。原rank16源参数复用，未改F/G。无所属进程残留。完整逐KN/RX见pairedcca-confirm r13报告；后续如要求每格+5或更广证据，不能继承为已达成。以下早期ACTIVE/未达成语句均为对应候选的历史状态。

## 固定范围

- F 为原完整三视图 SIG-TIME SSL（REPL全部机制、固定时序排列），原 seed2026092701，232993参数、源E50/11100；DURA v1.0保持原冻结身份。
- 每轮只用一个既有物理池的 MIX01，七个目标RX全部保留；K=1/5/10/20 × N=2/5/10/20，每单元全部注册类 query 补集。当前是有标签 K-shot support 的 Phase2 方法研发，不是零目标标签新类身份注册。
- 原版通过 r08 权威 jobs.json 连接 r04/r05 同 seed、同物理、同 split、同路径的冻结结果；不重新训练或重跑旧基线。
- 源L代理选择参数/规则。目标适应只消费当前 support；query始终逐样本测试，全部预登记预测固定后独立scorer才接真值。任何改进结论均按相同处理路径配对。
- 此基准已曝光，三池也不构成新盲测；不从目标成绩选择超参数、重排候选或选择性重跑。每张GPU最多两个新增训练/预测worker，其他owner保持。

## 已完成证据

前三轮及R9均完成了真实checkpoint无query smoke、针对性测试、每候选一次独立P0/P1审查、Git push后远端OID核实、一次release归档比较与远端compile、所属PID/CWD/argv/GPU/log核验、完整预测和独立truth-last评分。R8完成对应源端阶段，按科学停止规则未启动目标。所有所属进程正常结束。新增F/G训练与更新均为0。

| run | 物理池 | 方法 | 宏H主S0Q0配对变化 | S1Q1变化 | 科学判定 |
|---|---|---|---:|---:|---|
| [semantic r01](../../automation_reports/CV-SincNet/20261031-phase2-milkr-semantic-kn-s2026092701-r01/report.md) | P01 | R1新核；R2默认头残差；R3继承；R4/R5双视图 | R1 +1.58pp；主R3 -2.24pp | R1 +1.87pp；R3 +0.45pp | 主目标失败，R3各16格均未+5pp |
| [support-CV r02](../../automation_reports/CV-SincNet/20261031-phase2-milkr-supportcv-kn-s2026092701-r02/report.md) | P02 | R6固定核网格＋支持条件LOO | +2.42pp | +1.68pp | 总均值未达+5pp，16格中1格达到 |
| [pooled-metric r03](../../automation_reports/CV-SincNet/20261031-phase2-milkr-pooledmetric-kn-s2026092701-r03/report.md) | P03 | R7核类内scatter度量＋精确scatter/centroid LOO移除 | +0.22pp | +0.97pp | 总均值未达+5pp，16格中1格达到 |
| [group-calibration r04](../../automation_reports/CV-SincNet/20261031-phase2-milkr-groupcal-kn-s2026092701-r04/report.md) | 仅源端 | R8旧新分数偏移 | 源held -1.17pp，目标0行 | N/A | 源科学筛选失败，正常source-only结束 |
| [signature r05](../../automation_reports/CV-SincNet/20261031-phase2-milkr-signature-kn-s2026092701-r05/report.md) | P02 | R9额外读取实际投影前336维三视图描述符 | +3.13pp | +3.03pp | 总均值未达+5pp，16格中2格达到 |

前三轮使用不同物理池，只能逐轮与匹配原版比较，不能把三个原始均值当成同row候选排名。严格S1Q0分别有明显改善（R1 +19.94pp、R6 +21.88pp、R7 +18.32pp），但该路线原来严重失配，不能拿它替代主S0Q0或同路径S1Q1的5pp目标。

R9与R6在同一P02/seed/split/处理路径实际配对：S0Q0增加0.7111pp、S1Q1增加1.3526pp；两者各自按source/support选了核参数，不能称为完全固定核参数的单坐标因果消融。R9的S1Q0/S0Q1相对原MILKR为-0.23/-16.95pp，单侧恢复兼容性仍有明显问题，不能因同路径增益就发布统一替代。

每轮完整旧/新/H/F1/floor、四格及各K/N、逐RX值见原报告与同目录result_summary.json。共新增1960+560+560+560=3640行正式评分；多个行复用物理query，不是3640次独立物理试验。R8仅完成源端筛选，目标IQ/预测/评分0，其source_summary.json保留完整失败判定。

## 实现与版本

- `experiments/pamf_phase2_practical/milkr/semantic_residual.py`：固定语义多尺度核、默认头残差和B函数继承。r01执行commit `01395832c4d3c1f116cf3f07f86354c9864d248a`。
- `milkr/support_calibrated.py`：24核×5ridge；K1源规则，K>1当前support条件LOO。r02执行commit `d6f505c227b98057a18960cad78f4d352f38ce6b`。
- `milkr/pooled_metric.py`：合并类内散度核Mahalanobis质心判决；LOO从scatter及centroid同时移除该物理样本。r03执行commit `d5d874cc73afd97f7a3d1dba7a5f4ade6524d251`。
- `milkr/group_calibrated.py`与`milkr_group_source.py`：源K1系数、支持LOO旧新偏移，预登记源阈值失败即0目标访问。r04执行commit `e144ae51dda60fe5838294bd33e133e76a7753df`。
- `milkr/signature_kernel.py`与`signature_features.py`：真实output_projection输入捕获；三112维分别单位化。r05执行commit `9db994cc5c60d3fd56aac4aec824b88f4940714f`。真实checkpoint smoke核对336维、原z/logits逐位一致、F/G更新0。源K1规则signature_identity/all/global0.5/single/ridge0.001；24核×5ridge保持源/支持选择。
- `milkr_screen.py` / `milkr_cv_screen.py`：源冻结→目标预测→独立评分。已释放快照不可覆盖、不可热改。

源码/测试/配置已在 `code/worktrees/standard_ce_baselines_20261010` 及发布承载面镜像；其他owner未提交更改保留。机器状态和完整私有产物不提交。根实验索引已登记三run，实际状态以各run证据为准。

## 最新四路验收与后续证据

用户追加：support必须严格区分接入/不接入DURA；两组各自要求query不接入时比原版高，接入query后更高。原+5个百分点目标保留。当前按16个K/N等权平均核算，同时完整报告每格；若用户指定每格+5则按更严格口径。R9的S0Q1大幅下降，不能作为合格替代。

- R10任务归一化（r06）源端none胜出，最优非none平均-0.2285pp，target0。
- R11支持Fisher全局强度（r07）源端强度0胜出，其余均下降，target0。
- R12物理支持outerCV（r08/r09）两次技术失败：源旧Torch NumPy索引、目标审计np.bool_ JSON。均已精确定位、定点修复与复审，尚无任何有效prediction/评分；r09曾接触B/K1 support，不能声称target未访问。旧root保留，所属进程全部退出。
- r10仅预备未发布，因用户四路要求替代旧raw-only晋级标准，STOPPED/NOT_LAUNCHED。远端release/run均不存在，无健康进程被停止。
- 新R13（sourcebridge r11）：合法sourceL paired raw/restored特征学习固定rank8残差映射；目标raw路映射到restored特征空间，真实G路不映射。严格S0/S1分别处理，不混入另一支持路。R12头与同head无映射R12C控制，首RX1-1覆盖全部K/N/四路，共160行。先导未达到两个S的Q0+5pp且Q1>Q0则不扩展。映射是离线参数checkpoint，目标不访问源样本特征；不改F/G或默认分类头。

goal保持ACTIVE。没有候选满足最新四路条件与原+5pp要求；不能以技术发布完成代替科学达成。新run状态以其报告和独立post-state为准。

### R13先导实际证据

源6300物理记录/12600配对观察，五源RX留出pair MSE均下降，实际checkpoint/数值smoke通过。P01/MIX01/首RX1-1全部160预测与评分完成，所属进程退出。

- S0 Q0较原版+6.032pp，query接入DURA增量-9.471pp；均值验收False，每格验收False。
- S1 Q0较原版+27.024pp，query接入DURA增量+25.389pp；均值验收True，每格验收False。

R13是否扩展：False。完整四路/各K/N及同head R12C控制见sourcebridge r11报告和result_summary.json。该单RX证据不代表goal完成；不进行目标反馈调参或选择性重跑。

### R14待验证机制

R13仅映射raw仍残留单侧失配。新R14拟合source-only配对CCA，两路各自编码到共同空间，严格Q0只raw/Q1只G，无混合query。rank由留一sourceRX的四路H事前门槛选择；未通过则source-only停止、目标0。通过才P01首RX全部K/N/四路80行。没有降低用户+5pp与Q1>Q0要求。

### R14实际源端证据

- rank16：源四路较R13 +1.483pp，源晋级True。
- rank32：源四路较R13 -0.276pp，源晋级False。
- rank64：源四路较R13 -1.546pp，源晋级False。
- rank96：源四路较R13 -2.268pp，源晋级False。
- S0：Q0较原版+7.070pp，Q1增量+0.592pp；均值验收True，每格验收False。
- S1：Q0较原版+38.763pp，Q1增量+15.916pp；均值验收True，每格验收True。

首RX先导晋级：True。不以singleRX声明goal完成；详见pairedcca r12报告。

### R14固定参数确认

首RX四路均值门槛通过，r13按事前规则确认其余六RX全部K/N。直接复用r12源rank16参数，无目标反馈重训/改参。分别报告confirm6与combined7，逐KN/每RX保留。pilot S0每格尚未全部达标，不能外推每格均+5pp。

### R15当前下一步

固定R14源CCA16，校准当前support物理OOF旧/新组H阈值，并按K收缩；none/K1/B保持R14。源oldcounts2..5的组数不对称代理选择全局kappa，sourcegate失败target0，通过才首RX全KN四路80行。sourceold1带宽P1在发布前被独立审查发现并移除，边界测试/定点复审PASS；目标old6不改。当前最佳R14七RX S0Q0仍+4.110pp，goal ACTIVE、未达+5。
