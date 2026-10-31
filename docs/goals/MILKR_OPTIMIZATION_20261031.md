# MILKR 宏 H 提升目标与当前证据

用户目标：固定 **Tri-SIG-TIME SSL** 基座，优化 MILKR，使宏 H 比匹配原版提高至少 5 个百分点；可用适当小矩阵，覆盖各 K/N。状态：**ACTIVE，未达成**。不得把实验技术完成写成科学目标完成。

## 最新用户验收：每个K/N格（覆盖旧均值口径）

用户已明确回复“每个K/N格都至少+5个百分点”。K1/5/10/20×N2/5/10/20每格分别列support S0/S1，Q0与匹配原MILKR差至少5pp，且接入query DURA的Q1高于当前Q0。完整四路各自对原版+5pp结果也逐格检验；均值通过不能替代单格，不用跨support或K/N的高收益弥补失败格。多RX表格的每个K/N先对固定RX等权汇总，并保留每RX；不擅自把要求扩大成每条query都提高。

R14七RX S0Q0仅6/16格≥5pp、Q1>Q0仅11/16，严格目标明确未达成；mean+4.110也未达。R15source-only停止。R16在用户答复时源阶段健康运行，现已按源门槛正常停止，参数/C选择/矩阵不热改，只按最新逐格要求分析和决定是否扩展；不能再用mean-only先导晋级。goal保持ACTIVE。

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

### R15实际证据

- kappa0.0源四路较none-0.056pp，源通过False。
- kappa5.0源四路较none+0.107pp，源通过False。
- kappa20.0源四路较none+0.155pp，源通过False。

source-only停止，target0。目标仍ACTIVE，当前七RX R14 S0Q0+4.110pp；不扩展R15。

### R16当前下一步

R15校准源最佳+0.1547pp未过0.5pp，正常source-only停止target0。R16是明确的同核C-SVC分类目标对照（不冒充原ridge未改）：固定CCA16/F/G/神经默认头，source四C选择，通过才target首RX全KN四路80行。sourcegate/用户5pp门槛不改，source六类已见F和目标26类外推局限保留。新候选审查/针对性检查PASS。当前七RX最佳主路径仍R14+4.110pp，goal ACTIVE。

### R16实际证据

核C-SVC四C均未通过源门槛，正常source-only停止，target0，F/G更新0，所有所属进程退出。非ridge分类目标对照不晋级。用户已明确每K/N格≥5pp；R14七RX raw仅6/16满足、Q1高于Q0仅11/16，不能以mean过关，goal ACTIVE。下一步优先研究源L类内噪声协方差形成的冻结线性metric，作用于R14共同空间且保持ridge判决，先source K1及多shot验证，未实现/未发布，不保证收益。

### R17当前候选

源L类内噪声metric融合入R14原CCA16两路projection权重，保留ridge/nested头与F/G。原σ候选0/.25/.5/.75/.9，源五heldRX×固定epoch91×K1/5/20优先验证K1；训练用两epoch并排除heldRX，无target泄漏。无cov/classmeans/prototypes上传，仅冻结encoder参数。sourcegate未过target0；通过才首RX全部16KN四路80行。按用户每格Q0+5且Q1>Q0判定，mean不能晋级。唯一审查/合成检查PASS，准备发布；不保证收益。

### R17实际结果

源s0.9门槛PASS，源K1两S Q0+9.5492/+4.5899pp，实际首RX80预测/评分完成。S0Q0相对原版均值+6.7692pp，但Q0≥5仅11/16，Q1>Q0仅10/16；S1两项均16/16。按最新逐格要求不晋级，无其他六RX实验。源known6收益不能外推new20，goal仍ACTIVE。原R14七RX证据保持，source class-noise model不改默认。下一步需要更接近未见类别的源验证或新的支持集几何，不以目标分数选择σ重跑。所有所属进程退出。

### 下一步：适配器TX留出源代理与K1重新校准（尚未实现/登记/发布）

R17单RX同case分析显示：K1/N20对原版仅+0.652pp，K20四N增益为+4.248/+3.161/+3.103/+3.622pp；源known-six K1+9.55pp不能代表未见20类。不能按这些目标数值挑σ重跑。下一候选应改变源验证方式，减少已见身份对源超参数的偏好，并校准新增编码器真实几何下的K1核。

建议事前固定五折，每折同时留一个source RX和两个source TX不参加CCA/类内metric训练；RX按原排序，held TX对按源码类序循环(0,1),(1,2),(2,3),(3,4),(4,5)，不据IQ/成绩选择。source F本身已见六TX，这只能叫adapter-TX-heldout，不能声称F从未见或真novel。每fold在其它RX的四个old TX上训练source encoder，以heldRX的old4+new2合法support/query评价。仍仅source L与既有ep91/92；评估固定ep91、sourceK1/5/20，target仍全16KN四路。

可预先比较CCA rank16/32/64/96与metric强度0/0.5/0.9；原R17 helper只接受rank16/完整六类，需要参数化rank并显式允许source class子集，类均值索引改为unique inverse，保留现有六类默认合同和旧行为。最终encoder只用全source L训练，输出普通PairedCCA参数，不上传源cov/类均值/样本。每fold的held TX从CCA与metric训练均完全排除，而不只隐去标签。

K1目前一直沿用R9在额外336维但未CCA前选的signature_identity/all/mix0.5/single/ridge0.001，新增几何未重新校准。可在adapter-heldout源代理上对既有24核×5ridge完整重选K1规则，K>1保持当前support R14 nested选择。新的K1 ridge可明确定义为mean-loss归一化（以source6个支持类为reference，target K1按当前support类数/6缩放），必须测试/披露，不能按目标N逐格手设。若不采用该归一化则明确保持绝对lambda，二者不混称。

源选择必须同时看held新TX macro而非仅已见类平均H：例如四路H≥control+0.5pp、new macro≥control+1pp、各源K/S Q0均有正收益并保留Q1优势，具体门槛在运行前固定。不要将只改变K1而K>1数学上等于R14的候选当作能解决已知K20所有失败格的完整方案。全局source候选固定后，首RX全16格；按用户每格Q0+5pp/Q1>Q0决定晋级，不再mean-only。

这只是下一条可证伪假设，不代表已实现或保证收益。R17旧root/模型/预测保留；当前无所属远端进程。goal ACTIVE，尚未达标。

### R18当前实现

已实现adapter-TX/RX-heldout源代理、rank16/32/64/96×metric0/.5/.9、每表示K1 24核×5ridge重新校准，K1按support类数/6缩放lambda、K>1原R14不变。F本身仍见过六sourceTX，不能冒称F-novel。源gate同时要求held-new和每个sourceK收益，不能只改善K1。sourcepass才首RX全16KN四路80行，仍按每格验收；唯一审查/针对性检查PASS，准备发布，不保证收益。

### R18实际源证据

1440候选全部完成，唯一源参数搜索不访问target，按原完整门槛正常source-only停止、target0、所属进程退出。rank32/metric0.9源四路宏H+3.851pp、new macro+2.474pp、各K/S Q0宏H最小+0.871pp，但K1 new S0Q0 −1.5625pp，没过额外K1新类每S+1pp，原run不晋级。该分量要求比用户宏H目标更强，未来如改用仅宏H对齐的源选择必须另建run、保留本失败，不改原门槛/输出；所有参数选择仍只源、目标未接触。当前严格目标ACTIVE。

### R19新的源宏H选择策略

原R18严格aux源门槛失败保留；另建R19按用户宏H目标重用同一source-only1440grid，不用target选择参数。取消仅source每K1new单项+1的必要条件，保留everyK rawH正、queryG不降、overallnew正和old/new均值guard。新source候选唯一配置后全sourceL拟合encoder，目标每KN+5/Q1更高完全不变。新增输入复用路径审查/测试PASS，准备发布；不是原run放宽重跑，不保证target收益。

### R19实际结果与当前结论

源宏H策略选rank32/metric0.9/K1 signature_identity-nearest-mix0-single/refλ10（当前support类数/6缩放），源old/new raw均值guard通过，原R18aux失败不改。首RX1-1全部80预测/评分完成、所属进程退出。S0Q0对原版均值+7.3475pp，但每格+5仅11/16、Q1>Q0仅11/16；S1均16/16。严格不晋级，无其他六RX实验。S0 K1/N20仍仅+1.849pp，K20四N约+3.886/+3.095/+3.285/+3.180pp；queryG在K5/N2、K10/N2、K20/N2,N5,N20仍低于Q0。不能按这些目标分数挑另一个source候选重跑。R14七RX仍为最广证据，R19只是单RX；goal ACTIVE。

下一步待实现的物理假设：现有线性raw/G共同空间不能完全消除G引起的相位/表征变化，且source6类监督metric可能丢失未知硬件细节。考虑固定复相位群的计算型特征：同一个received记录（S1/Q1仅一次G输出）在F内部做若干固定相位旋转，共享原F权重，逐记录取神经特征轨道均值和标准差，再用source paired线性encoder映射回原6字段、保留ridge头。旋转是当前同一观测的确定性计算，不重抽信道、不增加K/物理观测、不混入S0/S1数据，G只在原始输入运行一次，源/目标query不得跨记录统计。

若实施，优先C8（八等分相位）或先C4原型，需证明群旋转不变性/逐query独立及G先于轨道、S0不调用G，FFT相位不变可复用一次，F/G/neuraldefaulthead冻结。均值+std保留分布形状，避免简单均值抵消RF信息；新输入维度是原来的2倍，可训练专用paired encoder输出原schema，rank64作为事前固定表示预算而非target选rank，源不使用TX类内监督metric；K1核仍只source校准。源F见过六TX、代理不是真F-novel，不能因source无正收益自动排除一个固定物理先验对照；具体方法/停止规则须在运行前登记。可先目标四corner(K1/K20×N2/N20)四路24行（含B），若未过每格+5与Q1>Q0不扩；通过才同冻结模型补全16格，最终要求不缩小。此方法尚未实现/登记/发布，不保证收益。

### R20 固定 C4 相位轨道特征

四相位同记录计算、G先于orbit、每相位单位特征的均值/标准差；source unsupervised pairedCCA rank64 固定，无classmetric，source RX/TXheld只选K1规则，F/G不更新。先K1/20 N2/20四角点四S/Q24rows，每格两S Q0+5/Q1更高才同模型补剩余12KN；最终仍需16KN全过再扩RX。源门槛仅数值/合法性，不以重复修改旧gate代替新特征验证。

### R20 C4 实际结果

真实checkpoint smoke及独立审查/测试通过；source rank64和C4固定、source120核规则选唯一K1。目标24行评分完成，所属进程退出。S0角点Q0+5为3/4，Q1>Q0为1/4；S1角点Q0+5为4/4，Q1>Q0为4/4；角点晋级False。不扩其余KN/RX，不用目标结果调rank或相位数。基座仍Tri-SIG-TIME SSL，goal ACTIVE。详见本run result_summary/report。

### R21 完整C4分布核

新候选保留同记录四相位完整分布、source同phase raw/G PairedCCA64；16phasepair核值均值而非moment压缩，物理K/标签数不变，G仍原包先执行一次。source-heldRX/TX只选K1规则，K>1物理support条件LOO，无Fisher；无query统计/selfdiag校准。固定同四角点24行证伪，每cornerbothS Q0+5/Q1>Q0才补12KN，最终all16目标不变。R20失败保留不扩展。

### R21 C4完整分布核实际结果

真实checkpoint smoke及独立审查/测试通过；source rank64和C4固定、source120核规则选唯一K1。目标24行评分完成，所属进程退出。S0角点Q0+5为2/4，Q1>Q0为1/4；S1角点Q0+5为4/4，Q1>Q0为4/4；角点晋级False。不扩其余KN/RX，不用目标结果调rank或相位数。基座仍Tri-SIG-TIME SSL，goal ACTIVE。详见本run result_summary/report。

### R22 有界非线性配对编码

新source-only非线性残差：C4moments/CCA64不变，八block raw/G2d-64-64有界MLP初始等价CCA、幅度≤.25。只同记录配对MSE+.1anchor，固定600step/256/AdamW1e-3，F/G/head不更新。每sourcefoldCCA/NN均excludeheldRX/twoTX，source四H+.5/rawguard/querygain门槛失败target0，通过才全sourcefit及严格G-only4corner24rows。待询问的dual-query口径不影响本run，不热改。最终16KN两S每格+5/Q1更高不变。

### R22 实际结果

五sourcefold各600step有界配对NN完成，F/G/head更新0；source四H相对control+0.2640pp，gate=False。源科学停止、target0、最终checkpoint未拟合；不能用pairMSE代替分类H。全部所属进程退出；strictG-only Q1保持；最终每16KN+5及queryDURA收益目标仍ACTIVE。

### R23 全注册support带宽

新结构变体tau由当前全部注册物理support估计，old_count仍只定义old/new宏H角色，query不入尺度。C4moments/CCA64固定，freshsource五RX/TXheld只选K1规则，原nested训练折尺度；旧默认old不变，同配置B同公式但新source规则可改实际B。无R22NN/不改旧stop，源仅合法数值门槛后同四角点24行，最终16KN两S每格+5/query收益不变，strictG-only保持。

### R23 全注册support带宽实际结果

真实checkpoint smoke及独立审查/测试通过；source rank64和C4固定、source120核规则选唯一K1。目标24行评分完成，所属进程退出。S0角点Q0+5为3/4，Q1>Q0为1/4；S1角点Q0+5为4/4，Q1>Q0为4/4；角点晋级False。不扩其余KN/RX，不用目标结果调rank或相位数。基座仍Tri-SIG-TIME SSL，goal ACTIVE。详见本run result_summary/report。

### R24 局部support核

固定当前每SUP一centre局部RBF显式特征，tau由其他类SUP定、列norm仅SUP、逐record行unit，形成PSD核；预合并localW查询不拟合。K1源120规则，K>1真实物理三折OOF全train-only，再当前全SUP重拟合，无Fisher。F/G/head/C4moments/CCA64保持，strictG-only，四角点24行每格两S+5/query收益才扩，最终16KN不变。

### R24 局部support核实际结果

真实checkpoint smoke及独立审查/测试通过；source rank64和C4固定、source120核规则选唯一K1。目标24行评分完成，所属进程退出。S0角点Q0+5为3/4，Q1>Q0为0/4；S1角点Q0+5为4/4，Q1>Q0为4/4；角点晋级False。不扩其余KN/RX，不用目标结果调rank或相位数。基座仍Tri-SIG-TIME SSL，goal ACTIVE。详见本run result_summary/report。
