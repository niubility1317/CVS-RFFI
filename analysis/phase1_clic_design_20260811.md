# Phase1 P1-CLIC冻结设计与实现追踪卡

日期：2026-08-11

状态：<code>DESIGN_FROZEN / USER_APPROVED_SINGLE_RECEIVED_I_AND_TARGET_REGISTERED_UNKNOWN_LEO_REQUIREMENTS / INDEPENDENT_REVIEW_P0=0_P1=0 / NO_IMPLEMENTATION / NO_N607 / NO_PERFORMANCE_RESULT</code>

本卡冻结P1-CLIC（Complex Local Invariant Curvature，复数局部不变曲率）的科学假设、数据权限、数学算子、模型接口、公平对照、技术回执、后冻结五门和deployment bundle合同。它只授权后续编写实施计划；不构成代码已经实现、N607已经运行、Phase1候选已经晋级、真实unknown拒识已经完成或Phase3已经完成的声明。

## 1. 用户批准与协议审计

用户的批准条件是：从同一份<code>received_i</code>提取固定lag集合<code>{1,2,4,8}</code>的多尺度三点复曲率token不得违反Phase1数据协议。审计结论为不违反，理由如下。

1. <code>项目.md</code>第4节允许Phase1使用source clean与卫星信道增强视图训练，也允许重训Sinc／时域前端、<code>z_id</code>、<code>z_dom</code>、normalization、projection和fusion。
2. <code>项目.md</code>第5.2节明确规定：从同一固定接收IQ计算归一化、FFT或其他数学表征仍属于同一个物理样本，不增加K。CLIC的四个lag只索引同一个IQ序列中的时刻，不调用LEO信道模拟器，不生成第二份<code>received_i</code>，不复制物理样本，也不增加shot或K。
3. Phase1训练只读取source-L的合法TX／source receiver信息；U、V、source proxy、target、正式registered query和正式unknown query均不参与训练forward、更新、校准或选择。
4. 正式Phase2／Phase3输入仍遵守单物理样本单LEO观测：registered与unknown每个物理样本各自只读取一份在truth／role前冻结的<code>leo_*_weak received_i</code>，scene和seed使用同一规则。禁止多scene择优、融合、重采样或把clean unknown写成正式结果。
5. CLIC deployment bundle不得保存raw IQ、单样本feature、样本ID cache或历史unknown query。

因此，CLIC是接收后的单观测数学表征，不是新增数据view或协议外数据生成。用户据此批准冻结设计。

用户随后补充实验判定口径：每个候选都必须报告叠加LEO weak星地信道的目标接收机域测试，Phase1继续以未知类拒识潜力和域泛化为同等目标。该要求不改变source-only训练权限：

1. 每个F1至F6的C／G checkpoint都必须在同一封存目标接收机capsule上完成一次零适配推理；<code>R_t∩R_s=∅</code>，C／G共享完全相同的<code>physical_sample_id</code>、scene、seed和<code>received_i</code>字节SHA。
2. capsule同时包含互斥的<code>target_registered_known</code>和<code>target_unknown</code>物理样本集合。每个物理样本只绑定一份在truth／role前确定的<code>leo_*_weak received_i</code>；registered／unknown使用同一scene／seed生成规则，三个scene使用互斥物理样本分区。禁止从同一物理记录生成三份LEO观测再平均、择优或融合。
3. 模型侧运行器只生成不可变预测、拒识分数和质量输出，不读取target标签、真实角色或scorer结果，不执行support适配、参数更新、阈值拟合、温度校准、prototype更新、早停、回滚或重试选择。分类头、类别几何、radius／energy／tail和拒识决策规则全部只由source证据冻结；12份预测artifact全部封存后，隔离truth-side scorer才可连接标签与registered／unknown角色。
4. 目标域LEO registered-known指标确认跨接收机域泛化；TX互斥source-proxy unknown指标仍是Phase1候选冻结前的拒识研发信号；冻结后的target registered／unknown共同LEO blind-confirmation报告单节点拒识结果。三者都必须同row报告，任何一组都不能由clean或另一组的平均值替代。
5. target blind-confirmation只允许发生一次，结果不得反馈训练、结构、阈值、校准、checkpoint选择、候选排序、重跑或路线复活。它是Phase1冻结表征的单节点确认性证据，不等于Phase3多节点协同、anonymous entity关联、可信确权、注册授权或运营unknown能力。

## 2. 科学假设与可证伪边界

CLIC只主张：全局复增益、常相位和线性相位漂移是弱星地接收链中的共同扰动；同一接收IQ内部的三点复曲率可解析消除这三类共同量，同时保留更局部的幅相曲率供身份编码。它不主张消除时变多径、周期干扰、接收机非线性或所有LEO信道效应。

最大反例是：时变多径或周期干扰本身形成稳定曲率，而TX身份恰好主要存在于被消去的共同幅相量中。此时CLIC会把信道纹理写入<code>z_id</code>，或删除有用身份信息。clean／LEO最差切片、source-proxy双门和真实bundle门必须直接否证这种失败；不得用平均性能、挑fold、挑scene、query反馈或阈值补偿。

CLIC不是ICMT、CAGM、RCRMD、RCAT、RECTE、HSCF、RCMMC或HNCCD的改名、重加权或拼接，也不复活CB-SFCE、CP-SFCE、GD-ProtoNLL、CCPC、PAMR、dual-readout／disagreement、OE、Q98、teacher或clean—LEO附加对齐。它不增加第九个局部geometry、tail、moment、covariance或proxy loss；改变点只在exact分类头之前的单观测表征路径。

## 3. 输入、定义域与固定常量

输入是已经pad／crop到同一长度的实数IQ张量<code>x∈R[B,2,T]</code>。WiSig冻结长度为<code>T=256</code>，实现必须拒绝<code>T&lt;17</code>。对每个样本和时刻定义

    x_t = I_t + jQ_t
    a_t = hypot(I_t,Q_t)
    p_t = x_t / a_t
    L = {1,2,4,8}

在计算任何mask前，完整输入必须全部finite。对lag <code>l</code>和位置<code>t</code>，定义域<code>D_l</code>同时要求：

- <code>l≤t&lt;T-l</code>；
- <code>a_(t-l)&gt;0</code>、<code>a_t&gt;0</code>、<code>a_(t+l)&gt;0</code>；
- <code>hypot</code>、除法、乘法和对数的所有中间量均finite。

越界位置或输入finite但三点中至少一个精确零幅位置属于预注册数学定义域外，只能产生zero-token和zero-mask。任何NaN、Inf或由有限输入计算出的非有限中间量必须fail-closed，不能静默置零、clamp回有限值或回退base。

固定常量如下：

|常量|冻结值|
|---|---:|
|lag集合|<code>{1,2,4,8}</code>|
|token通道数|<code>4×4=16</code>|
|幅度曲率clip|<code>[-8,8]</code>|
|identity维度|<code>d=160</code>，漂移即拒绝|
|CLIC模块初始化seed|<code>7281164</code>|
|训练batch size|<code>128</code>|
|本地类别数|<code>4</code>|
|训练epoch|<code>40</code>|
|正式LEO场景|<code>leo_clear_weak</code>、<code>leo_low_elev_weak</code>、<code>leo_rain_weak</code>|

## 4. G算子、C算子与不变性证明

### 4.1 G：复数局部不变曲率

在<code>D_l</code>内定义

    u_l(t) = p_(t+l) p_(t-l) conj(p_t)^2
    h_l(t) = log a_(t+l) + log a_(t-l) - 2 log a_t
    r_l(t) = min(a_(t-l),a_t,a_(t+l)) / max(a_(t-l),a_t,a_(t+l))

G-token为

    T_G(l,t) = [Re u_l(t), Im u_l(t), clip(h_l(t),-8,8), r_l(t)]

无效位置四个分量均为零，并由独立valid-mask标记；valid-mask不能从token数值反推。

### 4.2 C：同shape原始相位对照

C使用同一输入、lag、valid-mask、可靠度、编码器和融合路径，但不计算三点曲率：

    T_C(l,t) = [Re p_(t+l), Im p_(t+l), Re p_(t-l), Im p_(t-l)]

<code>r_l(t)</code>仍按同一公式只供质量gate和receipt使用。C／G token shape均固定为<code>[B,16,T]</code>。

### 4.3 解析不变性

令

    x'_t = α exp(j(φ+ωt)) x_t,  α∈C, |α|>0

则

    p'_t = (α/|α|) exp(j(φ+ωt)) p_t

在<code>u_l(t)</code>中，两个邻点相位与中心点二次共轭相位相消：

    (φ+ω(t+l)) + (φ+ω(t-l)) - 2(φ+ωt) = 0

复增益单位相位的指数也为<code>1+1-2=0</code>，故<code>u'_l(t)=u_l(t)</code>。幅度满足<code>a'_t=|α|a_t</code>，所以<code>h_l</code>中的三个<code>log|α|</code>系数为<code>1+1-2=0</code>，且min／max比例中的<code>|α|</code>相消。因此在定义域内<code>u</code>、<code>h</code>和<code>r</code>均严格不变。该证明不使用epsilon、pinv、拟合或数据统计。

## 5. CLIC编码器、融合和输出

令<code>d=160</code>，两臂共享同一参数化模块：

1. depthwise <code>Conv1d(16,16,kernel_size=5,padding=2,groups=16,bias=false)</code>；
2. <code>GroupNorm(4,16)</code>与<code>SiLU</code>；
3. pointwise <code>Conv1d(16,32,kernel_size=1,bias=false)</code>；
4. <code>GroupNorm(8,32)</code>与<code>SiLU</code>；
5. 按valid-mask做严格masked mean；分母为有效位置数，零有效位置输出精确零；
6. <code>Linear(32,d,bias=true)</code>得到<code>e(T)</code>；
7. <code>W_c=Linear(d,d,bias=false)</code>；
8. gate为<code>LayerNorm(2d)</code>后接<code>Linear(2d,1)</code>和sigmoid。

existing exact-head输入<code>feat_joint</code>记为<code>z_base</code>。定义

    r_bar = valid位置上的r均值；无valid位置时为0
    gamma = r_bar × sigmoid(w^T LN([z_base;e(T)]) + b)
    z_id = z_base + gamma W_c e(T)

分类只调用现有exact CosFace head一次：

    tx_logits = exact_head(z_id)

禁止第二readout、第二head、第二模型forward或并行分数融合。existing domain backbone继续输出<code>z_dom</code>，CLIC不新建domain分类loss。节点质量输出固定为

    q_clic = [gamma, r_bar, valid_fraction, full_fallback_flag]

其中<code>full_fallback_flag=1</code>只表示当前样本所有lag／位置均在数学定义域外；它不是unknown标签或拒识阈值。

初始化严格固定：depthwise／pointwise／E采用seed <code>7281164</code>下的Kaiming规则，<code>W_c</code>使用<code>0.01×orthogonal</code>，gate权重<code>w=0</code>，gate bias为<code>logit(0.1)</code>。初始化过程必须保存并恢复调用方RNG状态。每个matched C／G pair的新增模块初始state SHA必须相同；最终训练权重不要求相同。

在<code>d=160</code>时新增参数固定为32529个：depthwise 80、两组GroupNorm共96、pointwise 512、E 5280、<code>W_c</code> 25600、LayerNorm 640、gate 321。C／G参数数、forward次数和额外张量shape完全一致。

## 6. Phase1训练权限与C／G公平

训练run ID冻结为<code>phase1_clic12_20260811_v1</code>，候选为<code>F1C_CLIC12</code>至<code>F6G_CLIC12</code>。C／G使用同一GeoSat-C final-only warm start、source fold、local4类顺序、物理batch顺序、seed、sampler、40E、new AdamW、AMP和existing共同<code>L_base</code>。每个source-L训练物理样本只使用existing clean和当批唯一单LEO视图；CLIC不调用信道模拟器，不增加view。

C／G均启用CLIC模块并训练全部相同参数；唯一方法变量是<code>operator_mode=raw_phase_control</code>或<code>operator_mode=complex_local_invariant_curvature</code>。所有旧Phase1候选开关和loss权重必须为false／0。U零iterate／zero-forward；V、proxy、held、target、正式unknown、day、fold和scorer均零训练、校准、选择或状态反馈。目标域LEO测试只允许在final checkpoint与预测输出都不可变后执行，不能改变这一训练权限。

baseline checkpoint的existing keys必须逐项strict加载。新增CLIC keys只能按本卡固定seed和初始化规则生成；optimizer state、RNG state和旧方法state均不得从checkpoint恢复。new AdamW必须同时包含existing trainable参数与CLIC参数。

## 7. VJP、AMP和图生命周期

每个C／G臂在clear、low-elev和rain三个scene中，分别对首个valid且<code>gamma W_c e(T)</code>非零的批次审计一次raw-unscaled共同<code>L_base</code>：

- <code>∂L_base/∂T_C</code>或<code>∂L_base/∂T_G</code>必须finite且总范数非零；
- depthwise／pointwise／E、<code>W_c</code>和gate参数VJP必须finite且总范数非零；
- existing base shared encoder参数VJP必须finite且总范数非零；
- exact CosFace head权重VJP必须finite且总范数非零。

token、<code>hypot</code>、单位相位、对数、曲率、masked mean和gate质量统计固定在autocast外以FP32计算。诊断只能对同一图使用<code>autograd.grad</code>，随后每批只允许一次正常scaled backward、一次unscale和一次step／update；不能为审计增加第二forward或第二optimizer update。

无论finite step或普通AMP overflow skip，纯标量telemetry物化后、下一forward前都必须释放所有output、token、mask、loss、VJP和日志tensor的连图根；禁止<code>gc</code>、<code>empty_cache</code>和跨批feature cache。

## 8. 错误处理与技术回执

配置schema固定为<code>cvs.phase1.clic_receipt.v1</code>，terminal schema沿用同一method identity。任一以下情况必须写data-free failure receipt并终止该arm：

- 输入或任一中间／输出nonfinite；
- 输入长度、token shape、<code>d=160</code>、lag、clip、module init SHA或operator identity漂移；
- baseline strict keys、class order、source split、physical order、scene或C／G共同绑定失败；
- G或C任一scene没有valid token批、没有非零gate批或VJP未闭合；
- optimizer／AMP事件、资源观察或图释放账本缺失；
- 旧方法开关、target／proxy／held／U训练访问或第二LEO view非零；
- terminal前CLIC checkpoint state、exact-head path或receipt SHA不闭合。

receipt只保存标量、计数、枚举和SHA，不保存raw IQ、receiver token、physical ID、单样本feature、token张量或成员清单。至少封存：

- operator ID、lag、clip、input length、token shape、<code>d</code>和初始化state SHA；
- source split／class order／physical order／common batch sequence SHA；
- 每scene×lag的valid、zero-mask、full-fallback、非零gate、gate均值／范围；
- C／G每scene的token路、CLIC参数路、base路和exact-head VJP范数／计数；
- AMP attempt、effective step、raw-finite overflow skip、raw／material nonfinite和terminal consecutive skip；
- 每common batch一项peak CUDA bytes、step-time和<code>selection_feedback=false</code>；
- final checkpoint SHA、terminal contract和failure stage。

训练receipt不得写入target指标或target成员信息。后冻结target测试另用<code>cvs.phase1.clic_target_leo_eval.v1</code>封存capsule／split／scenario assignment／received-IQ聚合SHA、checkpoint／bundle SHA、source-only决策规则SHA、预测SHA、truth-side scorer SHA、逐scene计数和指标；不得保存raw IQ、成员ID或把scorer输出回写checkpoint。模型侧prediction artifact不得包含label、真实registered／unknown role、receiver／day真值或truth-sidecar路径。

## 9. 资源合同

主token张量固定为<code>[B,16,T]</code>，FP32主占用为<code>4×B×16×T</code>字节；在<code>B=128,T=256</code>时为2097152字节。实现不得物化<code>B×T×T</code>、<code>B×16×T×T</code>或跨批feature cache，不得增加第二LEO view、epoch、采样、persistent prototype或query state。

C／G逐batch记录相同口径的CUDA peak和step-time。资源只用于证明公平和预算，不得用于挑fold、挑scene、反选operator或重启实验。

## 10. Deployment bundle合同

bundle schema固定为<code>cvs.phase1.clic_deployment_bundle.v1</code>。每个G fold的真实final checkpoint必须可导出一个不可变bundle，至少包含：

- exact feature extractor与CLIC／existing domain分支state；
- <code>operator_mode=complex_local_invariant_curvature</code>、lag、clip、zero／nonfinite policy和模型结构配置；
- <code>z_id</code>、<code>z_dom</code>、<code>q_clic</code>的维度与语义；
- 只由source-L聚合的已注册类基础几何、radius、energy／尾部先验；
- checkpoint、代码、配置、source split聚合身份和bundle manifest SHA；
- <code>clean_source_runtime_access=false</code>、<code>query_fit_access=false</code>和<code>single_leo_observation_required=true</code>。

bundle不得包含raw／clean IQ、单样本feature／logit、source replay、成员ID、unknown query、proxy rows、target rows或可替换sidecar。bundle通过只表示Phase1交付物形成，不表示真实unknown拒识或Phase3完成。

每个bundle还必须支持对封存<code>p2_min_v1</code>目标capsule进行无support、无更新的<code>reload→forward</code>。该接口输出<code>z_id,z_dom,q_clic,tx_logits,e_unknown,decision∈{registered,unknown,defer}</code>与状态SHA；<code>e_unknown</code>方向、threshold和defer规则只由source-L／source validation几何冻结，目标域真值和角色不进入bundle或模型进程。

## 11. 后冻结矩阵与五项非补偿门

后冻结run ID固定为<code>phase1_clic_postfreeze_20260811_v1</code>。保留既有42步：12 source clean export、12 source LEO export／binding、12 fixed400 TX互斥source-proxy和6 same-fold pair；再增加6个G deployment bundle export，并为12个C／G final checkpoint各执行1次封存目标capsule的registered／unknown共同LEO weak零适配推理／隔离评分，总计60步。阶段计数固定为<code>12+12+12+6+6+12=60</code>，不因target unknown增加训练臂、fold、epoch、checkpoint或反馈式重试。F6必须重开F1—F5原始source clean／LEO／binding／proxy／checkpoint／bundle和12份target预测／评分原件，不能信任prior pair自报摘要。

目标capsule必须在启动前固定<code>protocol_schema=p2_min_v1</code>、<code>capsule_id</code>、<code>split_id</code>、<code>R_t</code>集合SHA、registered-known／unknown TX集合SHA、两种角色合并后的physical-ID集合SHA、三scene物理ID两两不交分区SHA、truth／role-blind scene／seed assignment SHA和received-IQ聚合SHA。C／G及六fold共用这些字节；任何checkpoint、operator或运行状态变化不得触发数据重建、数据重验证或重新分scene。offline sealer只复核既有builder／validator receipt并输出IQ-only predictor package与隔离truth sidecar；模型进程只可见opaque token、scene和received-IQ SHA。每个target阶段只允许一次backbone forward／样本，12份输出全部按字节SHA封存并验真后，truth-side scorer才可独立读取标签与角色。

F6 raw reopen还必须逐项重验上述<code>capsule_id／split_id</code>、scene／seed assignment、三scene物理ID互斥、received-IQ聚合SHA、C／G共同字节绑定、source-only决策规则清单／SHA、12份prediction artifact SHA和scorer代码／输入／输出SHA。它只读取既有builder／validator receipt和小型封存artifact，不重复打开raw IQ或重复执行数据验证。

每个C／G row必须同时报告三组结果：

- <code>source_open_world_proxy</code>：TX互斥source validation／proxy上的AUROC和<code>u_gap=mean(u_proxy)-mean(u_V)</code>，仅代表Phase1 unknown拒识研发信号；
- <code>target_leo_weak_registered_dg</code>：对<code>R_t</code> registered-known query按<code>leo_clear_weak</code>、<code>leo_low_elev_weak</code>、<code>leo_rain_weak</code>分别报告overall accuracy、min-class accuracy、min-receiver accuracy和min-day accuracy；reject／defer均按身份错误计；
- <code>target_leo_weak_open_set_blind_confirmation</code>：registered与unknown共同评分，报告threshold-free AUROC／AUPR-out／FPR95，以及source-only冻结决策下的<code>unknown_false_accept_rate</code>、<code>unknown_safe_rejection_rate=1-unknown_false_accept_rate</code>、registered false-reject／defer rate和registered accepted accuracy；另报告unknown TX、receiver和day切片的最差safe-rejection及覆盖数。该组是单节点确认性结果，不是Phase3协同结果。

每个<code>fold f×scene s</code>必须记录registered、unknown、class、receiver、day和unknown-TX的精确覆盖数及每项指标的分子／分母。<code>fold三scene等权</code>固定为三个scene标量的算术均值；<code>global cell等权</code>固定为18个<code>fold×scene</code>标量的算术均值；将全部target行合并计算的sample-pooled overall必须另列，不能称为“等权”或替代cell等权。clean target若存在只能标为隔离诊断，不能替代任何LEO weak结果。

五门按同一C／G row、同一fold、同一物理样本和不可补偿规则判断：

1. 技术稳定：12／12训练arm final checkpoint、terminal、completion、resource、heldout和CLIC receipt闭合；无collapse、nonfinite、shape、VJP、AMP、图释放或协议错误。
2. source known跨RX：source clean 6／6中，overall accuracy、min-class accuracy、min-RX accuracy和min-day accuracy的<code>G-C</code>均不低于<code>-2pp</code>。
3. source与target LEO weak域泛化：source三scene×6fold共18个LEO cell的上述四项<code>G-C</code>均不低于<code>-2pp</code>；每fold三scene等权overall和global 18-cell等权overall也均不低于<code>-2pp</code>。此外，12个checkpoint的target零适配输出必须技术闭合；每fold每个target scene的四项registered-known指标、每fold三scene等权overall和global 18-cell等权overall均要求<code>G-C≥-2pp</code>。source与target两组内部、两组之间均不可补偿。
4. source-proxy研发正信号与target blind-confirmation完整性：每fold fixed400 TX互斥proxy的AUROC增量与<code>mean(u_proxy)-mean(u_V)</code>增量都必须严格大于0，要求6／6通过；proxy只评分，零fit、零阈值、零训练／停止反馈。target registered／unknown LEO weak盲测的12份prediction、source-frozen决策规则、scorer和上述全部指标必须同row完整报告；本轮不新增target-unknown数值门，target结果只能限定确认性声明和下一阶段准入，不能驱动候选重排或任何修改／重跑。
5. bundle交付：6／6 G checkpoint导出的bundle均通过schema、state、source-only聚合、禁止成员、SHA和独立reload→forward一致性验证。

任一门失败即<code>REJECT_P1_CLIC_PERMANENT</code>。只有五门全部通过，状态才能变为<code>PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW</code>；仍不得直接声称真实unknown、Phase2、Phase3、在轨或注册成功。

## 12. 正式registered／unknown与K-shot交接

Phase1不使用target unknown训练、调参或确定拒识规则。冻结后单节点盲态确认中，每个registered或unknown物理样本先按同一truth／role-blind规则绑定一份且仅一份<code>leo_*_weak received_i</code>，CLIC只读取这份IQ并输出<code>z_id,z_dom,q_clic,e_unknown,decision</code>；全部预测封存后才连接truth sidecar。clean unknown只允许隔离标记<code>DIAGNOSTIC_UNKNOWN_NO_LEO_NON_FORMAL</code>。该确认不实现多节点协同、anonymous track、可信确权或注册授权。

Phase2旧类support适应继续使用现有合法同公式接口；新类只有在<code>registration_authorized=true</code>后，才能由K个新采集、互不重复、各自单LEO的物理support加行。每个query独立面对全部已注册类，query零update；历史unknown query不得追溯成为support。

## 13. 计划目标文件

本节只是实施映射，不表示文件已经存在或功能已经实现。

|责任|目标文件|
|---|---|
|CLIC算子、模块、配置、receipt、terminal、failure、VJP和资源合同|code/cvsrffi/phase1_clic.py|
|exact-head前单次CLIC融合和aux输出|code/model.py|
|dual wrapper构造、<code>z_id/z_dom/q_clic</code>暴露|code/model_dual_cvsincnet.py|
|checkpoint参数合并与bundle reload builder|code/post_stage_common.py|
|CLI、strict warm-start、训练接线、AMP、receipt和terminal|code/SSDG/train_ssdg.py|
|训练纯函数／模型／VJP／receipt／负例／图释放测试|code/tests/test_phase1_clic.py|
|12臂冻结launcher|code/scripts/launch_phase1_clic12_20260811.sh|
|clean／LEO／pair／bundle后冻结实现|code/export_phase1_clic_features.py、code/export_phase1_clic_leo_features.py、code/evaluate_phase1_clic_postfreeze_pair.py、code/export_phase1_clic_deployment_bundle.py|
|目标域单LEO零适配推理与隔离评分|code/evaluate_phase1_clic_target_leo.py|
|后冻结测试与60步launcher|code/tests/test_phase1_clic_postfreeze.py、code/scripts/launch_phase1_clic_postfreeze_20260811.sh|

## 14. 可追溯性矩阵

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|CLIC-01|用户条件／项目协议|同一<code>received_i</code>内多lag数学表征不增加LEO观测、物理样本、shot或K|本卡第1、12节|verified|逐条对照<code>项目.md</code>第4、5.1、5.2、7.1节|正式registered／unknown同规单LEO|
|CLIC-02|独立监督|本轮target registered／unknown LEO weak修订达到<code>P0=0/P1=0/ALLOW-DESIGN-REVISION</code>|本卡全篇|verified|独立监督对latest actual diff终裁为ALLOW-DESIGN-REVISION|仅设计许可|
|CLIC-03|数学合同|无epsilon的<code>u/h/r</code>、zero-mask、nonfinite fail-closed和幅相／CFO不变性|code/cvsrffi/phase1_clic.py|verified|本卡解析证明；ssr-gpu数值微验证最大误差：u=6.80e-7、h=5.96e-7、r=2.38e-7|实现仍pending|
|CLIC-04|算子实现|固定<code>L={1,2,4,8}</code>的C／G同shape token|code/cvsrffi/phase1_clic.py|pending|实现后纯函数与shape测试|仅<code>T_C↔T_G</code>|
|CLIC-05|模型结构|固定depthwise E、gate、<code>W_c</code>、单exact head和<code>q_clic</code>|code/cvsrffi/phase1_clic.py、code/model.py|pending|实现后模型forward／state SHA测试|新增参数32529|
|CLIC-06|dual接口|CLIC后的<code>z_id</code>与existing<code>z_dom</code>／quality暴露|code/model_dual_cvsincnet.py、code/post_stage_common.py|pending|实现后strict reload／aux测试|不增加第二readout|
|CLIC-07|C／G公平|同模块、同初始SHA、同forward／参数／资源，仅operator不同|code/SSDG/train_ssdg.py|pending|实现后pair receipt和tamper负测|C也主动训练token支路|
|CLIC-08|数据权限|source-L-only、U／V／proxy／target／unknown零训练反馈、single-LEO|本卡第1、6、12节|verified|协议逐项审计|不把source proxy写成unknown|
|CLIC-09|VJP合同|C／G三scene分别验证token、CLIC参数、base和head finite-nonzero VJP|code/cvsrffi/phase1_clic.py、code/SSDG/train_ssdg.py|pending|实现后正例与detach／zero篡改负测|raw-unscaled诊断|
|CLIC-10|AMP／图释放|一次正常backward／unscale／step，finite与skip均释放图根|code/cvsrffi/phase1_clic.py、code/SSDG/train_ssdg.py|pending|实现后saved-tensor与AMP skip测试|禁gc／empty_cache|
|CLIC-11|receipt终态|scalar／count／SHA-only config、failure、terminal和资源逐batch闭合|code/cvsrffi/phase1_clic.py、code/SSDG/train_ssdg.py|pending|实现后terminal／tamper负测|不存IQ／feature／ID|
|CLIC-12|训练机械验证|CLI、py_compile、focused测试、help、12臂dry-run和旧机制关闭|test／launcher目标文件|pending|本地ssr-gpu验证|不得以AST替代真实forward|
|CLIC-13|训练矩阵|6fold×C／G、40E、固定seed和immutable run root|training launcher／report|pending|本地dry-run、独立审查、唯一Runner|N607前另建报告|
|CLIC-14|后冻结五门|sealed60、F6 raw reopen和五项非补偿门|postfreeze目标文件|pending|本地合成测试后真实artifact重开|无性能补偿|
|CLIC-15|deployment bundle|6／6真实G checkpoint导出、reload和禁止成员闭合|bundle exporter／tests|pending|真实checkpoint bundle验证|第五门，不等于Phase3|
|CLIC-16|N607训练|唯一release、launch、监控和小工件回收|N607 runner／automation report|deferred|仅在本地实现、复审、commit后执行|当前未授权启动|
|CLIC-17|性能与晋级|读取sealed同row性能并作五门判定|主Agent／最终报告|deferred|需ARTIFACTS_COMPLETE后独立分析|当前NO_PERFORMANCE_RESULT|
|CLIC-18|旧路线去重|八个永久拒绝机制及其它旧loss不得复活或拼接|全实现面|rejected|设计静态排除；实现复审再搜旧identity|非CLIC组成|
|CLIC-19|SQSF替代案|Sinc频率轴假设与warm-start映射未闭合，本轮不采用|无目标文件|rejected|独立二选一审查|不是性能永久拒绝|
|CLIC-20|用户目标域补充|每个C／G checkpoint均报告同一封存目标capsule的registered-known DG与registered／unknown单LEO weak盲态开放集指标，并与source-proxy研发信号同row封存|code/evaluate_phase1_clic_target_leo.py及postfreeze矩阵|pending|IQ-only package、目标预测SHA、source-frozen规则、隔离scorer、逐scene分母／指标、C／G same-row及zero-feedback负测|单节点确认，不等于Phase3协同|

当前追踪计数：verified=4，pending=12，deferred=2，rejected=2，blocked=0。

最高风险项是CLIC-14／CLIC-20：解析不变性只覆盖全局复增益、常相位和线性CFO，不能保证时变多径下的identity／channel分离；必须同时由source 18个LEO最差切片、封存目标域LEO weak零适配切片和6fold source-proxy双门证伪，不能由本地纯函数测试解除。

## 15. 规范自审

- 完整性检查：本卡中的方法常量与合同字段均已明确定义。
- 一致性检查：C／G参数、shape、forward和训练合同一致，唯一科学变量为<code>T_C↔T_G</code>。
- 范围检查：本卡覆盖Phase1 CLIC表征、checkpoint、bundle及冻结后单节点target blind-confirmation；不实现Phase3多节点协同、anonymous entity、可信确权或注册授权。
- 歧义检查：单份<code>received_i</code>、定义域外zero-mask、nonfinite fail-closed、source-proxy与target blind-confirmation边界、目标域zero-adaptation truth-side scorer及五门阈值均已显式固定。
- 设计对齐：当前是严格设计规范，不是近似实现；所有代码、测试、N607和性能项仍按追踪表保持pending或deferred。
