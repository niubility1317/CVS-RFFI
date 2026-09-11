# CORE90锚定角度几何与净收益融合：设计实现计划

> 后续复查更新：当前存在OOF交付、动作数值一致性及source汇总缺口，尚不满足严格全量验收。下面的完成标记保留上次检查历史，最新状态以[再次审查报告](core90_anchored_geometry_recheck_20260911.md)为准；设计要求本身不因缺口而放宽。

日期：2026-09-11。计划编制状态为PLAN_READY；后续按用户授权完成T1–T9适用实现，当前为**IMPLEMENTATION_VERIFIED**。见[实现与确认验收](core90_anchored_geometry_acceptance_20260911.md)。未执行完整source矩阵或独立确认实验；下面保留设计定义和原实施顺序。

目标：在冻结H0身份骨干和原始部署判别定义的前提下，实现有界低秩角度头G、物理样本配对的source缓存训练、H0正确样本间隔保留和经过交叉拟合检验的概率融合；部分证据E作为独立扩展。

依据：[用户设计报告全文](core90_anchored_geometry_design_source_20260911.md)、[上一轮完整实验报告](../automation_reports/CV-SincNet/core90_evidence_frozen_392005_20260911/comprehensive_report/report.md)、[要求追踪表](core90_anchored_geometry_traceability.md)。代码基线：`7fd06087d2215e19707b4106af0c4fad2261979a`，分支`codex/core90-evidence-head-20260911`。

技术栈：现有PyTorch/NumPy、项目`ssr-gpu`环境、Windows原生本地开发、既有N607发布流程。此计划不引入新服务、训练框架或强制安装依赖。

执行方式：按下面任务顺序在当前隔离Git承载面实施。每个任务完成直接相关检查；不默认派生Agent、不额外增加固定审批或重复审查。计划编制时仅获计划授权，后续用户明确授权严格实施并验收；实际完成证据记于验收报告。

## 1.设计判断与实施范围

采用报告推荐主线，不把新方法命名成H3/H4增强版。四项职责分开：H0保存原判别能力，G调整有限角度关系，E描述部分观测，融合器估计实际修正动作的净收益。

|路线|本计划安排|原因|
|---|---|---|
|共享、有界低秩角度G|第一优先实现|可以从H0等价点启动，隔离度量贡献|
|固定概率混合|先于动态门控完成|先证明两头存在可利用的互补性|
|小型收益门控|代码纳入主线；有source净收益后才拟合/启用|无互补收益时增加门控不能替代改进专家|
|部分证据E|单独任务和矩阵P1|完整角度判别与缺失边缘化不是同一个观测模型|
|类别方向有限修正|条件性后续，不在首轮开放|避免与度量变化互相补偿|
|共享/类特有条件响应、中心support后验|分别制定接口与准入条件，首轮不执行|不把状态混杂与注册任务重新塞进Phase1总损失|
|H5、类特有协方差、联合骨干训练|首轮不执行|不需要这些模块才能回答本轮核心问题|

上一轮的5203 rescue、16013 harm只用于解释设计动机；它们不能成为下一轮目标样本选择、类别专属规则或超参数拟合数据。此前H1–H4名称、配置和结果保持原义。

### 1.1不可混淆的三层证据

1. **实现正确性**：H0等价、矩阵约束、梯度、缓存角色、融合保护、校准顺序、导出重载等可以用本地技术测试证明。
2. **source开发证据**：冻结骨干下的头部留RX预测、融合净收益、拟合稳定性和成本；不能冒充整模型未见RX泛化。
3. **确认性泛化证据**：必须在更独立、未用于设计反馈的合法确认数据上，由冻结候选预测后独立评分。本次计划不把现有target重新包装成新确认集。

当前target已用于上一轮失败分析。即使G只在source上拟合，新设计仍受已观察target结果启发；再次测同一target最多属于明确披露的开发研究，不能用于干净确认性结论。现行协议不允许把确认集评分反馈到选择或重跑，因此本计划默认入口停在`SOURCE_SYSTEM_FROZEN`，不安排现有target自动重测。独立确认数据的实际物理集合尚未提供，不虚构新数据集，也不因改seed就视为独立。此项不阻止代码实现和source实验准备。

## 2.已核实代码与复用边界

以下行号对应当前基线，实施时以函数名定位。

|现有入口|观察|实施决定|
|---|---|---|
|`code/model.py:711`，`CosFaceHead.forward()`|特征/weight以`eps=1e-4`归一化；有labels时减margin，无labels时返回纯角度logits|缓存必须走`labels=None`；读取实际`s`，不靠默认30猜测|
|`code/model.py:863`，`PhysicalAwareClassifier.forward_logits()`|最终角度头读取`feat_joint`|只在真实前向等价后读取原类别方向；不略过joint特征构造|
|`code/model_dual_cvsincnet.py`，`backbone_forward_compat()`及双分支forward|无aux可走身份快速前向；双分支aux路径仍调用域分支|新增只提取身份aux的适配器，比较完整部署、旧fast路径和适配器|
|`code/cvsrffi/evidence_pipeline.py:55`，`fit_frozen_head()`|每epoch重新过IQ；传`y_tx`；只做现有source输入的冻结头训练|G新增独立缓存训练器；保留历史H1–H4入口，不静默修改旧实验含义|
|`code/cvsrffi/evidence_pipeline.py:17`，`verify_checkpoint_contract()`|核对当前H0训练契约及scratch/source-only lineage|复用当前H0的已有核对；不能只拿不含来源的deployment.pt宣称lineage完整|
|`code/cvsrffi/evidence_source_evaluation.py:14`，`_collect()`|无标签margin的source收集，仍调用双分支aux|复用source角色约束，收集主体接入新身份适配器|
|`code/cvsrffi/evidence_target_evaluation.py`|已有opaque ID、逐样本信道、固定预测后独立评分|复用科学边界；新协调器不能直接继承旧“source结束即自动target”行为|
|`code/cvsrffi/partial_gaussian_head.py:62`|已经有共享低秩Woodbury求解及正确观测边缘化|E复用，不另造不一致的概率内核|
|`code/cvsrffi/evidence_head.py:151`|当前额外density NLL重新调用稠密高斯|E的新路径复用一次低秩结果，不直接调用旧混合总损失|
|`code/cvsrffi/evidence_decision.py:25`|现有校准按观测维数和quality分箱，要求Mahalanobis|G使用独立最终概率温度校准；E使用版本化pattern校准器，保留旧v1读写|
|`code/cvsrffi/evidence_observation.py`、`pairwise_evidence.py`|已有固定块尺度、partial mask、Schur类对诊断|E继续复用；证据块语义必须从实际提取schema确认|

这一轮主要增加独立模块，不重构庞大的H0训练器，也不覆盖旧模型默认行为。必要的公共内核改动必须保留旧H1–H4数值回归。

## 3.数据、缓存和部署合同

### 3.1数据角色

固定当前数据契约：source RX=`[1,3,4,6,8]`，day=`[1,2,3]`，`L_s=6300/U_s=56700/V=27000`，split seed=`392005`。当前H0来自本次scratch E200，正式加载前引用已有来源证据并核对实物；若来源变化，只重新检查受影响的checkpoint兼容性。

|状态|允许数据|不可做的事|
|---|---|---|
|H0/G缓存提取|L_s的IQ及预定义source增强|混入V、U_s或target训练记录|
|G和普通角度头拟合|L_s缓存；内部折只取所属RX|从V选择epoch/拟合分类参数|
|质量/观测误差拟合|本折训练L_s的受控退化配对|把target错误反传到quality|
|融合器训练与开发评价|L_s内部诚实交叉拟合预测|用专家自身训练集准确性标签代替OOF；向专家回传梯度|
|最终校准|冻结系统在V及对应预定义视图上的输出|改内部专家温度、重训门控或把V结果称独立风险认证|
|独立确认预测|将来明确的合法received IQ与冻结bundle|任何参数、统计、阈值、候选或缓存更新|

本轮G不使用U_s；H0历史source U_s训练预算保留在成本和lineage中。头部RX交叉拟合不重新定义外部数据角色，也不要求重验未改变的data capsule。

### 3.2缓存结构

新增`code/cvsrffi/anchored_cache.py`，每份source cache由一个manifest和分块tensor文件构成，不创建签名/receipt链。

```python
@dataclass(frozen=True)
class CacheIdentity:
    checkpoint_sha256: str
    data_contract_sha256: str
    preprocessing_version: str
    view_recipe: dict
    split_seed: int
    role: str                 # L_s或V

@dataclass
class FeatureRows:
    h: Tensor                 # FP32[N,160]，原feat_joint，未再归一化
    baseline_inference_logits: Tensor  # FP32[N,C]，y=None真实前向
    quality: Tensor           # FP32[N,Q]，固定received-IQ描述
    observed: Tensor          # bool[N,D]，完整G路径必须全True
    labels: Tensor            # int64[N]，只在训练/校准数据层存在
    physical_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    receiver: Tensor          # 仅分折/报告，不进入头或门控
    day: Tensor               # 仅分层采样/报告
```

`training_margin_logits`不进入融合缓存。如果实现普通CosFace监督对照时需要margin，临时从监督分支计算，字段名称与baseline严格区分。source缓存键包含原checkpoint、契约、预处理、增强参数和种子、physical ID、view_id；这仅防止错误复用，不能代替lineage检查，也不触发额外数据复验。

缓存以`(physical_id, view_id)`唯一；不同view共享同一physical ID。checkpoint/坐标/预处理/增强变化时重提取相应缓存，不静默命中旧缓存。一次提取后所有epoch和头部seed重用同一份增强样本。`eval()`与`no_grad()`同时使用；写出普通CPU tensor。若采用`inference_mode()`，必须退出上下文后clone成普通tensor并验证可参与后续weight梯度。

### 3.3source视图和物理权重

首版每个L_s物理样本保存clean和一个固定LEO视图，共12600行。在每个TX×RX×day组的70条内，按固定view seed打乱后以24/23/23分配三种LEO场景；多出的场景在90组之间轮换，使全L_s每场景恰为2100条。场景标签只用于生成与训练诊断，不进入G或门控。V每组300条可均分三种场景，clean/LEO各27000行。所有候选与head seed共用缓存；增加增强seed是独立实验因子，不与head seed混用。

每个物理样本总权重1；配对训练为`0.5 CE_clean+0.5 CE_LEO`。以后若增加view，总LEO权重仍为0.5，在该物理样本的LEO views内均分。不能按缓存行简单平均造成多view样本加权。

全source每batch从90组各抽2个physical ID：180个physical、360个feature行；35个batch覆盖一轮，组内无放回。4RX折为144个physical/288个feature，3RX折为108/216，同样35步/轮。缺行或组数不符合合同时显式报错，不靠补采样悄悄变更预算。

### 3.4训练与部署状态分离

训练cache可以有source样本级embedding和RX/day/label；推理bundle不得携带它们、数据路径、索引或可恢复样本的结构。部署保留冻结H0身份模型、G参数、原类别方向/映射、门控参数和明确的最终校准参数。新bundle声明`stage=Phase1`，不自动获得Stage2权限。向Phase2转用时，还要单独核对其状态是否符合当前穷尽式白名单，尤其不能把source cache或普通经验统计打包带入。

## 4.G的精确定义和默认起点

新增`code/cvsrffi/anchored_geometry.py`。所有类共享

```text
Q = qr(B).Q；B初始化为非退化随机正交基
a = rho × tanh(b)；b初始化为0
M = I + Q diag(expm1(a)) Qᵀ
score_G(y) = tau0 × (hᵀMw_y) / (sqrt(hᵀMh) sqrt(w_yᵀMw_y))
```

`tau0`和类别方向从通过等价检查的H0真实头读取。固定`r=4`、`rho=log(2)/2`，故`kappa(M)≤2`；全source首阶段仅训练B和b（160×4+4=644个参数），W、tau、骨干全部冻结。不存在每类bias、自由协方差、状态响应、support loss或density NLL。

### 4.1低秩计算与退化点

```python
def metric_logits(h, w, q, a, tau, eps=1e-4):
    c = torch.expm1(a)
    hq, wq = h @ q, w @ q
    numerator = h @ w.T + (hq * c) @ wq.T
    hn = (h.square().sum(-1) + (hq.square() * c).sum(-1)).clamp_min(eps**2).sqrt()
    wn = (w.square().sum(-1) + (wq.square() * c).sum(-1)).clamp_min(eps**2).sqrt()
    return tau * numerator / (hn[:, None] * wn[None, :])
```

这段为实施公式，不代表已入生产。负`expm1(a)`合法，因为M始终正定；不能用只允许正增量的`I+AAᵀ`近似取代。若W/h处于极小范数区，按H0的`eps=1e-4`处理并记录；正比例不变性仅对未触及eps下限的有效输入严格成立。零向量、非有限输入、秩亏B都必须有显式数值处理，不无限增大jitter。

初始化a=0时M=I；a路径在非对称数据上必须有非零梯度，Q首步梯度为零是正常现象。Q采用QR重参数化，每次forward满足正交；保存B并在导出时保存规范Q/a，验证重载。推理预计算`Qᵀw`及类分母，不生成160×160矩阵；只在低频诊断/稠密参考测试中构造M。

### 4.2严格H0锚点

新增`IdentityAnchor.extract(x)`返回`h/s0/valid`，内部明确`y=None`、关闭source统计更新。通过以下三方对照后才宣称等价：原H0完整部署、原身份fast路径、仅身份aux提取后原CosFace。

FP32起始容差采用`atol=1e-5, rtol=1e-5`并保存最大误差及近并列样本；argmax要求非并列有效样本一致。技术不等价时先修正读取路径，不把不同读出仍命名H0。`alpha=0`部署分支直接返回锚点原始决定；不能靠重算另一个近似头实现“回退”。如未来联合更新骨干，必须取消严格回退声明或保留独立原H0。

## 5.拟合目标、预算和收敛

主线损失严格分三项：

```text
L_G = L_cls + lambda_keep L_keep + lambda_metric Σ a_j²
gamma0 = s0[y] - max_{k≠y} s0[k]
gammaG = sG[y] - max_{k≠y} sG[k]
L_keep = mean_physical(mean_view(valid × 1[gamma0>0]
                    × relu(min(gamma0,gamma_max)-gammaG)²))
```

mask按每个实际view上H0是否正确产生，必须使用`baseline_inference_logits`；不使用训练margin判断correct。分母是全部有效physical权重，不只对“被保留样本”归一化；无H0正确样本时返回可反传的0。H0错误样本由真实label的CE驱动，不统一施加教师KL。

以下是**本计划建议的可执行起点，尚无最佳性证据**，实施配置显式记录而不是借用旧头默认值：

|参数|首版建议|归因|
|---|---|---|
|rank、rho|4、log(2)/2|报告指定保守起点|
|optimizer|Adam，lr=0.001，weight_decay=0|不对正交基重复施加隐式L2；偏离由显式a正则控制|
|lambda_metric|0.01|固定source开发起点|
|gamma_max|0.1×tau0|以H0尺度表示截断间隔|
|lambda_keep|1/tau0²|使平方margin约束不随CosFace分数尺度任意放大|
|拟合上限|80轮＝2800步/模型|不是默认20轮即充分；source检查点20、40、80|
|head seeds|392005、392006、392007|仅改变头初始化/训练顺序；split/view seed不变|
|lambda_H|2|门控utility对harm惩罚为rescue的两倍；仍另报未加权净正确数|
|候选alpha|0、0.25、0.5、0.75、1|报告固定动作集|

默认不做大范围超参搜索。若source诊断显示需修订，保存失败配置和原因，形成新配置版本；所有选择必须在独立确认预测前结束。

### 5.1停止与公平预算

训练记录每epoch的clean/LEO CE、keep、metric、各项加权梯度范数和夹角、`a/Q/M`变化、物理访问数、view数、步数。梯度诊断在每epoch首个固定batch执行，不为统计重复优化；共用autograd图后仍只step一次。

建议稳定窗口为最近5轮：source内部留出CE的max−min≤0.002、相邻M相对Frobenius变化≤0.001、keep无持续上升。只看a变化不够，因为Q也会变化。留出风险来自内部RX折，不看V/target；`converged`和`budget_exhausted`分别记录。门槛是source拟合诊断，不是新增远端启动审批。

首轮默认各专家完整运行80轮，20/40/80保存诊断截面；先从内部RX诊断确认预算是否充分，最终全source拟合仍用预登记的80轮。记录首次满足稳定条件的轮次，结束时再记录`converged`或`budget_exhausted`，不因看到某个target分数就扩轮。这样本轮不引入额外的自适应停止选择偏差，A2/A3/A4和匹配普通角度对照具有相同优化步数。

若后续确需缩短最终预算，只能在source内部从20/40/80选择，并使用嵌套RX折：评估外层RX时，预算选择只使用其余4RX的内部折；最终预算取各外层所选值的最大值。增加的内部拟合次数必须纳入资源清单，不能沿用首轮120次上限假装成本不变。该自适应预算分支首轮关闭。

## 6.融合：实际动作utility、诚实评估和保护区

新增`code/cvsrffi/anchored_fusion.py`及`anchored_crossfit.py`。

### 6.1先固定混合

内部专家尺度锁定，`p0=softmax(s0)`、`pG=softmax(sG)`；`p_alpha=(1-alpha)p0+alpha pG`。不是独立似然乘积，不声称贝叶斯去重。每个动作直接计算混合后的argmax、rescue、harm、净正确变化和改变比例，必须覆盖“三类示例中两个头分别选1/3、混合选2”的测试。

utility使用**实际部署动作**：`u_j=rescue_j−2×harm_j`。0动作恒为0。固定融合先在source OOF比较全部5个动作；不能凭新专家独立准确率决定组合收益。

### 6.2五折OOF与门控独立评价

基础OOF：每次用4个source RX拟合G，预测剩余RX；physical ID的所有view留在同一折。最终部署G另用全部5RX拟合。H0骨干见过全部source RX，所有报告必须标为“冻结骨干下的头部留RX”。

**补齐报告隐含但未展开的门控评价问题**：把所有OOF行训练门控后，再在同一OOF行报告其收益仍是训练集评价。首版采用嵌套RX交叉拟合：

```text
外层留RX=r：
  G_outer用其余4RX拟合，预测r
  在这4RX内部逐个留s，用其余3RX拟合G_inner，预测s
  用这4份inner OOF拟合本外层门控/输入标准化/阈值
  在r上评估(G_outer + 本外层门控)
最终：
  用5份外层G的OOF训练最终门控；用全部5RX重拟合最终G
```

任何质量拟合、feature标准化或可选source-support描述在对应fold也只使用训练RX。H0骨干作为固定公共条件例外明确披露。不同排除顺序但训练集合相同的3RX模型可缓存复用；每专家配置/seed最多10个3RX模型＋5个4RX模型＋1个5RX最终模型，不按20个inner角色重复训练。所有cost如实计入。

固定alpha、保护区域阈值及是否启用A6同样用上述inner OOF选择、outer汇总评价。A6在A4的基础五折OOF中出现非零固定动作的正净收益后才进入嵌套评价；这个触发只表示值得评价，不作为门控已获益的证据。outer结果仍是source开发证据；不宣称只有5个RX就获得高置信度跨域保证。固定checkpoint的3个head seed不能替代3个完整骨干seed。

### 6.3小型、类别置换不变的门控

首版为带L2的线性多动作utility回归，每个非零alpha输出一个预测期望收益；不用固定六类切换表。输入由共同公式形成：两头top1−top2概率间隔、两头归一化熵、预测是否分歧、JS差异、对称top-k差异、received-IQ质量、观测覆盖。训练时的RX/TX/day、场景名、physical ID及label一律不进入phi。

可选source支持程度只有在定义为按同一公式计算的距离/覆盖并完成source验证后加入；首版不为它保留经验特征库。标准化仅由本fold训练OOF拟合，运行时冻结，导出时声明其阶段权限。质量低不强制选择G。

默认`utility_margin=0`：仅当最大非零动作预测收益>0才采用，否则回到0；utility并列选更小alpha，绝不按TX编号处理。门控训练detach专家输出。输入类列置换后phi与选择动作不变；并列预测以固定tie policy处理，不伪造严格唯一argmax的置换保证。

### 6.4保护区和动作上界

保护区只由合法推理量定义。建议从inner OOF中H0唯一top1概率间隔的90%分位数确定高间隔区，不用“运行时H0是否正确”这种需要truth的条件。每fold阈值只拟合训练OOF；最终阈值由最终OOF固定。

对保护区中唯一H0预测k，计算每个j≠k的`m_j=p0[k]−p0[j]`和`d_j=pG[k]−pG[j]`。当`d_j<0`时边界`b_j=m_j/(m_j−d_j)`；无负d则上界1。候选动作必须严格位于所有翻转边界内，浮点上用朝0的`nextafter`留出安全侧，最后验证实际混合top1仍为k。不满足则退到更小离散alpha；数值非有限或H0并列时回原H0决定并记录原因。

不能保护全部样本，否则rescue必为0。保护区外允许修改，但必须评价实际净收益。保护保证的是H0决定不变，不是它正确；不能写“性能不会下降”。

若保护策略使原alpha被降档，utility label、训练phi关联的动作和推理action必须使用**保护后的实际动作**。禁止用未裁剪混合收益训练、部署再裁剪的错配。建议函数`realize_actions()`先生成所有实际动作分布，训练和部署共同调用。

### 6.5最终校准顺序

```text
缓存/内部尺度固定 → 专家拟合 → OOF融合器拟合 → 全source G与整个决策系统冻结
→ V最终输出温度 → V部署置信阈值 → 导出
```

新增`FinalProbabilityCalibrator`，输入冻结系统的log probabilities，拟合单一正温度`T_F∈[0.05,20]`；输出`softmax(log p_mix/T_F)`。H0、cosine、G、固定融合、动态融合使用相同V权限和物理加权方式分别校准，不在门控后修改T0/TG。采用稳定log-softmax/logaddexp避免概率下溢；精确alpha0/1分别分支，校准前后唯一argmax一致。校准不更新G或门控。

闭集主表全部样本均纳入。另以相同V目标覆盖90%的规则固定部署置信阈值，ties整组处理并记录实际V覆盖；不承诺target仍为90%。同时报告部署门控的risk/coverage与纯置信排序曲线，两者分开；不按truth打破ties。

## 7.独立扩展P1：部分证据E

新增`code/cvsrffi/partial_evidence_fit.py`和`mask_pattern_calibration.py`，复用原`evidence_observation.py`、`partial_gaussian_head.py`、`pairwise_evidence.py`。

### 7.1观测模型与分阶段拟合

`z=S psi(x)`使用固定块宽缩放，不按剩余观测范数归一化。首版类均值固定、协方差共享：`C(q)=D+UUᵀ+R(q)`，无条件均值、无状态协方差、无support episode。

拟合分开：L_s clean求类中心和共享类内残差；受控退化配对单独拟合观测误差R；最后冻结统计做E推理。首版D/U采用共享残差收缩协方差及rank=4分解，正对角floor沿用现有配置并记录坐标单位；收缩强度建议0.1，作为明确source开发起点。R保留已有受控退化代理接口，报告偏置和误差方向，不能声称clean残差与退化误差严格独立或实测物理协方差。不给E叠加一个未经验证的CE+NLL联合总loss。

一次低秩求解返回`scores/mahalanobis/logdet/observed_count`，分类、density诊断和校准共用。共享logdet是同一样本所有类的公共项，只用于密度/一致性，不能作为额外类别区分能力。报告NLL/观测维数用于诊断，但分类仍使用原归一化高斯分数，不偷换成每维平均。

### 7.2真正可解释的缺失模式

P1优先使用现有`blocks`提取接口的`t_emb/f_emb/pa_local`命名块，在缓存记录实际`block_sizes`；不把160维joint随便切三段命名成物理分支。首版模式：`full`、`missing_t`、`missing_f`、`missing_pa`、`all_missing`。模式作用于**提取完成后的块输出不可用**，这是结构化特征块缺失实验，不声称等价于原始IQ时间段丢失。

原始IQ局部时间缺失/受污染mask需单独通过提取依赖关系定义：若污染会影响整个joint，就不能事后mask几个latent维度声称局部概率边缘化成立。此类模式先做观测接口设计与source扰动验证，通过前不进入P1正式模式；不会因为报告列举它们就自动增加无法解释的掩码。

校准按`(pattern_id, quality_bin, support_level)`索引，保留物理ID配对。同样维数不同pattern不合并。首版support_level是由模式schema和有效坐标数确定的`nonempty/empty`，不是RX/TX标签；任何以后更细层级都需source定义。V生成与训练/评价相同机制的模式视图，计数同时报告physical count和view count。旧v1维数校准器不改变；新pattern状态使用独立version。

未覆盖pattern或不足20个物理样本的组默认defer；首版不启用粗粒度fallback。全缺失输出无信息分数并defer，不能产生“unknown TX”标签。registered样本的defer计入有效准确率错误。

P1不把H0/G在被污染joint上产生的输出当可靠保底，也不默认并行运行H0/G/E三专家。P1单独比较E全观测、E各结构化缺失、简单缺失读出对照；完整观测主线不调用E。

### 7.3新增身份信息诊断

复用Schur补计算`C_B|A`和`delta_B|A`，输出全部类对的`J_B|A=deltaᵀ C⁻¹ delta`，与同一source RX上的实际风险变化配对。可靠性R、相关性C和区分度J各自报告。不只盯历史失败的某一个TX，也不将已有H0(A,B)乘以`p(B|A,y)`称作严格无重复贝叶斯融合。

## 8.条件性后续设计边界

|后续模块|进入条件|实现形式和必须验证的内容|
|---|---|---|
|G有限类别方向修正|共享M先取得稳定source留RX收益|`normalize(w+delta_w)`，所有类同一角度上限；单独消融，不和M首轮同时开放|
|E共享响应|常均值E稳定、state/quality关联审计完成|先`mu_y+B_shared(e−e0)`，再有限`delta_B_y`；均值-only、协方差-only、组合分开|
|响应误差传播|source配对退化能描述残差|先分析`delta_z−[mu(e+delta_e)−mu(e)]`的偏置/方向/协方差；不能默认z/e误差独立并重复叠加|
|中心support后验|明确source注册任务胜过均值/收缩原型|固定斜率；旧类使用合规导出先验，新类用共享结构先验，不能借某个旧TX中心|
|support响应斜率|状态设计矩阵满秩、条件数可接受且query处于支持域|只由support确定状态；K足够不等于状态覆盖足够；重复physical ID/view不增加shot|
|联合训练|另立实验线|保留独立H0并计额外推理成本，或放弃严格回退；新特征坐标要求重提缓存、误差模型、中心先验和V校准|

source注册中“留出TX”仍是骨干见过的source类，不能称真正新TX泛化。真正Stage2-C同时要求合法old/new support与全部注册类逐样本竞争；本轮不直接扩展到target support或Phase3。

## 9.实验矩阵与科学判定

### 9.1按报告保留A0–A6与必要对照

|ID|专家/训练|保留项|输出与问题|
|---|---|---|---|
|A0|真实H0冻结|匹配最终校准|闭集锚点、统一校准基线|
|A1|source clean均值cosine|匹配最终校准|简单角度读出；固定尺度，不在V训练头|
|A2|G、冻结W、clean-only|metric正则|有限几何修正本身|
|A3|G、冻结W、配对clean/LEO|metric正则|头部拟合分布贡献|
|A4|A3|增加正确样本keep|是否减少harm并保留rescue|
|A5-0/25/50/75/100|H0与A4专家固定概率混合|实际action/保护策略全记录|固定组合是否有增量；0/100复用端点|
|A6|A4＋小型utility门控|嵌套RX交叉拟合、保护区|动态是否优于相同权限下选出的固定混合|
|C_angle|普通归一化线性角度头，同clean/LEO缓存|W初始化H0、M=I，训练W，不加bias|任何头见到匹配数据是否都会改善|
|C_angle_keep|C_angle|同A4的keep|将保留项贡献与低秩度量贡献进一步分开|
|P1|简化E、命名块缺失、pattern校准|单独矩阵|剩余证据在缺失时是否产生有效识别|

`C_angle`不是A1：前者是监督拟合的普通角度头，后者是均值cosine。所有普通角度对照使用部署logits CE，首轮不另加训练margin，以免又引入损失差异；若研究margin，另命名而不覆盖C_angle。

A2/A3/A4不根据旧target挑选方向或TX。A5以A4为预定专家，不在target中从多个G挑最好。各非零alpha在source上没有可信收益时，A6记`NOT_ACTIVATED_NO_SOURCE_UTILITY`，这是报告要求的研究分支停止，不是程序错误、GPU技术故障或新审批。

### 9.2建议source晋级口径

先报告闭集净正确数和`rescue−harm`；utility的lambda_H不替代原始准确率。A6的必要条件建议为：嵌套outer汇总净收益>0，至少4/5RX净收益非负，clean平均下降不超过0.2个百分点，最弱RX/最弱TX下降不超过0.5个百分点，并在三个head seed上方向一致。阈值是本计划的保守source开发起点，不是用户报告已给出的数值，也不是统计显著性保证。

这些条件只决定是否有理由研究更复杂分支。有限5RX下不能将逐IQ独立bootstrap当跨域置信证据。必须列出所有RX/day/TX及seed原始差值；如使用区间，应按实际采集单元/physical ID成组，报告其适用边界。最终科学晋级还要独立确认、多骨干seed与成本，不以source筛选通过代替。

### 9.3资源预算核算

A2/A3/A4/C_angle/C_angle_keep是5种需训练的专家配置。基础5折OOF＋最终拟合每种每seed6个模型；三head seed为90个头拟合。只有A4若进入A6，额外10个3RX模型/seed用于嵌套门控评估，共30个；因此首轮最多120个轻量缓存头拟合，加门控和E独立成本。若预算选择需要在其他专家上进行嵌套诊断，必须在配置清单中增加相应3RX模型，不能把它隐藏在90/120计数内；固定80轮的配对消融可避免这部分额外选择成本。

以上是模型次数预算，不是墙钟承诺。每模型最多2800步，缓存避免反复IQ/骨干计算。先测一次缓存提取、一个head step和一次独占推理profile，再估时。保留按epoch和同step截面的对照，不能因为新batch只35步就和旧99步直接比较epoch速度。

## 10.文件责任与接口

|新建/修改|责任与接口|
|---|---|
|新增`code/cvsrffi/anchored_cache.py`|`IdentityAnchor.extract(x)`；`build_source_cache(anchor, loader, identity, view_recipe, output)`；`load_source_cache(path, expected_identity)`|
|新增`code/cvsrffi/anchored_geometry.py`|`AnchoredMetricHead(w0, tau0, rank, rho)`；`forward(h)`；`geometry_diagnostics()`；普通角度对照|
|新增`code/cvsrffi/anchored_fit.py`|`fit_expert(cache, train_rx, config, seed, output)`；physical配对batch、loss与history|
|新增`code/cvsrffi/anchored_crossfit.py`|`build_rx_folds(cache)`；`run_expert_oof(...)`；`run_nested_fusion_audit(...)`；每份OOF附fit/heldout RX和physical集合|
|新增`code/cvsrffi/anchored_fusion.py`|`mix_log_probs()`；`realize_actions()`；`action_utilities()`；`utility_features()`；`UtilityGate`|
|新增`code/cvsrffi/anchored_calibration.py`|`FinalProbabilityCalibrator.fit(log_probs, labels, weights, role='V')`；`predict()`；冻结版本化状态|
|新增`code/cvsrffi/anchored_pipeline.py`|冻结系统、导出/加载、逐样本预测与artifact字段；不持有训练cache|
|新增`code/cvsrffi/partial_evidence_fit.py`|E分阶段拟合与复用低秩solver，不调用旧混合loss|
|新增`code/cvsrffi/mask_pattern_calibration.py`|PatternSpec和PatternCalibrator；pattern/version覆盖检查|
|新增`code/scripts/run_core90_anchored_geometry.py`|独立分阶段CLI，默认source-only，无旧协调器隐式target连跑|
|新增`code/configs/core90_anchored_geometry_v1.json`|上述配置、seed/view/fold定义、候选矩阵和允许阶段，字段必须被消费|
|新增`code/tests/test_anchored_*.py`及`test_partial_evidence_patterns.py`|下面各任务直接行为测试|
|按需修改`partial_gaussian_head.py`|仅共享求解结果/批处理接口；旧API语义与数值必须保持|
|不改H0默认`model.py`/`train_ssdg.py`|当前设计通过外接身份适配器和头实现；除发现明确等价性问题外不动历史训练路径|

G拟合返回`ExpertArtifact`：参数、配置、train_rx、head_seed、实际预算、history路径和来源摘要。OOF返回`OOFArtifact`：逐行s0/sG、physical/view键及fold；labels独立提供给source训练器。`FrozenAnchoredSystem`持有anchor/G/gate/final_calibrator；其`predict(x)`不接受labels、RX、day或场景标签。

## 11.逐任务实施与验收

下面保留计划中的可执行测试目标。对应生产模块与测试现已实现；勾选表示适用实现和技术验收通过，不代表正式source矩阵完成。实施使用已验证`ssr-gpu`解释器；每个任务先补能揭示目标错误的测试，完成后只跑相关文件及受影响旧接口回归。

### T1：真实H0身份前向与角色化缓存（R01–R06）

- [x] 新建`anchored_cache.py`与`test_anchored_cache.py`，定义第3节数据结构、manifest不匹配错误和只接受L_s/V的入口。
- [x] 比较原H0完整部署/fast/identity aux路径，覆盖clean及三种source弱LEO实际IQ；验证无label margin、无参数/BN统计更新、domain分支调用数0。
- [x] 实现一次提取和view配对；缓存重载后训练一个weight检查梯度可用；拒绝同键重复、角色交叉和预处理/权重不匹配。
- [x] 测试必须有：

```python
def test_anchor_uses_deployment_logits(anchor, source_iq):
    h, s0, valid = anchor.extract(source_iq)
    ref = anchor.original_model(source_iq, y_tx=None)
    torch.testing.assert_close(s0, ref, atol=1e-5, rtol=1e-5)
    assert s0.shape[1] == anchor.class_count
    assert valid.all()
```

运行：`python -m pytest code/tests/test_anchored_cache.py -q`。通过产物：可重用source缓存、H0等价误差和调用计数；此时不训练G。

### T2：有界低秩G（R07–R11）

- [x] 新建`anchored_geometry.py`与`test_anchored_geometry.py`；实现QR/tanh参数化和低秩公式。
- [x] 测a=0等价、随机有效输入正缩放不变、cond≤2、W/tau冻结、b首步可学习、导出重载和类列置换。
- [x] 稠密M参考只在测试里构造；比较forward及B/b梯度，检查极小范数、秩亏输入和非有限数错误。

```python
def test_identity_metric_and_trainable_scale(head, h, w0, tau0):
    ref = tau0 * F.normalize(h, dim=-1, eps=1e-4) @ F.normalize(w0, dim=-1, eps=1e-4).T
    torch.testing.assert_close(head(h), ref, atol=1e-5, rtol=1e-5)
    F.cross_entropy(head(h), torch.arange(len(h)) % len(w0)).backward()
    assert head.b.grad is not None and torch.isfinite(head.b.grad).all()
```

运行：`python -m pytest code/tests/test_anchored_geometry.py -q`。通过产物：G可在缓存训练，推理不分配D×D协方差；不宣称性能收益。

### T3：配对训练与预算遥测（R12–R16）

- [x] 新建`anchored_fit.py`、`test_anchored_fit.py`，实现35批物理分层遍历、clean/LEO权重和三个loss。
- [x] 验证6300个physical每轮各访问一次；复制一个view并分摊权重不改变该样本总loss；H0错误样本keep=0，无正确样本batch可反传。
- [x] 对加权CE/keep/metric分别提取梯度，不增加step；记录optimizer步数和`converged/budget_exhausted`。

```python
def test_keep_does_not_distill_h0_errors(keep_loss):
    s0 = torch.tensor([[0., 2.], [3., 0.]])
    sg = torch.tensor([[3., 0.], [1., 2.]], requires_grad=True)
    y = torch.tensor([0, 0])
    loss = keep_loss(s0, sg, y, gamma_max=1.)
    loss.backward()
    assert torch.equal(sg.grad[0], torch.zeros(2))
    assert sg.grad[1].abs().sum() > 0
```

运行：`python -m pytest code/tests/test_anchored_fit.py -q`。通过产物：A2/A3/A4和C_angle/C_angle_keep有可区分的配置与history。

### T4：source OOF与嵌套融合评价（R17–R19）

- [x] 新建`anchored_crossfit.py`、`test_anchored_crossfit.py`；以RX及physical ID生成排除集合，所有views随同一physical移动。
- [x] 在每个OOF行保存专家fit_rx；外层门控的训练记录不得含heldout RX；标准化/quality fit遵守相同边界。
- [x] 用计数模拟器验证5个4RX、10个3RX、1个最终模型的缓存复用，不执行重复拟合；预算选择只在内部折进行。

```python
def test_oof_is_disjoint_at_physical_level(oof):
    for fold in oof.folds:
        assert not set(fold.train_physical_ids) & set(fold.predict_physical_ids)
        assert fold.heldout_rx not in fold.expert_train_rx
        assert fold.heldout_rx not in fold.gate_train_rx
```

运行：`python -m pytest code/tests/test_anchored_crossfit.py -q`。通过产物：诚实source开发预测；骨干仍见过全部source RX的事实明确保留。

### T5：固定融合、utility和保护边界（R20–R25）

- [x] 新建`anchored_fusion.py`、`test_anchored_fusion.py`；实现稳定概率混合、所有实际动作、utility标签及小线性门控。
- [x] 测第三类获胜、alpha端点、零utility回退、utility并列、小alpha优先、保护区严格边界、并列H0处理和保护后动作标签一致。
- [x] 用禁止字段变异测试证明RX/TX/day/ID/scenario不能流入phi；验证专家tensor无梯度、类别置换时共享门控不变。

```python
def test_mixture_can_predict_neither_expert():
    p0 = torch.tensor([[.40, .35, .25]])
    pg = torch.tensor([[.25, .35, .40]])
    assert p0.argmax(-1).item() == 0
    assert pg.argmax(-1).item() == 2
    assert ((p0 + pg) / 2).argmax(-1).item() == 1
```

运行：`python -m pytest code/tests/test_anchored_fusion.py -q`。通过产物：A5动作审计和可选A6；无source收益则可合法停在A5，不编造动态收益。

### T6：一致的最终校准（R26–R28）

- [x] 新建`anchored_calibration.py`、`test_anchored_calibration.py`；只接受V最终log probabilities，拒绝重复fit或fit后改专家版本。
- [x] 验证正温度不改唯一argmax、alpha混合前后顺序、物理配对权重、概率下溢、ties实际覆盖；温度拟合不接受RX/TX专属参数。

```python
def test_final_temperature_preserves_decision(log_mix):
    before = log_mix.argmax(-1)
    for t in (.05, .5, 1., 2., 20.):
        assert torch.equal((log_mix / t).argmax(-1), before)
```

运行：`python -m pytest code/tests/test_anchored_calibration.py -q`。通过产物：所有候选匹配权限的校准、部署阈值和原始闭集结果。

### T7：source协调器、导出与预测接口（R29–R32）

- [x] 新建`anchored_pipeline.py`、配置和CLI；阶段为`cache/fit/oof/fuse/calibrate/export/profile`，默认不提供自动target串联。
- [x] 配置验证未知字段、角色、矩阵ID、后续分支开关、缓存身份、预算和输出占用；已经存在的输出报错，恢复只认本run未完成阶段。
- [x] 导出无cache/labels/RX/day数据路径；重载后alpha0与H0一致，候选顺序/批次切分/单样本预测不改变结果，前后state逐tensor不变。
- [x] source各阶段先用小型合法L_s样本完成端到端技术测试；不创建smoke审批文件。正式N607执行留待实现验收及相应实验授权，不因空闲GPU自动启动。

```python
def test_export_contains_no_training_rows(bundle):
    forbidden = {'source_cache', 'physical_ids', 'labels', 'receiver', 'day', 'dataset_path'}
    assert not forbidden & set(bundle)
    assert bundle['stage'] == 'Phase1'
```

运行：`python -m pytest code/tests/test_anchored_pipeline.py -q`及新CLI `--help`、`--validate-config`。通过产物：可执行source链路和冻结bundle，尚无新target结果。

### T8：独立P1和模式校准（R33–R38）

- [x] 新增E拟合器、pattern schema、版本化校准；复用既有边缘化和Schur实现，保持旧H1–H4路径不变。
- [x] 使用实际block schema生成full/单块缺失/全缺失，源V与评价保持同机制；低秩score只求解一次，NLL从结果复用。
- [x] 缺失位置填NaN/极值不影响观测结果；相同维数不同pattern不能共享阈值；未覆盖模式defer；块内原坐标不随mask重缩放。

```python
def test_missing_values_do_not_become_zero_evidence(score, z, mask):
    z_nan = z.masked_fill(~mask, float('nan'))
    z_big = z.masked_fill(~mask, 1e20)
    torch.testing.assert_close(score(z_nan, mask), score(z_big, mask))
```

运行：`python -m pytest code/tests/test_partial_evidence_patterns.py -q`，加现有partial Gaussian相关测试。通过产物：单独P1 source实验能力；不默认融合进G主线。

### T9：完整诊断、成本与文档交付（R39–R42）

- [x] 汇总第12节所有文件；用合成已知计数的预测验证rescue/harm、precision/recall/流入错误、各分组加权汇总一致。
- [x] 独占/受控profile分开IQ增强、身份骨干、head、fusion/calibration、保存；同步CUDA计时并报告warmup/重复次数和环境，峰值显存按阶段reset/readback。
- [x] 将参数/损失激活、源开发收益、未执行分支和科学限制写入同一run报告；检查完整产物后才记录完成。
- [x] 聚焦验证、一次适用P0/P1独立实验审查后，按AGENTS显式stage本次路径、commit/push并独立比对远端OID；文档检查不重新触发GPU实验。

运行：`python -m pytest code/tests/test_anchored_reporting.py -q`；`git diff --check`。通过产物：完整source开发报告及可复核部署产物，不含虚构的确认结果。

## 12.报告schema和完成定义

|文件|必须记录|
|---|---|
|`head_fit_history.csv`|candidate/fold/head_seed/epoch/step；clean/LEO CE、keep、metric；各加权梯度范数/夹角；physical/view访问；停止原因|
|`geometry_diagnostics.csv`|M特征值/条件数、Q正交误差、a变化、M变化、所有类方向偏移、范数/间隔分位数|
|`fusion_audit.csv`|raw与realized alpha、rescue/harm绝对数及占总数/各自基准数比例、净正确数、utility、改变比例、保护回退原因、所有RX/TX/day|
|`class_flow.csv`|所有类的预测占比、precision、recall、流入错误数、混淆计数；不设类3专属约束|
|`mask_calibration.csv`|pattern/quality/support_level、physical/view支持数、可用性、覆盖、拒绝原因、正确样本拒绝数|
|`calibration_report.json`|最终TF、固定内部尺度、V角色、物理权重、校准前后argmax一致、阈值规则与ties|
|`runtime_profile.json`|IQ/增强、身份骨干、头、融合/校准、保存分别计时；缓存命中、峰值显存、环境和共享负载限制|
|`protocol_manifest.json`|H0来源、当前数据契约、cache绑定、fold边界、训练/融合/V顺序、是否使用support/更新骨干、既有target观察声明|
|`oof_predictions.pt`|source逐行s0/sG/实际动作、fold与physical/view关联，训练侧保留，不进bundle|
|`frozen_system.pt`|冻结部署白名单状态、架构/类别schema、stage；无source样本cache|
|`report.md`|A0–A6/C_angle/P1完整或未启动原因、全部seed与分组、技术正确性/source收益/确认性证据分层|

完成分层：

- **PLAN_READY（已完成）**：完整报告要求已有去向，文件接口、默认起点、步骤和验收明确，文档已验证并Git交付。
- **IMPLEMENTATION_VERIFIED（本次已完成）**：T1–T9适用代码和直接正确性测试通过；deferred分支不冒充已实现，实际证据见验收报告。
- **SOURCE_ANALYZED（未来）**：实际source预算结束、全部source OOF与最终校准产物闭合；未收敛明确标注。
- **CONFIRMATION_COMPLETE（另行具备条件后）**：合法独立确认数据、冻结候选、先预测后评分以及完整性能/稳定性/成本证据。当前不存在这一结果。

## 13.最高风险和本次交付核对

最高实现风险是**融合训练标签对应的动作与实际部署动作不一致**：内部温度变化、保护裁剪、margin logits混入或不诚实OOF都会使所谓净收益失真。其次是缓存误用和把完整joint污染伪装成局部mask。计划通过统一`realize_actions()`、冻结顺序、嵌套OOF和观测schema直接约束这些问题。

最高科学风险是把已有target启发的新设计再次在同一target评分后，称为未见目标的独立确认。该风险无法靠重置head seed、从零重训或增加哈希消除。

计划编制阶段仅核对报告、源码、覆盖、链接、编码与Git diff。后续实施阶段的代码、测试、只读N607取证和有界技术缓存/拟合均另记于验收报告；未启动正式source矩阵或恢复自动监控。本计划中的超参数与门槛仍是建议起点，技术验收不赋予性能最优性或独立确认结论。
