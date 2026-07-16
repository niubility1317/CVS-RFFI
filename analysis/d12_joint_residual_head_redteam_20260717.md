# D12联合注册残差logit head红队门

日期：2026-07-17
范围：只读审查`AGENTS.md`、`项目.md`、主报告D12段及D5-D11追踪；本文不修改D12实现、runner或原追踪表。
当前判定：**D12在以下硬门全部通过前只能运行strict-K10 support-only，不得创建candidate-bound query。**

## 0.WIP实现级复核

当前已出现：

- `code/cvsrffi/stage2_joint_residual_logit_head.py`
- `tests/test_stage2_joint_residual_logit_head.py`

聚焦测试为`11 passed`，但这只证明当前测试自洽，不能授权正式runner。实现级结论如下。

### 已正确落地

1. `_validate_old_lineage_exact_reuse`按`label+rank`逐行核对`physical_sample_id`、父IQ SHA和feature SHA，Before/After旧support exact reuse门是实质检查，不是只比类数。
2. `evaluate_joint_leave_two_out`在每fold重新拟合Before old K8和After old K8+new K8；old/new各held2不进入head或prototype拟合，held old面对全部旧+新类。
3. query打分公式本身是单行、全注册类argmax，没有query role/quota/global assignment参数。
4. 参数量公式正确：`D=288,r=8,C=11`时为2,392。

### 当前NO-GO阻断

|优先级|阻断|当前证据|最小修复|
|---|---|---|---|
|P0|D12没有复用D11 artifact，重新实现了带`Callable`的builder和模块级`_ARTIFACT_TOKEN`|外部可导入`d12._ARTIFACT_TOKEN`直接构造任意feature artifact；实测伪造成功|D12核心只消费统一authority artifact；正式runner闭包生成不可伪造receipt，D12不得重建token/callback入口|
|P0|runtime/code/checkpoint SHA仍是调用方字符串|builder只检查长度64，未从实际runtime/code/checkpoint bytes重算|runner从authority allowlist和实际bytes重算并写COMMIT；核心核验receipt|
|P0|state content SHA没有覆盖hyperparameters和resource|把`alpha=0.10`改为`0.20`后沿用原content SHA，构造成功；把resource中的activation改为`forged`也成功|canonical hash覆盖全部超参数、resource、operator/view和序列化metadata；resource使用不可变映射|
|P0|alpha0只存在于私有score单测，不是可拟合/可选择的回退candidate|`_validate_hyperparameters`要求`0<alpha`；实测正式fit的alpha0直接失败|允许并预登记alpha0；selector必须显式包含它，并验证state/prediction等价base|
|P0|fold内K8 state错误绑定full-K10 artifact SHA|L2O调用`_make_state(before_artifact,old_rows[train_old],...,k_shot=8)`，但state保存完整K10 artifact SHA；After同样如此|为每fold构造只含train K8的subset authority fingerprint，或保存明确的fold-train row root；full-K10 SHA不能冒充K8 fit证据|
|P0|query未绑定candidate-bound state package/COMMIT|predict只核对state content及runtime/code/checkpoint字符串，没有state package SHA、COMMIT SHA或prediction receipt|formal predictor只加载sealed state package，核验文件SHA、content SHA、COMMIT及query package SHA|
|P0|operator/view策略没有进入state或predict绑定|artifact有`operator_id/view_seed`，state不保存；Before/After也不检查二者一致，query可换operator而不触发当前绑定门|state hash保存operator/view policy；Before/After/query必须逐项一致，除非预登记的统一多view schema|
|P1|状态字节只计prototype/W1/W2 tensor|class strings、hyperparameters、binding metadata和序列化容器未进入持久状态字节|同时报告tensor bytes和实际部署package bytes，以后者执行256KiB硬门|
|P1|训练日志同一epoch的accuracy与residual幅度可能来自optimizer step前后两个状态|`logits`在`optimizer.step()`前生成，随后重新调用`model.residual`记录幅度|step后统一重算全部训练诊断，或明确记录pre-step/post-step|

因此当前WIP只能作为算法原语。即使后续support性能通过，以上P0未清零时仍必须写`SUPPORT_PROTOCOL_BINDING_INCOMPLETE_NOT_SELECTED`。

## 1.攻击面与准入门

|ID|攻击或失败模式|D12硬门|最小反例测试|
|---|---|---|---|
|RT-P01|普通`Mapping[str,np.ndarray]`可直接进入fit/register/predict|D12核心API只接受runtime-authorized artifact；fit、After联合fit与predict均拒绝普通Mapping、数组和自定义对象|把形状、标签和数值都合法的普通dict传入三个正式入口，必须在任何训练/计分前fail closed|
|RT-P02|D12重新暴露任意feature callback，或仅用Python下划线隐藏factory|复用既有D11 validated artifact类型，但D12模块不得新增callback/`Callable`参数、public factory、token或raw-feature wrapper；正式runner中的extractor由sealed runtime闭包固定，物理batch恒为1|检查公开签名无callback；传入“首行依赖第二行”的extractor不可到达D12；若能通过导入私有token伪造artifact，则正式runner还必须要求不可伪造的authority receipt，否则NO-GO|
|RT-P03|64字符自报runtime/code/checkpoint/feature SHA被当成真实绑定|所有SHA必须由runner从实际bytes、allowlist和authority manifest重算；COMMIT绑定module、runner、runtime、checkpoint、输入package、feature artifact和state实际SHA，自声明字符串不得成为authority|提供长度正确但内容伪造的SHA，或改一字节代码/checkpoint/IQ后沿用旧SHA，必须在fit前失败|
|RT-P04|artifact的feature与实际received-IQ脱钩|逐物理sample核验`physical_sample_id`、实际IQ SHA、operator、view seed、feature SHA；D12不得从外部直接接收预造feature而跳过formal extraction receipt|同一IQ分别注入全`+1`和全`-1`feature，二者都不得被接受；替换IQ但保留旧feature/SHA必须失败|
|RT-P05|state数组、类别顺序、alpha或metadata在fit后被篡改|state张量必须bytes-backed readonly；每次predict前重算content SHA。哈希覆盖prototype、W1/W2、bias/temperature（若有）、alpha、activation、class order、K、注册代次、超参数、support/runtime/code/checkpoint绑定和资源字段|原地改W2/prototype失败；重建一个数组未改但alpha、class order、resource或绑定字段改变的state也必须因content SHA失败|
|RT-P06|state文件内容合法但换壳、替换metadata或加载错误COMMIT|formal predictor只加载candidate-bound只读state package，并复算文件SHA、content SHA及COMMIT引用；不得接受内存临时state或未提交state|交换两个candidate的NPZ/JSON、只替换metadata、或删除COMMIT任一引用，预测必须停止|
|RT-P07|K5/K1进程可达K10余量，或D12内部对K10切片冒充exact-K|package pre-open、allowlist和loader输出对每类必须恰好K个物理ID；K5/K1使用独立密封包，D12无“取前K”路径|manifest写K5但payload含10条/类，即使rank只选择0—4也必须在feature extraction前失败|
|RT-P08|同一物理样本多view被计为多个support，或同一fold只held一个view|所有view按`physical_sample_id`成组；held一个物理sample时其全部view同时held；view数不增加K|复制同一physical ID形成两行或只从base held、辅助view仍留在train，必须失败|
|RT-P09|After联合head的held old/new泄漏进W、prototype、alpha选择、蒸馏teacher或normalization|每fold旧类和新类各held2；只用old K8+new K8拟合After W1/W2与全部prototype。old held/new held及其全部view不得进入任何loss、统计、early stop、候选选择或teacher；fold内Before teacher也只能由old K8构建|把held行改成极端值，训练state/content SHA必须不变而held预测可变；若训练state随held内容变化即泄漏|
|RT-P10|先用full K10拟合head，再只重算prototype做“L2O”|每fold必须重新初始化并只在fold-train拟合head；full-support fit只能在统一候选锁定后用于最终support-only state，不能产生promotion指标|构造可记忆support的高容量样本：full-support accuracy=1但真正L2O差；候选必须回退base，不能因full-support loss低晋级|
|RT-P11|旧head/旧prototype冻结被误当作无遗忘|After held-old必须面对旧+新全部注册类；新增类抢占旧query直接计入forgetting和逐类old退化|构造新prototype与old0重合、旧state完全不变的样本；门必须检测old0→new0并拒绝|
|RT-P12|support loss更低覆盖逐类old退化|候选资格先执行每场景每个旧类非退化硬门，再比较new floor、H和joint；训练loss只用于日志，不能进入越过硬门的排序|候选A loss较低但任一旧类下降，候选B为alpha0且不退化；必须选择B|
|RT-P13|没有真正的alpha0回退，或alpha0仍执行残差造成数值漂移|候选集合显式包含`alpha=0`；回退state的residual contribution严格为0，预测与base cosine逐位或在预登记容差内一致；可跳过W分支并报告实际MAC|随机query上比较alpha0 D12与同prototype base cosine的scores/predictions；不一致即失败|
|RT-P14|按query真实old/new角色走不同head、bias或阈值|query入口仅接收一个sealed物理query；同一公式对全部注册类计分，无role、truth、quota、batch count、排序、global assignment参数或side channel|检查签名与execution receipt；同一query单独预测和混入不同大小/顺序批次时结果相同，且runner每次只提交1行|
|RT-P15|重复/伪造未绑定query feature或query参与拟合|formal query artifact绑定实际IQ SHA、唯一物理ID、runtime/code/checkpoint/state SHA；prediction前后state content SHA相同，query update/fit/selection/rollback均为0|同一token换IQ、同一IQ换token、任意单行feature重复提交均失败；预测前后state SHA必须一致|
|RT-P16|只报告自报参数、状态和MAC，遗漏prototype/metadata/运行时实际开销|从实际dtype/shape和序列化文件计状态；参数计入W1/W2及所有trainable bias/scale；报告head+prototype MAC、平均/P95 forward/FFT、singleton时延、峰值显存、enrollment时长及相对identity-only单qKNN Pareto|增加一个未计数bias或metadata数组、伪报state bytes、超过256KiB、参数>12k或epoch>15均fail closed|

## 2.联合old/new性能门

D12 support promotion必须在三个场景使用同一`rank/alpha/activation/loss权重/epoch/seed策略`。不得按场景选择arm，也不得以平均值掩盖单场景失败。每个joint fold的唯一合法数据流为：

```text
old K8 + new K8
  -> fit After joint residual head and fold-local prototypes
old held2 -> score over all old+new classes
new held2 -> score over all old+new classes
```

必须同时输出并按同一fold聚合：

- `before_old_overall`、`before_old_per_class`和`before_old_floor`：fold-local Before状态，仅在旧类上计分；
- `after_old_overall`、`after_old_per_class`和`after_old_floor`：同一old held面对全部注册类；
- `after_new_overall`、`after_new_per_class`和`after_new_floor`；
- `joint_accuracy`、`H_old_new`；
- `old_forgetting=before_old_overall-after_old_overall`及逐类old delta；
- residual幅度统计：`alpha`、base/residual logit范数比、最大绝对logit修正和回退原因。

候选排序必须是硬门优先：

1. 每个场景、每个旧类`after>=before`，且总体old forgetting不大于0；
2. 每个场景new overall与new floor均优于可比较基线；
3. 最差场景new floor；
4. 平均`H_old_new`；
5. 平均joint accuracy；
6. 完全并列时优先`alpha=0`、更小rank、更少epoch和更小状态。

不得把更低support loss、full-support accuracy或旧参数bitwise锁定放在逐类old非退化门之前。

### 基线口径警告

- D11-v6提供可比较的joint L2O new overall/floor：clear`0.52/0.10`、low`0.46/0.20`、rain`0.60/0.40`，同时提供同fold old、H和forgetting。
- D10现有D8b数字是11类support删除验证的总体/floor，并非明确的joint-held new-only指标。D12不得把D10的`0.50/0.20`、`0.46/0.10`、`0.70/0.50`直接当作new-only门。若要声称超过D10，必须先用与D12完全相同的old/new held2、全注册类竞争和聚合口径重算D10；否则只标`NOT_COMPARABLE`。

## 3.support过拟合专项判据

残差head虽只有约2.4k参数，但K10/new5时每场景只有110个support，W2又是类特异列，仍可能记忆support。以下任一现象出现即不得开放query：

- full-support accuracy接近1而joint L2O的旧类、新类或floor明显下降；
- 选中的非零alpha只改善平均值，却在任一fold/场景造成旧类退化；
- residual/base logit范数比或单类修正异常大，性能主要依赖少数support margin翻转；
- 候选网格扩大后winner频繁变化，或只靠某一个fold/场景通过；
- label permutation后仍得到异常高held准确率，提示fold泄漏；
- 不同训练seed方差大，统一arm不能稳定复现。

最小稳定性报告应包含每fold同row指标、三场景最差值、训练seed重复、alpha0配对差值和每类混淆。支持集内门只能授权“一次锁定候选的后续独立query”，不能证明最终泛化。

## 4.资源审计清单

|项目|至少记录内容|
|---|---|
|参数|W1、W2、所有bias/scale/temperature的实际trainable参数；new5 After目标<3k，硬门≤12k|
|状态|prototype、W1/W2、bias/alpha、class order、binding metadata和序列化文件实际字节；硬门≤256KiB|
|训练|每fold与full fit的epoch、loss各项、early-stop状态；首轮≤15epoch，完整JSONL|
|MAC|base prototype cosine、W1、activation、W2和argmax分别计数；alpha0回退报告实际跳过残差后的MAC|
|forward/view|每物理sample的平均/P95 backbone forward、operator/view和FFT次数；view不增加K|
|时延|support enrollment总时长、singleton query warm-up后平均/P95；注明设备、重复次数和同步策略|
|显存|训练与singleton推理的peak allocated/reserved，不能用support batch峰值冒充query峰值或反之|
|Pareto|与同checkpoint、同feature/view、同注册类的identity-only单qKNN比较MAC、时延、显存、状态和准确率变化|

## 5.最小发布门

D12只有同时具备以下证据才可从support-only进入一次candidate-bound query：

1. strict-K10三场景包pre-open与exact-K通过；
2. runtime-authorized artifact闭环通过，D12未重新暴露Mapping/callback/self-report SHA；
3. state不可变、content SHA、COMMIT代码身份和加载闭环通过；
4. 真正joint old/new L2O无held泄漏；
5. 每场景每旧类非退化，old forgetting≤0；
6. new overall/floor、`H_old_new`和joint指标通过预登记同口径门；
7. alpha0为可执行且数值等价的显式回退；
8. 参数、epoch、状态、MAC、时延和显存硬门全部通过；
9. query接口逐物理sample、全注册类、无role/quota/query fit；
10. 写入`SUPPORT_ONLY_D12_SELECTED_QUERY_AUTHORIZED`的不可变COMMIT；任一项失败则写`SUPPORT_ONLY_D12_NOT_SELECTED_NO_QUERY_OPEN`。

## 6.追踪状态

|ID|要求|状态|验证|
|---|---|---|---|
|RT-01|协议攻击门清单|verified|已覆盖Mapping、callback、自报SHA、state篡改、exact-K、role/quota和query fit|
|RT-02|held old/new联合fold隔离|verified|已明确old/new各held2及所有fit/teacher/statistics禁入|
|RT-03|性能与过拟合门|verified|已明确逐类old优先、new floor、H、forgetting和alpha0回退|
|RT-04|最小反例测试|verified|16组fail-closed反例|
|RT-05|资源审计项|verified|参数、状态、MAC、forward/view、时延、显存和Pareto|
|RT-06|D12当前实现符合性|blocked|WIP单测11 passed，但存在7项P0：artifact/callback authority、self-report SHA、state hash、alpha0、K8/full-K10绑定、state package COMMIT和operator/view绑定|

反向审计：5项`verified`，0项`deferred`，0项`rejected`，1项`blocked`。当前代码中的fold训练数组隔离已成立；最高风险转为“临时K8 state绑定full-K10 artifact SHA”和artifact/state authority可伪造，使正确的数值fold不能形成正式可审计证据。

## 7.实际验证

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q tests/test_stage2_joint_residual_logit_head.py
...........                                                              [100%]
11 passed
```

只读攻击脚本实测：

```text
alpha_mutation_accepted 0.2 True
resource_mutation_accepted forged True
private_token_forgery_accepted <artifact_sha> (2, 32)
alpha0_formal_fit JointResidualLogitHeadError hyperparameter/resource drift
fold_k8_binds_full_k10_before_sha 8 True
query_operator_mismatch_accepted different_operator 999 new0
```

`git diff --check -- analysis/d12_joint_residual_head_redteam_20260717.md`通过。本文未stage、未commit。
