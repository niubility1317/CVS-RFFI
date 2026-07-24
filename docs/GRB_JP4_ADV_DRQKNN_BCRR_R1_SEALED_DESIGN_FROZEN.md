# GRB-JP4-ADV-DRQKNN-BCRR/r1-sealed设计冻结

状态：`DESIGN_FROZEN`

监督状态：`DESIGN_FROZEN / AWAITING_INDEPENDENT_REVIEW`。本冻结文档不构成`MERGE`、`P0=0`、`P1=0`或`P2=0`结论；这些结论只能由固定可审diff上的独立review签发。

协议：`p2_min_v1`

## 1.候选与证据边界

候选ID为`GRB-JP4-ADV-DRQKNN-BCRR/r1-sealed`。它在现有r6双qKNN＋BCRR共享执行链前增加一个ground-receiver-basis joint-projection适配器：Phase1地面知识只定义4维模型更新子空间和旧类聚合锚点，当前row的target-old support只估计4个共享系数。Stage2-C冻结该模型增量，用同一适配模型编码并append新类。

当前证据仅为真实checkpoint上的`FEASIBILITY_SPIKE_NON_FORMAL_NOQUERY`：q4 ground basis condition=`2.5324`、energy=`0.7503`，解析Jacobian与autograd参考的最大误差小于`1e-6`；3个scene×K1/K5/K10共9个support组合中，残差9/9下降、margin 9/9变化、neighbor class 6/9变化，query读取为0。该证据证明真实层可动和闭式求解可执行，不证明held性能。

D93/D94的ground→target全坐标transport负结论继续有效。本revision只更新一个真实权重层、自由度为4、使用target support估计系数并连续收缩；ground不直接参与query score，也不加入qKNN bank。若完整125中`M_DA`不能优于ground-off对照，该重入假设即被证伪。

## 2.合法输入与禁止输入

正式输入仅包括：

- checkpoint绑定的ADV3B02 sealed runtime；
- 与同一checkpoint通过既有joint-seal container、固定authority、production signature和method lock共同封存的Phase1 INT8聚合`P_g/L_g/R`；
- 当前row固定LEO弱观测的target-old support；
- Stage2-C当前row的新类support；
- r6既有qKNN、BCRR、GEOFF/r8 archive与coverage执行接口。

每次正式入口必须重新验证既有outer bundle的固定authority signature、detached seal和全部成员SHA，并由该次同文件描述符验证直接重新materialize runtime；正式fit不得消费调用方内存中可替换的runtime、component或协调receipt字段。当前live runtime的结构/parity、method lock、checkpoint与GRB组件只以该次重验的8-member outer bundle为准；该要求不新增authority、gate或sidecar。

禁止输入包括raw/clean/source IQ、单样本source feature、成员或physical ID列表、source replay、独立可替换sidecar、目标receiver标签、query、query view、query truth/role、真实batch类计数、class quota和跨query图。当前`inputs_unverified/phase2_zid_prototypes.pt`只能保留为非正式spike证据，不得进入正式run。

本candidate继续复用既有`p2_min_v1`、`VALIDATED_ONCE`、capsule、split、receiver/TX、scene、K和support/query划分。方法、bundle、rank、checkpoint或资源变化不触发Phase2数据重验。

## 3.Phase1 ground组件

Phase1 exporter已对每个source domain形成class-balanced公共`z_id`变化向量`d_r∈R^160`。正式builder只读取target访问前的多样本聚合export，不读取sample成员。对所有有效source domain先减去固定均值，再作canonical SVD：

```text
D_centered = [d_1-d_bar; ...; d_R-d_bar]
D_centered = U diag(sigma) V^T
L_g = V[0:4, :]
kappa_G = sigma_1 / sigma_4
```

每个奇异向量以最大绝对值坐标为正进行符号规范化。旧类聚合锚点`P_g∈R^(6×160)`来自同一Phase1 export并逐行单位化。固定checkpoint权重`W_0=joint_proj.0.weight∈R^(160×320)`的前4个canonical右奇异向量构成`R∈R^(4×320)`。

`P_g`、`L_g`和`R`均使用逐向量对称INT8 codes＋FP16 scale封存；`kappa_G`、旧类顺序、生成规则digest、checkpoint SHA和method lock共同绑定。组件不得包含source路径、sample count、成员索引或可逆source状态。实现沿现有8-member joint-seal与固定签名链做最小schema扩展，不新建authority、receipt或validator体系。

## 4.Stage2-B闭式模型适应

唯一模型增量作用于：

```text
W_B = W_0 + DeltaW
DeltaW = L_g^T diag(theta) R
```

对target-old support样本`i`，令`z_i`为当前单位化`z_id`，`J_i∈R^(160×4)`为4个冻结权重方向在ReLU和L2归一化后的解析Jacobian，`b_i=z_i-P_g[y_i]`。按class与physical sample等权最小化：

```text
sum_c (1/K) sum_{i:y_i=c} ||b_i + J_i theta||_2^2 + lambda ||theta||_2^2
```

实现按样本流式累积`T=A^T A∈R^(4×4)`和`u=A^T b∈R^4`，禁止物化`(6K×160)×4`的完整`A`。K5/K10使用：

```text
lambda = 0.01 * trace(T) / 4
theta_ridge = -(T + lambda I)^(-1) u
g_K = K / (K + 4)
kappa_H = cond(T + lambda I)
s_kappa = min(1, sqrt(kappa_G / kappa_H))
tau_W = ||W_0||_F / sqrt(160)
s_W = min(1, tau_W / ||DeltaW_ridge||_F)
theta = g_K * s_kappa * s_W * theta_ridge
```

`tau_W`只由checkpoint确定并随method lock预锁。不得按support残差、scene、receiver、class、accuracy、query或held结果改变rank、ridge、shrink、层或启用状态。

K1按当前活动目标精确identity：`theta=0`。K5是首个正式falsifier，K10确认同一公式。rank只写诊断；当输入有限且`trace(T)>64*eps_float64`时，即使数值rank小于4也正常执行ridge与连续收缩。`trace(T)`近零或任一数值非有限时确定性`theta=0`并继续生成prediction/score，不得把低coverage或低rank升级为技术失败。wrong seal/hash、非法INT8、hook漂移或query进入fit仍按P0停止。

最终`theta`量化为INT8 codes＋单个FP16 scale；模型合并只使用量化回放值。不得持久化FP32 ground或theta sidecar，也不得保存完整`DeltaW`。适配权重在现有模型内原位合并，序列化恢复只依赖冻结checkpoint与INT8增量状态。

## 5.Stage2-C生命周期

`S_B`产生后逐字节冻结：

- `theta`及其量化状态；
- 适配后的`joint_proj.0.weight`语义hash；
- old`z_id`bank前缀；
- r6的`Q/A/rho/alpha`与旧类bandwidth；
- old BCRR状态；
- checkpoint、ground bundle和method lock绑定。

Stage2-C不重新拟合JP4、ground basis、theta、ridge、shrink、旧bank或BCRR旧状态。新类support只经同一适配模型编码并append；新类没有ground身份原型。每个query独立面对全部注册旧类和新类。

`VerifiedADV3B02DeploymentBundle`只能由已完成外部authority signature、detached seal和member SHA验证的production factory构造。`FormalGRBJP4State`的public constructor必须拒绝直接调用，正式runner只通过fit/append orchestrator生成状态；该API限制用于防止普通误用，不把Python同进程任意反射、monkeypatch或closure introspection伪装成不可绕过的authority边界。真正的P0/P1接受条件是：每次fit重新验证外部签名链和全部成员，append/predict每次核对当前runtime对象身份、完整TorchScript方法图、全部parameter/buffer状态指纹、bundle与method-lock绑定、同row class/token/IQ闭包、state coordinator hash和真实弱引用生命周期。复制一个逐字段相同且通过全部内容验证的对象不产生新的科学状态；任何runtime语义、support、class、token、JP4、qKNN、BCRR、resource或lifecycle字段变化都必须失败关闭。仅有相同`joint_proj.0.weight`但forward语义不同的runtime不得进入正式生命周期。

## 6.五臂与互补机制

|arm|JP4|qKNN|BCRR|
|---|---|---|---|
|`M0`|关闭|基础`z_id`Student-t qKNN|关闭|
|`M_DA_NG`|关闭|逐字节复用r6 no-ground双qKNN|关闭|
|`M_DA`|开启|适配后`z_id`＋同一r6双qKNN|关闭|
|`M_OTHER`|关闭|与`M0`共享基础bank|开启|
|`M_JOINT`|开启|与`M_DA`共享模型、bank和DomainState|开启|

`M_DA-M_DA_NG`隔离ground驱动模型增量。`M_DA/M_JOINT`必须共享相同JP4、qKNN bank和DomainState hash；`M0/M_OTHER`必须共享相同基础bank hash。K1时r6 DA与JP4均identity，因此`M_DA_NG=M_DA=M0`且`M_JOINT=M_OTHER`。

JP4通过输入相关、非正交的真实模型权重增量改变ReLU active set和单位球角距离，目标是改变邻居、margin、argmax与净正确决策。BCRR不改变encoder或邻域，只使用同physical-ID support的LOO残差修正剩余近边界分数。二者的互补性只能由held五臂结果证明。

协同量固定为：

```text
I_syn = H(M_JOINT) - H(M_DA) - H(M_OTHER) + H(M0)
```

## 7.资源与INT8合同

- 拟合自由度：4；optimizer step：0。
- `P_g`：972B；`L_g`：648B；`R`：1,288B；`theta`：6B；数值payload合计2,914B。
- JP4含metadata/hash的wire上限：4,096B。
- parent r6最大state：159,691B；联合上限：163,787B，小于256KiB。
- FP16合并scratch上限：102,400B；不得同时保留第二份完整模型权重。
- formal入口必须先消费调用方Verified bundle持有的runtime，再materialize重验runtime；FP32 teacher、INT8 deployed和返回base runtime之间均采用release→reload单所有权转换。每次release必须以真实TorchScript对象的弱引用已失活为证据；保留任何外部runtime引用时必须失败关闭。materialize/release/reload次数与最大live实例数只能由这些实际弱引用观测导出并写入resource receipt，禁止手填或硬编码推断。
- 合并后每query额外adapter MAC：0；r6在`C=26,K=10`的head MAC继续为42,466。
- 一次性support forward、解析Jacobian、4×4求解、weight merge MAC、fit时延、query时延、峰值显存和总state必须进入既有resource receipt。
- INT8 theta要求只在support端以FP32 theta teacher和INT8 theta deployed表征执行同一冻结teacher-support qKNN决策审计，`top1 agreement≥0.995`且`large-margin flip=0`；审计不得读取query，结果及support/hash/state绑定进入formal fit-state和resource receipt，失败属于技术失败，不得作为性能结果。

## 8.完整125与立即证伪

首个正式实验直接执行完整125：5receiver×5seed×`{K10/new5,K10/new10,K10/new20,K5/new20,K1/new20}`，每job覆盖3个LEO弱场景和5个arm。闭包为125 jobs、375 scene slices、1,875 score rows、1,250 arm-state prediction artifacts。

立即证伪条件：

1.K5/K10的`M_DA`相对`M0`没有净正确决策正收益，或不优于`M_DA_NG`；
2.`M_DA`只改变logit而没有真实neighbor、margin、argmax或净正确变化；
3.`M_DA`使old、seen-new、floor、min-old、min-new任一净变化为负，或增加forgetting；
4.`M_OTHER`相对`M0`没有独立正收益；
5.`M_JOINT.H<=max(M_DA.H,M_OTHER.H)`；
6.mean`I_syn<=0`，正协同少于188/375个scene slice，或不足2/3个scene具有正协同均值；
7.old→new、new→old、INT8、state、MAC、时延、显存或协议门失败。

完成后必须按同row报告old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、双向混淆、逐类、receiver、scene、K、seed、new-count、ground-on/off净正确变化和资源。

## 9.冻结实现范围

允许修改：

- 新增最小Phase1 GRB-JP4 compact component builder/loader；
- 对现有ADV3B02 joint deployment bundle增加同容器、同authority的GRB component profile；
- 将现有非正式`stage2_grb_jp4_adv_drqknn_bcrr.py`收敛为正式bundle/state、闭式fit、merge、append和五臂编排；
- 新增对应thin full125 runner与专项测试；
- 修正旧spike JSON的错误`REJECT_SPIKE`状态，但不得把它升级为性能证据。

禁止修改r6双qKNN公式、BCRR公式、GEOFF/r8数据入口、support/query split、共享scorer、truth绑定、完整125 slice或资源上限。若实现需要改变核心层、rank、loss、shrink、ground格式、qKNN或OTHER，必须创建新revision并重新进行可行性审查。

正式发布前必须生成production-signed联合bundle。若现有固定authority无法产生合法signature envelope，本revision停在`BLOCKED_P0_FORMAL_ASSET / NO_PERFORMANCE_RESULT`；不得使用unverified/unsigned组件绕过，也不得新建authority系统。

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
