# D4a单接收观测FloorLock设计预注册

日期：2026-07-17
状态：`DESIGN_ONLY_NOT_IMPLEMENTED_NOT_RUN`
目标阶段：正式Stage2-B/Stage2-C
候选暂定名：`d4a_single_observation_loo_floorlock`

## 1.设计目的与证据边界

D4a针对当前Phase2主线的三个共同问题设计：

1.在单个target receiver上仅凭少量target support完成旧类目标域适应；
2.注册5/10/20个真实target-new TX时，防止新类logit侵入旧类，尤其保护最弱旧类；
3.在不使用query拟合、角色Oracle、类别配额或query图的前提下，同时提高`old_acc`、`min_old_class_acc`、`seen_new_acc`和`H_old_new`，并降低注册遗忘。

旧D1/D3实验使用同一物理IQ生成多种LEO状态平行观测，不符合`项目.md`2026-07-17新增的单物理样本单LEO接收观测协议。它们只能用于解释历史机制，不得作为D4a的当前正式baseline、超参数选择证据、Pareto对照或晋升依据。D4a必须在新生成的单观测合法数据包上与当前合法baseline同row重跑。

## 2.不可改变的协议边界

### 2.1每scenario原子row独立K-shot

每个`receiver×seed×scenario×K×new规模`是一个独立原子row。该row中每个注册类只能使用K个互不重复的物理support样本：

```text
S_row(c)={x_c,1,...,x_c,K}
physical_sample_id(x_c,i)全部不同
```

每个`x_c,i`在进入Phase2前已且仅已叠加一种`leo_*_weak`状态，只有一份密封接收IQ。D4a禁止：

- 把同一物理IQ的`clear/low_elev/rain`副本合并成3K support；
- 把三个scenario的support拼接后训练共享head；
- 在一个scenario拟合的adapter、prototype、bias或其它状态传给另一scenario；
- 把同一个接收IQ的计算view计作额外物理support或增加K。

三场景只共享候选定义和锁定超参数。三个scenario分别拟合、分别产生不可变prediction、分别由独立scorer评分。matched三场景的support物理ID集合必须两两互斥，query物理ID集合也必须两两互斥。

### 2.2固定接收IQ上的合法计算view

Phase2只接收唯一密封LEO_weak IQ`x`。允许的representation view必须满足：

```text
view_v = T_v(x)
same IQ payload
same physical_sample_id
same scenario
same satellite seed
same support/query role
additional physical sample count = 0
additional LEO channel state generation = false
```

D4a首版只使用base IQ一次提取的288维特征，并在特征空间进行确定性小扰动，避免额外backbone/FFT前向：

```text
h_i = normalize(
  concat(
    normalize(z_id160(x_i)),
    4 * normalize(FFT96(x_i), RF32(x_i))
  )
)

h_i^(0) = h_i
h_i^(+) = normalize(h_i + sigma * r_i)
h_i^(-) = normalize(h_i - sigma * r_i)
sigma = 0.01
```

`r_i`必须由opaque support token和candidate lock确定性派生，不能读取TX明文、query信息或外部随机状态。三个feature view共享同一接收IQ和同一次backbone/FFT输出，不能被表述为三次卫星观测、三种LEO状态或3-shot。

K≥2时，主稳定性约束采用leave-one-physical-sample-out。K=1压力评估时不能伪造第二个物理sample，只允许在同一IQ的三个合法feature view之间执行leave-one-view-out，并继续按1-shot报告。

### 2.3禁止的数据与决策信息

D4a的适配、注册、校准、早停和预测均不得使用：

- clean/raw IQ、clean feature、clean logit或其它clean-derived signal；
- source样本、source cache、source prototype、source统计或checkpoint外source artifact；
- query IQ/feature参与适配，query标签、query真实old/new角色；
- query真实批次类别数量、每类quota、query排序或全局分配；
- Hungarian、optimal transport、batch reassignment或dense query图。

正式预测对每个query独立计算所有已注册类别的logit，并执行一次全类argmax。只有不可变prediction发布后，隔离scorer才能连接truth和角色计算指标。

## 3.特征与符号

一个原子row中，注册类集合为：

```text
C_old={1,...,C_o}
C_new={C_o+1,...,C_o+C_n}
C_all=C_old union C_new
```

旧类数当前为`C_o=6`，新类数`C_n∈{5,10,20}`。每个support特征`h_i∈R^288`，由`z_id160+FFT96+RF32`构成。

共享对角尺度为`s∈R^288`：

```text
q_s(h)=normalize(exp(s) elementwise_mul h)
```

所有尺度元素限制在：

```text
-1.5 <= s_j <= 1.5
```

温度固定为：

```text
tau=18
```

## 4.Stage2-B：旧类LOO-Floor适应

### 4.1leave-one-physical-sample-out原型

对旧类`c`及其support样本`i`，训练时使用不含样本`i`的球面原型：

```text
p_c^(-i)(s) =
normalize(
  sum_{j:y_j=c,j!=i} q_s(h_j)
)
```

K=1时使用同一IQ的合法feature view执行leave-one-view-out：

```text
p_c^(-v)(s) =
normalize(
  sum_{u in {0,+,-},u!=v} q_s(h_i^(u))
)
```

训练logit为：

```text
l_i,c = tau * cosine(q_s(h_i),p_c^(-i)(s)) + b_c
```

Stage2-B只训练：

- 288维共享对角尺度`s`；
- 6个旧类bias`b_old`。

不训练288×6自由旧类权重。最终部署旧类原型使用全部K个物理support重新计算：

```text
p_c(s)=normalize(sum_{j:y_j=c}q_s(h_j))
```

### 4.2class-balanced总体损失

逐类交叉熵：

```text
CE_c = mean_{i:y_i=c} cross_entropy(l_i,y_i)
L_balanced = mean_c CE_c
```

该定义防止高质量旧类或样本数量差异掩盖弱旧类。

### 4.3逐旧类floor损失

对旧类support样本定义LOO旧类margin：

```text
m_i,c =
l_i,c - max_{o in C_old,o!=c} l_i,o
```

逐旧类margin风险：

```text
M_c =
mean_{i:y_i=c} softplus(margin_old - m_i,c)

margin_old=0.50
```

使用平滑worst-class聚合：

```text
L_floor =
beta_floor * logsumexp(M_c / beta_floor)

beta_floor=0.10
```

Stage2-B完整目标：

```text
L_B =
L_balanced
+ lambda_floor_B * L_floor
+ lambda_scale * mean(s^2)

lambda_floor_B=0.50
lambda_scale=0.05
```

`min_old_class_acc`是与overall old并列的优化目标和硬推进门。任何只提高平均old而降低floor的候选直接失败。

## 5.Stage2-C：冻结旧类的FloorLock增量注册

### 5.1冻结边界

Stage2-C必须从同row Stage2-B不可变COMMIT加载父状态，并bitwise冻结：

- 共享对角尺度`s`；
- 全部旧类原型`p_old`；
- 全部旧类bias`b_old`；
- 特征定义、温度和计算view规则。

Stage2-C不得重新训练或漂移旧类状态。before与after必须复用同一旧类query和同一query representation策略。

### 5.2新类稳健原型与受限残差

新类`n`的初始球面原型为：

```text
p_n =
normalize(
  mean_{i:y_i=n} q_s(h_i)
)
```

新类分类权重仅允许在原型附近学习小残差：

```text
w_n=normalize(p_n+delta_n)
norm(delta_n,2)<=rho
rho=0.15
```

Stage2-C只训练：

- 每个新类的288维`delta_n`；
- 每个新类一个bias`b_n`。

旧类logit：

```text
l_i,c^old =
tau*cosine(q_s(h_i),p_c)+b_c
```

新类logit：

```text
l_i,n^new =
tau*cosine(q_s(h_i),w_n)+b_n-g
```

其中`g`是训练完成后仅由旧类support确定的共享new安全offset。

### 5.3新类分类与类间分离

新类support交叉熵只在全部已注册类上计算：

```text
L_new =
mean_{i:y_i in C_new}
cross_entropy(
  concat(l_i^old,l_i^new),
  y_i
)
```

新类间球面分离：

```text
L_sep =
mean_{n!=m}
relu(cosine(w_n,w_m)-0.35)^2
```

该项只使用注册support派生的新类权重，不访问query类别数量或query图。

### 5.4逐旧类侵入与floor保护

对旧类support样本`i`及其真实旧类`c`：

```text
u_i,c =
margin_intrusion
+ max_{n in C_new} l_i,n^new
- l_i,c^old

margin_intrusion=0.25
```

逐旧类侵入风险：

```text
I_c =
mean_{i:y_i=c} softplus(u_i,c)
```

平滑worst-old-class保护：

```text
L_intrusion_floor =
beta_floor * logsumexp(I_c/beta_floor)
```

这使`20-19`、`6-15`或其它弱旧类不会被总体均值掩盖。

### 5.5Stage2-C完整目标

```text
L_C =
L_new
+ lambda_floor_C * L_intrusion_floor
+ lambda_residual * mean_n norm(delta_n,2)^2
+ lambda_sep * L_sep

lambda_floor_C=2.00
lambda_residual=0.05
lambda_sep=0.10
```

训练后每个`delta_n`投影回半径`rho=0.15`的L2球。新类bias范围限制为：

```text
-2.0<=b_n<=2.0
```

### 5.6支持集低margin尾部安全offset

训练完成后，只使用旧类support计算真实旧类对最强新类的margin：

```text
a_i,c =
l_i,c^old - max_n(l_i,n^new without g)
```

对每个旧类计算10%分位数：

```text
q_c=quantile_0.10({a_i,c:y_i=c})
```

共享new安全offset为：

```text
g=max(0,margin_guard-min_c q_c)
margin_guard=0.10
```

最终对所有新类logit统一减去`g`。`g`是单一support-only标量，不依赖query真实角色、query类别数量或类别quota。它保护每个旧类的低margin尾部，同时避免D3按单个最小support margin强制零侵入导致的过度新类抑制。

## 6.固定优化配置

D4a首版在任何结果产生前锁定：

|项目|固定值|
|---|---:|
|optimizer|AdamW|
|learning rate|0.01|
|weight decay|0.002|
|batch size|32|
|gradient clip|5.0|
|Stage2-B epoch|20|
|Stage2-C epoch|20|
|temperature|18|
|feature noise/view sigma|0.01|
|old margin|0.50|
|intrusion margin|0.25|
|guard margin|0.10|
|new residual radius|0.15|
|`lambda_floor_B`|0.50|
|`lambda_floor_C`|2.00|
|`lambda_scale`|0.05|
|`lambda_residual`|0.05|
|`lambda_sep`|0.10|
|query representation|base view only|

new5/new10/new20必须复用同一配置，不得分别调epoch、损失权重、margin、offset或残差半径。K5/K1/K20不得用于修改上述配置。

## 7.资源预算

### 7.1可训练参数

Stage2-B：

```text
288 shared diag scale + 6 old bias = 294
```

Stage2-C：

```text
C_new * (288 residual + 1 bias)
```

|新类数|Stage2-C可训练参数|
|---:|---:|
|5|1,445|
|10|2,890|
|20|5,780|

每阶段及合计均远低于80,000参数硬上限。

### 7.2持久化状态

new20 FP32核心状态上界：

```text
diag scale: 288
old prototypes: 6*288
old bias: 6
new weights/residual state: 20*288
new bias: 20
guard offset: 1
```

裸数组约31KB；加入registry、NPZ header、receipt和必要metadata后，预估状态低于64KB。最终审计必须以实际序列化状态文件大小为准，并强制：

```text
persistent_state_bytes<=256KB
```

不得持久化Adam optimizer state。

### 7.3MAC与前向

new20全注册26类的head MAC/query约为：

```text
feature scale: 288
cosine head: 26*288
total: 7,776 MAC/query
```

query固定一次backbone前向、一次FFT96和一次RF32提取，不构建query-query图。support的三个feature view复用同一次backbone/FFT结果，不增加backbone前向。

K10/new20单scenario粗略适配MAC：

```text
Stage2-B约3.1M head MAC
Stage2-C约100M head MAC
```

正式receipt必须分别报告：

- backbone MAC、FFT/RF MAC和head MAC；
- enrollment总前向数；
- singleton query p50/p95时延；
- 平均与P95 backbone forward count；
- 峰值CUDA显存；
- 实际optimizer update数。

硬资源门：

```text
trainable_parameters<=80,000
adaptation_epochs<=30 per stage
persistent_state_bytes<=256KB
dense_query_graph_bytes=0
```

## 8.结果产生前的开发顺序

### 8.1先修复数据协议，不复用旧D1/D3性能包

必须先生成并验证符合2026-07-17单观测协议的development package：

- 每个物理sample仅一个LEO状态和一次overlay provenance；
- 三场景support物理ID集合两两互斥；
- 三场景query物理ID集合两两互斥；
- K10由每类10个独立物理support组成；
- new5/new10/new20使用预登记真实嵌套TX集合；
- clean/source及其派生信号物理不可达。

随后在完全matched row上重跑：

1.直接ADV3B02；
2.identity-only单qKNN；
3.ProtoNet CDA；
4.D4a。

旧D1/D3数值不得进入当前协议的正式比较。

### 8.2K10统一开发顺序

只使用预登记development receiver和development seed：

1.K10/new5；
2.K10/new10；
3.K10/new20。

每个新类规模必须同时生成Stage2-B before和Stage2-C after的不可变prediction，并由隔离scorer计算：

- `old_acc_before_increment`；
- Stage2-B `min_old_class_acc`；
- Stage2-C `old_acc`；
- Stage2-C `min_old_class_acc`；
- `seen_new_acc`；
- `H_old_new`；
- `average_forgetting`；
- old→new、new→old、new→new；
- 每个旧类的support/query margin与侵入；
- 资源Pareto。

### 8.3D4a绝对推进门

三个K10规模必须同时通过：

|指标|new5|new10|new20|
|---|---:|---:|---:|
|Stage2-B old|≥92%|≥92%|≥92%|
|Stage2-B old floor|≥88%|≥88%|≥88%|
|Stage2-C old|≥92%|≥92%|≥92%|
|Stage2-C old floor|≥88%|≥88%|≥88%|
|seen-new|≥92%|≥90%|≥86%|
|H|≥92.00%|≥90.99%|≥88.89%|
|average forgetting|≤0.5pp|≤0.5pp|≤0.5pp|

另外：

- Stage2-C任何旧类query accuracy不得低于同rowStage2-B对应类超过3pp；
- support审计必须逐旧类报告10% margin分位数、最坏margin和old→new侵入数；
- 任何总体old改善但floor下降的规模直接失败；
- 任何新类达标但旧类或floor未达标的规模直接失败；
- 三个规模不得跨row拼接最好结果。

### 8.4后续压力与确认

只有D4a通过全部K10推进门后，才锁定candidate与全部超参数，依次运行：

1.matched K5；
2.K1压力；
3.K20遗忘锚点；
4.5个target receiver×至少5个独立确认seed×3个物理样本互斥LEO场景×new5/10/20正式确认。

K5/K1/K20和confirmation query只能用于验证，不能重新选择candidate、epoch、margin、loss weight、offset或view策略。

## 9.最小实现落点

本设计未授权代码修改。实现时的最小落点预期为：

|目标|建议文件|
|---|---|
|D4a状态、LOO原型、floor损失、新类残差、guard offset|`code/cvsrffi/stage2_diag_cosine_exploration.py`|
|standalone candidate/父状态入口|`code/scripts/run_cvs_stage2_diag_cosine_exploration.py`|
|Stage2-B COMMIT到Stage2-C父状态绑定|`code/scripts/run_cvs_somph_diag_row_pipeline.py`|
|单物理样本单LEO字段、overlay唯一性和跨场景物理ID互斥|现有package builder/pre-open validator|
|机制与协议回归|`tests/test_stage2_diag_cosine_exploration.py`及对应pipeline/package测试|

实现必须继续沿用COMMIT→execution receipt→opened member SHA的严格绑定，并新增以下断言：

- Stage2-C旧类状态bitwise冻结；
- 固定接收IQ的计算view不增加K；
- 同一物理ID不能对应多个LEO状态；
- matched三场景物理ID集合互斥；
- D4a公开API不存在query训练参数；
- query逐样本对全部注册类argmax；
- 参数、epoch、状态、MAC和实际optimizer step资源审计。

## 10.预注册结论

D4a的最小科学假设是：

> 在每个scenario原子row只有K个独立LEO_weak接收IQ的条件下，使用class-balanced LOO旧类原型、显式worst-old-class margin、Stage2-C旧状态冻结、受限新类原型残差和逐旧类低margin尾部guard，可以比自由重训全部head更稳定地保持旧类overall与floor，同时完成多新类注册。

该假设必须由新单观测合法数据上的K10/new5/10/20真实结果验证。在代码、不可变prediction、隔离score和资源receipt完成之前，D4a保持`DESIGN_ONLY_NOT_IMPLEMENTED_NOT_RUN`，不得声称性能改善、部署成功或满足正式门槛。
