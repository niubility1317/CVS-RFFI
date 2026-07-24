# GRB-JP4-CFM-qKNN-D92/r2-sharedK1设计冻结

状态：`DESIGN_FROZEN / RESOURCE_CONTRACT_ERRATUM_REVIEW_P0_0_P1_0_P2_0 / NOT_IMPLEMENTED / NOT_RELEASED`

## 研究问题

现有完整证据表明：D92能在K5/K10改善旧类均值与floor，但轻微损失新类且K1逐值不变；SVRN的重方差归一破坏类几何；ADV3B02的DA分支没有独立增益；SCXMAP虽改变17,580条注册后margin，但K5/K10的argmax完全不变，K1仅产生破坏。因此下一候选必须在真实网络层改变输入相关decision geometry，并在query前仅凭合法support冻结。

候选全名为`Ground-anchored Rank-4 Joint-Projection Cross-Fitted Margin adaptation with qKNN and D92`，简称`GRB-JP4-CFM-qKNN-D92/r2-sharedK1`。

## 可行性摘要

1.只读取target前共同封存的Phase1 INT8旧类多原型、ground rank4变化基及当前row合法target support。
2.ground组件必须覆盖全部旧类，每类不超过3个多物理样本聚合原型；缺失即fail closed。
3.不保存单样本feature、成员ID、source cache、clean IQ或可独立替换sidecar。
4.真实更新为`ΔW=L_g^T diag(a⊙θ)R`，合入checkpoint内`joint_proj.0.weight`。
5.更新位于ReLU前的真实层，输入相关、非正交，不是共同平移、旋转或输出feature后处理。
6.K1仅由6个target-old与ground锚配对估计4个共享系数，固定收缩`g_1=0.2`。
7.K1的新类单例只用于注册，不进入θ、尺度、trust region或候选选择。
8.K5/K10增加严格physical-LOO的全注册类margin监督，被评样本从全部拟合统计量移除。
9.旧类和新类先分别按类等权，再按任务各0.5，弱类由统一margin deficit连续加权。
10.只执行两次固定active-set解；所有权重、量化、trust region和score标定在held前冻结。
11.query只运行冻结后的模型，逐样本面对全部注册类，query-fit=0，无role/quota/global reassignment。
12.主四臂固定为`M0/M92/M_DA/M_DA92`，D62仅作非门控交互诊断。
13.Phase1-held54先于Target25，另有ground-off、TX-permuted和等能随机q4三种伪证。
14.每个K及K×scene/K×pseudo-new都要求真实neighbor/argmax变化和净纠错。
15.所有门同时保护old/new、H或floor、逐类旧类和forgetting，K1另需LOCO稳定。
16.四主臂各自完整state≤256KiB，post-backbone≤262,144 MAC/query，不跨臂抵扣。
17.JP4 update-factor wire≤4096B；ground多原型及其scale、权重、半径、证书和必要metadata另行完整计入每个DA臂的262,144B总state；最大K10/new20新增support计算<65M MAC；合入后query额外0 MAC。
18.held通过后才运行seed713102的5 receiver×5 slice=25 jobs；失败则不访问Target25。

## Phase1冻结知识

对每个旧类`c`，封存`1≤M_c≤3`个INT8压缩ground原型`p_{c,m}`及FP16尺度，每个原型至少聚合2个互不重复的Phase1物理样本。类内按冻结权重归一：

```text
p_bar_c = Normalize(sum_m alpha_c,m * Decode(p_c,m))
```

同时封存：

- `L_g`：由TX中心化后的receiver/day变化构造的4维左变化基；
- `R`：从冻结`joint_proj.0.weight`固定得到的4维右方向；
- ground原型聚合半径、量化证书、receiver/day nested LODO和leave-one-TX-out证据；
- qKNN-CFM的Phase1 margin常数`δ_q/τ_q`；
- 与checkpoint、类registry、method lock共同封存的SHA。

必须设置`ground_old_multiprototype_enabled=true`。组件只覆盖旧类，不进入qKNN bank，不充当额外邻居，不生成额外旧类logit，不在Phase2更新。

### 唯一确定性构造

Phase1对每个旧类`c`和source receiver/day域`d`，先由至少2个物理样本形成ReLU前`joint_proj.0`输出均值`μ_{c,d}`，再以receiver/day等权形成类中心`μ_c`。按“类先等权、类内receiver/day再等权”把`μ_{c,d}-μ_c`的加权行堆叠为float64矩阵`D`。对`D`做canonical SVD，取前4个右奇异向量为`L_g∈R^{4×d_out}`。

冻结`W_0=joint_proj.0.weight`的canonical SVD前4个右奇异向量为`R∈R^{4×d_in}`。canonical规则固定为：

1. 奇异值降序；
2. 相对差≤`1e-12`的退化子空间用标准基投影后按坐标索引递增执行modified Gram-Schmidt；
3. 每个向量最大绝对坐标为正；并列时选择最低坐标索引；
4. float64构造后以INT8 code＋FP16 RNE尺度封存并生成逐数组SHA。

方向能量为：

```text
a_j = sigma_j(D) / sqrt(sum_l=1^4 sigma_l(D)^2)
```

若`rank(D)<4`、任一`a_j`非正/非有限、`R`不足4个有效方向或量化后rank不为4，则bundle构建技术失败。ground原型混合权固定为`ω_{c,m}=1/M_c`，不从target或held结果学习。

qKNN-CFM的Phase1 margin集合`M_P1^q`只包含target访问前、固定receiver-LODO pseudoepisode中与pseudo-support物理ID互斥、且被正确预测的held pseudoquery全类`top1−logsumexp(other)`原始margin。冻结：

```text
tau_q = max(2^-10, 1.4826 * median(|M_P1^q - median(M_P1^q)|))
delta_q = max(0, Q10(M_P1^q / tau_q))
```

`Q10`采用type-7 linear interpolation；两者均以FP16 round-to-nearest-even封存。Target不得重估。D92不构造另一组`δ/τ`，也不另拟合adapter；`M_DA`与`M_DA92`必须共享逐字节相同的量化`θ_q/ΔW`。

## Phase2真实层适配与唯一求解器

令冻结权重为`W_0`，候选更新为：

```text
ΔW(θ) = L_g^T diag(a * θ) R
W(θ) = W_0 + ΔW(θ)
```

其中`θ∈[-1,1]^4`。更新被合入`joint_proj.0.weight`，随后正常经过ReLU及后续网络；support和每条query使用同一冻结`W(θ)`。

对target-old support在当前基点`θ_base`的特征`z_i(θ_base)`和4方向Jacobian`J_i(θ_base)`，每轮优化变量统一定义为增量`u`：

```text
z_i(θ_base + u) ≈ z_i(θ_base) + J_i(θ_base) u
L_ground = mean_class_old mean_i
  ||z_i(θ_base) + J_i(θ_base)u - p_bar_yi||_2^2
```

K5/K10再构造严格physical-LOO margin。对被留出的support`i`，从中心、neighbor bank、D92统计、margin和梯度拟合中同时移除该物理样本：

```text
m_i(θ_base) = s_yi^(-i) - logsumexp_{d != yi} s_d^(-i)
h_i(θ_base) = d(m_i/tau_q) / d θ
L_CFM =
  0.5 * mean_class_old
    [delta_q - m_i(θ_base)/tau_q - h_i(θ_base)^T u]_+^2
  + 0.5 * mean_class_new
    [delta_q - m_i(θ_base)/tau_q - h_i(θ_base)^T u]_+^2
```

qKNN neighbor数、Student-t公式、temperature、support量化和所有tie规则沿用冻结M0，不从本候选结果改变。

### normal equation

把ground线性残差按旧类等权、类内support等权、160维等权聚合；把当前active OOF tuple按old/new各0.5、组内按类与物理样本等权聚合。权重分别记为`w_i/v_i`，显式固定符号：

```text
G = sum_ground w_i J_i^T J_i
b_g = sum_ground w_i J_i^T (p_bar_yi - z_i(theta_base))
C = sum_active_OOF v_i h_i h_i^T
b_c = sum_active_OOF v_i h_i
      (delta_q - m_i(theta_fold_base)/tau_q)
H_N = G + ((K-1)/K) * C
b_N = b_g + ((K-1)/K) * b_c
lambda = max(2^-20, 0.01 * trace(H_N) / 4)
```

若`trace(H_N)≤64*eps64`、任一项非有限、`cond(H_N+λI)>2^24`或float64 solve相对残差>`1e-10`，整row为技术失败，不回退identity、不生成性能结果。

trust region固定为：

```text
r_W = ||W_0||_F / sqrt(160)
Pi(theta):
  theta_box = clip(theta, -1, 1)
  delta = L_g^T diag(a * theta_box) R
  return theta_box * min(1, r_W / max(||delta||_F, 2^-24))
g_K = K / (K + 4)
```

每次solve后先乘固定`g_K`再应用`Π`。不存在line search、early stop、第三次迭代、receiver/scene分支或coverage回退。

### 两次active-set与严格physical-LOO

所有support按`physical_sample_id`字节序排序，并列不存在；hinge deficit严格`>0`才active，等于0为inactive。

第`t`次重线性化统一求解`u^(t+1)`并累加到当前full-state：

```text
u^(t+1) = solve(H_N^(t) + lambda^(t) I, b_N^(t))
theta^(t+1) = Pi(theta^(t) + g_K * u^(t+1)), t in {0,1}
```

full-support ground在full-state`θ^(t)`处线性化；每个OOF tuple在不含held样本的fold-state`θ_t^(-i)`处线性化。不得把第二轮解直接覆盖`θ^(1)`。

第0次：

1. 令`θ^(0)=0`。
2. 对每个物理样本`i`构造唯一fold`F_0^(-i)`；`i`的全部token/view必须从`L_ground`、target中心、qKNN bank、D92 covariance/scale、类与任务统计、score calibration、normal matrix和右端项中完全移除。
3. 仅用`F_0^(-i)`对held`i`前向，形成`(m_i^(0),h_i^(0),class-group)`；held`i`不得回写该fold。
4. 聚合全部OOF tuple和full-support ground方程，解得`u^(1)`和`θ^(1)=Π(θ^(0)+g_K u^(1))`。

第1次：

1. 对每个`i`从第0次normal equation中同时删除`i`的ground贡献和OOF tuple，重算唯一`θ_1^(-i)`；它不得继承full-state中由`i`产生的任何项。
2. 用`θ_1^(-i)`重建`F_1^(-i)`及对应qKNN head，再只对held`i`前向形成`(m_i^(1),h_i^(1),class-group)`。
3. full-support ground在`θ^(1)`处重线性化；聚合第1次全部OOF tuple，唯一解得`u^(2)`和`θ^(2)=Π(θ^(1)+g_K u^(2))`。

最终只使用`θ^(2)`。D92统计虽然不参与θ求解，但必须在每个fold中同步排除`i`，用于验证`M_DA92`的support-proxy闭包。fold feature、neighbor、Jacobian和`θ^(-i)`只存在于fit scratch，均不得持久化。

### INT8量化

先对`θ^(2)`投影。若全零，则`q=[0,0,0,0]`且`scale=0`；否则：

```text
s0 = max(abs(theta^(2))) / 127
q = clip(round_to_nearest_even(theta^(2) / s0), -127, 127)
s_trust = r_W / max(||L_g^T diag(a * q) R||_F, 2^-24)
scale = largest_nonnegative_FP16_not_greater_than(min(s0, s_trust))
theta_q = float32(scale) * q
```

部署只合入`ΔW(θ_q)`。必须重新审计`||ΔW(θ_q)||_F≤r_W`。量化后rank以阈值`1e-6*max_singular_value`计算并仅记录；它不得触发降rank、回退、调参或候选选择，零更新由既有nonidentity与性能门处理。禁止保存FP32 θ、ΔW、ground、Jacobian或fold sidecar。

最终第1次normal equation同时形成加权设计系统`A_N u≈r_N`，满足`H_N=A_N^T A_N`、`b_N=A_N^T r_N`。coverage仅报告`ρ_J=||Proj_col(A_N)r_N||²/||r_N||²`；当`||r_N||=0`时报告`NA`。coverage不得选择rank、loss权重、trust region、identity回退或候选。

## K1共享识别

K1时`(K-1)/K=0`，不构造CFM tuple，不使用新类单例拟合域结构。6个target-old单例与6个ground聚合锚提供`6×160`个残差方程估计4个共享系数，固定`g_1=0.2`。第`c`个LOCO fold必须同时移除旧类`c`的target support项和对应ground配对项，再重算`G/b_g/λ/θ^(-c)`。报告：

- `rank/cond(G+λI)`；
- `cos(θ_-c,θ_full)`；
- `||θ_-c||/||θ_full||`；
- `||ΔW_-c−ΔW_full||_F/max(||ΔW_full||_F,2^-24)`。

held稳定门预注册为全部fold finite、median cosine≥0.75、min cosine>0、最大相对`ΔW`差≤1且`θ_full≠0`。LOCO不得计算被留类的support自邻居预测，也不得把ground原型放入qKNN bank。

冻结后query agreement仅定义为：

```text
A_c = mean_query 1[
  argmax_all_classes S(q; theta^(-c))
  == argmax_all_classes S(q; theta_full)
]
```

该审计逐query独立、无truth、无状态更新，只用于held解释；正式Target query不得据此改θ、回退或选臂。K1的neighbor/argmax与净纠错只在`θ_full`冻结后由独立held scorer计算。

## 分类头与因果臂

|臂|真实层适配|分类头|用途|
|---|---|---|---|
|`M0`|无|原始INT8 Student-t qKNN|主基线|
|`M92`|无|D92旧/新各0.5共享协方差头|头部基线|
|`M_DA`|GRB-JP4-CFM|同一qKNN|域适应主因果臂|
|`M_DA92`|GRB-JP4-CFM|同一D92|域适应＋头部协同臂|
|`M62/M_DA62`|对应无/有|冻结D62|非门控交互诊断|

D92公式、0.5旧类/0.5新类权重及score标定不得从held或Target25结果调整。D62只提供cross-fitted margin思想，不能参与主候选选择。

## Phase1-held54反证

矩阵固定为1个coverage-SHA确定的held receiver×6个pseudo-new×3个`leo_*_weak`场景×K∈{1,5,10}=54行。除四个主臂外，使用固定seed=`60720260724`运行三种伪证：

1. `ground-off`：删除`L_ground`；K1固定`θ=0`，K5/K10只保留同一CFM求解器；
2. `TX-permuted`：按canonical旧类registry循环左移1位绑定ground原型，`L_g/R/a`不变；
3. `equal-energy-random-q4`：用`PCG64(60720260724)`生成`d_out×4`标准Gaussian矩阵，float64 QR后转置为4行，按真实`L_g`相同规则消歧；复用真实`a/R`，因此4个方向能量逐项相同。

三伪证不得重新标定`δ_q/τ_q/λ/r_W`，不得选择第二个permutation/seed，也不得改变qKNN或D92 head。每个伪证产生与真实q4同构的`M_DA`和`M_DA92`，分别只与对应head比较。

主晋级比较`G_DA92=M_DA92−M92`与独立DA比较`G_DA=M_DA−M0`必须分别在每个K、每个K×scene和每个K×pseudo-new独立判定，禁止跨head、跨分层或总体均值补偿。两门均要求：

- 真实neighbor membership变化>0；
- argmax变化>0；
- wrong→correct>correct→wrong；
- old-after、seen-new、H、全注册类floor和min-new均不下降；
- 每个旧类准确率不下降，forgetting不增加；
- H或全注册类floor至少一项严格提高；
- 真q4的净纠错和ΔH均严格大于对应head下三个伪证的最大值；
- K1非identity且LOCO稳定门通过。

只有`G_DA92=PASS`且`G_DA=PASS`才能晋级。任一分层失败即`PHASE1_HELD_PROXY_NEGATIVE`，不创建Target bundle、不访问Target25、不做target结果驱动调参。

标签置换等价测试固定seed=`60720260724`，分别在old组与new组内执行Fisher-Yates置换；同步置换support标签、registry、ground原型、coverage mask及输出列。逆置换后要求量化`θ_q`字节、资源receipt、所有无标签数值状态、全query prediction和全部指标逐值一致。禁止依赖类名、registry原始排序或old/new之外的身份分支。

## seed713102的Target25

仅在held54通过且另行独立审查后运行：

|receiver|固定slice|
|---|---|
|`20-1`、`3-19`、`7-14`、`7-7`、`8-8`|`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`|

共25 jobs，每job覆盖三个物理ID互斥场景，并对同一query发布四主臂和D62非门控诊断。主比较为`M_DA−M0`与`M_DA92−M92`。每个same-row必须包含old-before、old-after、seen-new、H、全注册类floor、逐类旧类准确率、forgetting、neighbor/argmax转移和资源。

## 资源预注册

四个主臂采用独立可部署口径，不允许用跨臂共享抵扣。每个臂的完整persistent state硬门均为262,144B，Stage2 post-backbone总query计算硬门均为262,144 MAC/query：

|资源|上限或估算|
|---|---:|
|JP4 update-factor wire|包含`L_g/R/a/theta`及必要scale/receipt；硬上限4,096B|
|ground wire|包含全部多原型code、scale、权重、半径、证书及必要metadata；无4,096B子门，但必须完整计入每个DA臂总state|
|`M0`|完整qKNN state≤262,144B；post-backbone≤262,144 MAC/query|
|`M92`|完整D92均值、系数、量化与score state≤262,144B；post-backbone≤262,144 MAC/query|
|`M_DA`|完整qKNN＋JP4 state≤262,144B；post-backbone≤262,144 MAC/query|
|`M_DA92`|完整D92＋JP4＋ground摘要state≤262,144B；post-backbone≤262,144 MAC/query|
|qKNN＋JP4预估state|≤163,787B|
|4方向Jacobian，K10/new20|约53.25M MAC|
|LOO pair distance，K10/new20|约10.77M MAC|
|solve/merge|<1M MAC|
|新增support总计算|<65M MAC|
|合入后JP4 query额外计算|0 MAC|
|qKNN头|≤42,466 MAC/query|

`M0`计base qKNN全部state/MAC；`M92`计完整D92状态与score MAC；`M_DA`计base qKNN＋JP4；`M_DA92`计完整D92＋JP4。实现必须分别报告`update_factor_wire_bytes`、`ground_wire_bytes`、`total_component_bytes`和每个DA臂的`full_arm_state_bytes`。ground摘要、JP4 code/scale及head统计必须计入对应DA臂；`M_DA`与`M_DA92`分别独立满足总state硬门，不得共享抵扣。backbone公共MAC单列，不混入post-backbone硬门。任一臂超限即`RESOURCE_FAIL/NO_PROMOTION`，不得通过降rank、删类、coverage回退或Target结果驱动压缩补救。

资源契约勘误：冻结后实现前发现，最坏18个ground原型code本身为2,880B，与`L_g/R`合计已超过4,096B，旧“JP4 wire/payload≤4,096B”表述内部不可满足。上述勘误只把4,096B子门精确定义为update-factor wire；ground仍完整进入每个DA臂262,144B总state硬门，不改变科学方法、数据访问、拟合、四臂、held门或Target门。独立勘误审查=`P0=0/P1=0/P2=0`。

禁止保留FP32 ground、Jacobian、θ或权重差分sidecar；θ只以INT8 code＋FP16 scale持久化，运行前合入冻结权重并生成receipt。

## 禁止重演

- 不继续调SCXMAP beta/rank/gate；
- 不使用共同平移、正交旋转或输出层full-coordinate transport；
- 不使用SVRN式重全局方差归一；
- 不让新类K1单例驱动域适配；
- 不按receiver、scene、TX、类名、query角色或真实query类数分支；
- 不使用query真值、query batch配额、跨query重排或scorer回流；
- 不以nonzero θ、coverage、RMSE、margin变化或技术完成替代净纠错与same-row性能。

## 当前裁决

`REJECT_GRB_R1_TARGET25_DIRECT`。

本r2已完成独立监督逐项复核，最终裁决为`P0=0、P1=0、P2=0 / DESIGN_FROZEN`。下一状态只能是按本文实现并重新做独立代码审查；当前仍不得发布held54或Target25。
