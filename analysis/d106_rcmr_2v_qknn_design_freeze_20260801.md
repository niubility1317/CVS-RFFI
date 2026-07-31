# D106-RCMR-2V-qKNN/r1.1分类头设计冻结

状态：`DESIGN_FROZEN / IMPLEMENTATION_AUTHORIZED / HELD_NOT_OPENED / TARGET_NOT_OPENED`

## 1.方法定位

`D106-RCMR-2V-qKNN/r1.1`是D106四臂实验中的纯support-only分类头。它使用同一物理IQ一次冻结前向产生的非负视图`z=ReLU(pre_relu)`和signed视图`pre_relu`，以support内部的跨视图秩一致性估计可靠度，再按query与support的双侧拥挤度直接排名。

该头不读取ground bank、source样本、query truth、role、类配额或batch类计数；不更新模型或support state；不使用旧Student-t分数、残差、温度、阈值、hard gate、identity fallback或待调混合权重。本文冻结实现语义，不构成source-held、Target或性能结论。

## 2.输入与四臂归因

有序注册表为\(\mathcal C=(c_1,\ldots,c_C)\)。每类恰有\(K\)个合法support，\(N=CK\ge2\)。support槽位\(i\)的类别索引为\(y_i\)。

每个物理IQ只前向一次：

\[
z_i=\operatorname{ReLU}(p_i),\qquad p_i=\texttt{pre\_relu}_i,\qquad z_i,p_i\in\mathbb R^{160}.
\]

四臂共享同一HEAD公式，只改变输入映射：

\[
T=
\begin{cases}
I,&M_0,M_{\mathrm{HEAD}},\\
T_{\mathrm{DA}},&M_{\mathrm{DA}},M_{\mathrm{JOINT}}.
\end{cases}
\]

\[
x_i^+=\operatorname{norm}_2(Tz_i),\qquad
x_i^\pm=\operatorname{norm}_2(Tp_i).
\]

query使用同式。\(T_{\mathrm{DA}}\)只能由臂级DA runtime提供；HEAD接口只接收映射后的两视图和DA receipt，不读取RDCE ground资产。任一向量维度不是160、包含非有限值或L2范数为零时fail closed，不退回单视图。

## 3.Formal state与量化

每个view\(v\in\{+,\pm\}\)逐行量化：

\[
s_i^v=\operatorname{fp16}\left(\max_d|x_{i,d}^v|/127\right),
\]

\[
b_{i,d}^v=\operatorname{clip}_{[-127,127]}
\left(\operatorname{round\_ties\_to\_even}(x_{i,d}^v/s_i^v)\right).
\]

\(s_i^v\le0\)、scale非有限或量化后为零向量均拒绝。评分上下文使用：

\[
\bar x_i^v=\operatorname{norm}_2\left(
\operatorname{fp64}(s_i^v)\operatorname{fp64}(b_i^v)\right).
\]

formal state仅允许包含：

- schema、candidate/method-lock、arm DA receipt；
- capsule、split、validator、support physical root和有序registry root绑定；
- \(C,K,N,D=160\)；
- `codes_plus[N,160]int8`、`codes_signed[N,160]int8`；
- 两组`scale[N]fp16`、`class_index[N]uint8`和`R_i[N]fp16`；
- 各数组content root与总state receipt。

禁止持久化raw support、FP32/FP64 support、物理ID明细和任意\(N\times N\)数组。

## 4.确定性距离与mid-rank

同一view内：

\[
d_{ij}^v=\operatorname{clip}_{[0,2]}
\left(1-\sum_{d=1}^{160}\bar x_{i,d}^v\bar x_{j,d}^v\right).
\]

FP64点积按坐标升序归约；NaN/Inf拒绝；零规范为`+0.0`。对长度为\(L\)的FP64数组\(A\)和有限值\(t\)，定义：

\[
\operatorname{MR}(A,t)=
\frac{|\{a\in A:a<t\}|+\frac12|\{a\in A:a=t\}|+\frac12}{L+1}.
\]

等号指规范化后IEEE-754 binary64位相等，不使用epsilon。

## 5.Support可靠度与评分上下文

构建state时，以量化后的\(\bar x\)流式计算：

\[
D_i^v=(d_{ij}^v)_{j\ne i},\qquad
r_{ij}^v=\operatorname{MR}(D_i^v,d_{ij}^v),
\]

\[
R_i^\star=\exp\left[
-\frac1{N-1}\sum_{j\ne i}|r_{ij}^+-r_{ij}^\pm|
\right],\qquad R_i=\operatorname{fp16}(R_i^\star).
\]

pair工作区计算完一行即销毁，state receipt绑定`R_i`的原始FP16字节。评分前`prepare(state)`每个scoring batch只执行一次，生成：

\[
P_i^v=\operatorname{sort}_{\mathrm{asc}}(D_i^v).
\]

ephemeral context只含state receipt、\(N/C/K/D\)、双FP64解码support bank、双FP64排序profile和不可变创建token。它不含query、truth或预测，不得跨row复用。query路径不得重算support-support距离或profile。

## 6.Query拥挤度与分类分数

\[
d_{qi}^v=\operatorname{clip}_{[0,2]}
\left(1-\sum_d x_{q,d}^v\bar x_{i,d}^v\right),
\]

\[
\alpha_{qi}^v=\operatorname{MR}(Q_q^v,d_{qi}^v),\qquad
\beta_{qi}^v=\operatorname{MR}(P_i^v,d_{qi}^v),
\]

其中\(Q_q^v=(d_{q1}^v,\ldots,d_{qN}^v)\)。定义：

\[
m_{\mathrm{plus},qi}=(1-\alpha_{qi}^+)(1-\beta_{qi}^+),
\]

\[
m_{\mathrm{signed},qi}=(1-\alpha_{qi}^\pm)(1-\beta_{qi}^\pm),
\]

\[
R_q=\exp\left[-\frac1N\sum_i|\alpha_{qi}^+-\alpha_{qi}^\pm|\right],
\qquad w_{qi}=R_q\operatorname{fp64}(R_i),
\]

\[
e_{qi}=
\frac{m_{\mathrm{plus},qi}+w_{qi}m_{\mathrm{signed},qi}}{1+w_{qi}},
\qquad
S_c(q)=\frac1K\sum_{i:y_i=c}e_{qi}.
\]

\(S_c\)是唯一分类分数。\(R_q,R_i,w_{qi}>0\)，所以signed视图对每条query严格参与；视图秩不一致只连续减弱其贡献，不触发fallback。

## 7.并列、对称性与协议失败

类内按全局support槽位升序累加。以下任一条件触发fail closed：

- score非有限；
- 任一类support数不等于\(K\)；
- context与state receipt不一致；
- 两个或以上不同类的最大\(S_c\)按binary64 bit-exact相等。

跨类并列错误码固定为`CROSS_CLASS_SCORE_TIE`。禁止按class ID、registry顺序、物理ID或随机数解并列。标签置换只置换\(y_i\)和输出坐标，因此距离、profile、可靠度和等K平均保持不变，预测严格标签置换等价。

## 8.资源冻结

在正式最大规模\(C=26,K=10,N=260,D=160\)下：

|项目|冻结值|
|---|---:|
|双INT8码|83,200B|
|双FP16 scale|1,040B|
|FP16 `R_i`|520B|
|uint8 class index|260B|
|数值主体|85,020B|
|含固定registry/SHA/header的二进制payload|86,060B|
|arrays-only临时峰值|2,285,920B|
|临时数组加固定state resident峰值|2,371,980B|
|一次`prepare` support-support MAC|10,774,400|
|一次`prepare` support重归一化平方累加|83,200|
|一次`prepare`排序comparison-equivalents|约1,079,704|
|单query MAC|83,200|
|单query归一化平方累加|320|
|单query排序comparison-equivalents|约4,172|
|单query profile二分比较上界|4,680|

JSON包装、Python对象和allocator开销不包含在固定二进制payload或arrays-only峰值中，必须由正式资源receipt另行实测。

## 9.实现与放行门

实现包必须交付：

1.独立formal state builder、strict loader、一次性`prepare`context和query scorer；
2.canonical method lock、wire format、content roots和资源receipt；
3.同一IQ双视图、量化round-trip、标签置换等价、K1非identity、无query update、context不跨row、跨类并列拒绝和各类support计数负测；
4.真实特征G0反证：候选必须在预登记的train-only机械探针中至少改变一个query argmax；若与旧qKNN逐query同序，则`REJECT_NO_FUNCTION`，不得靠扫描参数补救；
5.独立复审达到`P0=0/P1=0`后，才可进入固定四臂source-held G1。

source-held G1只用于一次性比较预冻结的`M0/M_DA/M_HEAD/M_JOINT`。任何outer held结果不得回调公式、量化、可靠度、权重、并列语义或资源设计。
