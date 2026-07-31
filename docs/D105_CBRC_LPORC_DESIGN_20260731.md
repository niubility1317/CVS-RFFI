# D105-CBRC+LPO-RC功能设计冻结

状态：`DESIGN_FROZEN_FOR_IMPLEMENTATION`

独立审查：DA`P0=0/P1=0/P2=0`；HEAD`P0=0/P1=0/P2=0`；联合候选`P0=0/P1=0/P2=0`。

证据边界：本文件只授权本地实现和G0验证，不是release GO，不包含source-held或Target性能结论。

## 1.四臂与权限

|臂|坐标|分类器|
|---|---|---|
|`M0`|base z_id|base Student-t qKNN|
|`M_DA`|同一D105 state产生的canonical z_id|base Student-t qKNN|
|`M_HEAD`|base z_id|LPO-RC-qKNN|
|`M_JOINT`|与`M_DA`字节及hash相同的D105 state|与`M_HEAD`逐字相同的LPO-RC代码和配置|

DA可读取封存Phase1聚合bundle、当前row合法support及registry lifecycle；HEAD只能读取当前臂的target support、注册表、qKNN lock和metric。HEAD不得接收ground、bundle、z_dom、source、old/new role、query truth、quota或全局分配信息。

## 2.D105-CBRC-MB4共享DA

canonical变换：

```text
T_a(v)=normalize(ReLU(v+B a_dep))
```

`v`为checkpoint的pre-ReLU z_id；`B∈R^(160×4)`来自不可变Phase1 bundle。对support`i`：

```text
u_i=normalize(U zdom_i)
w_ij=softmax_j(u_i·g_j/T)
m_i=Σ_j w_ij t_j
c_i=Σ_j w_ij exp{−(1−u_i·g_j)/sigma_j²}
P_i=c_i Σ_j w_ij p_j
```

K1不估计单类scatter：

```text
Lambda_c=lambda0
b_c=lambda0⊙m_i
```

K5/K10：

```text
Lambda_c=mean_(i∈c) P_i
b_c=mean_(i∈c)(P_i⊙m_i)
mu_c=Lambda_c^(-1)b_c
```

任务权重：

```text
Stage2-B: alpha_c=1/C_old
Stage2-C old: alpha_c=1/(2C_old)
Stage2-C new: alpha_c=1/(2C_new)
```

固定4轮Huber-IRLS。每轮的`q_c`先在old/new组内归一，再分别乘0.5；若某组`q`数值退化，回到该组uniform，不能丢弃整组。Stage2-B只在old组归一。初值和迭代为：

```text
a0=Proj_A[(lambda0+Σ alpha_c Lambda_c)^(-1)Σ alpha_c b_c]
r_c=||Lambda_c^(1/2)(a−mu_c)||
q_c=min(1,kappa/(r_c+eps))
a_next=Proj_A[(lambda0+Σ alpha_c q_c Lambda_c)^(-1)Σ alpha_c q_c b_c]
```

`kappa`、leave-one-class-out稳定度和最终聚合均保持old/new任务各0.5，不能由new5/10/20的类别数主导。对每类重算`a_-c`：

```text
s_c=clip(1−||B a−B a_-c||/(||B a||+||B a_-c||+eps),0,1)
rho_B=median_old(s_c)
rho_C=0.5 median_old(s_c)+0.5 median_new(s_c)
a_dep=round_fp16(rho·gamma_ground·a)
```

首版`ground_old_multiprototype_enabled=false`，故`gamma_ground=1`。低rank、低coverage、`rho≈0`或FP16后恰为零均发布完整prediction并进入性能统计；只有schema/content root/receipt/method lock不符、非有限或代码异常才属于technical failure。

运行时只验证bundle最小句柄：schema、bundle ID、content root、checkpoint/runtime/method-lock digest和封存receipt root。`bank_t`统一称为`Phase1 coefficient-target codebank`，并锁定`target_rows=0`。

## 3.LPO-RC-qKNN纯HEAD

距离与base log-kernel逐字复用：

```text
D_M(u,v)=max(2(1−cos_M(u,v)),0)
ell_nu(d;h)=−gamma·p_eff·log(h)
             −0.5(ν+p_eff)·log1p(d/(νh²))
S_c(q)=logsumexp_i ell_nu(D_M(z_q,z_ci);h_c)−logK
```

LPO-RC不改变现有INT8 support bank、Student-t核或部署`class_scales`。K≥2时仅临时执行physical-LOO：

```text
e_c=mean_i D_M(z_ci,dequant(INT8(z_ci)))
u_c=(h_c²+e_c)/(K−1)
m_ci=S_c^(-i)(z_ci)−logmeanexp_(d≠c) S_d(z_ci)
delta_c=mean_i(m_ci)
s_c=sqrt(mean_i(m_ci−delta_c)²)

delta_bar=mean_c delta_c
s_bar=median_c s_c
u_bar=mean_c u_c
r_c=[u_bar/(u_bar+u_c+eps)]·tanh[(delta_bar−delta_c)/(s_bar+eps)]
b_c=[(K−1)/K]·s_bar·(r_c−mean_d r_d)
L_c(q)=S_c(q)+b_c
```

若`s_bar≤eps`，全部`b_c=0`。K1固定`h_c=h0,b_c=0`，必须逐值满足`M_HEAD=M0`和`M_JOINT=M_DA`的logit/prediction hash闭合。HEAD对K1只负责严格非劣，K1整体增益只能来自DA。

部署state只保存query实际需要的FP16`h,b`；`u,delta,s,e`只写receipt。保留合法INT8 support vectors，不保留raw IQ或FP32 support sidecar。receipt至少包含：

```text
int8_support_vectors_retained=true
raw_iq_retained=false
fp32_support_vector_retained=false
query_update_count=0
query_extra_dot_product_MAC=0
query_bias_add_ops=C
```

LOO fit MAC、base kernel scalar ops、`B@a`、wire和临时空间均由实现精确重算，不能以设计估算替代实测receipt。

## 4.G0与实现分工

WP-DA只拥有：

- `code/cvsrffi/stage2_d105_cbrc.py`
- `tests/test_stage2_d105_cbrc.py`

WP-HEAD只拥有：

- `code/cvsrffi/stage2_lpo_rc_qknn.py`
- `tests/test_stage2_lpo_rc_qknn.py`

主agent在两个模块独立通过后拥有四臂集成、runner、receipt汇总和最终复审。G0至少验证：K1恒等、physical-ID去重和LOO自排除、标签与lifecycle role同步置换、query顺序/分块不变、query零更新、support/query物理ID不重叠、bundle/root篡改拒绝、INT8边界、共同平移/正交零效应、真实`ReLU+normalize`非等距可观测作用及资源闭合。

通过G0后才能进入未打开source-held G1；通过G1后才能进入单seed Target25 G2；同method lock的一个fresh confirm seed25构成G3。每个Target25必须闭合`25 jobs×3 scenes×4 arms=300`个scenario-arm prediction/score。
