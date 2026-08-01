# D111-LOO-GAT轻量域锚运输方法

状态：`DESIGN_FROZEN / BUNDLE_IMPLEMENTATION_ONLY / NO_PERFORMANCE_RESULT`

## 1.出发点

D110-SCPM在未开封source-held四臂中明显退化：K1登记的`H_old_new`下降2.7957pp，old/new各下降1.5873pp，old floor下降10.0529pp，42行中30行H下降。这否定了“Phase1方差小的坐标应逆方差放大”这一代理，但不否定利用合法Phase1聚合知识降低K1不确定性。

D111不重加权support/query的共同坐标，而是用其他旧类support估计共享低秩域位移，只运输Phase1旧类源锚。这使位移保留在query到transported anchor的距离中，不会因support/query共同平移而代数抵消。

## 2.联合封存资产

运行时只接受一个与checkpoint、class registry、method lock共同外层签名的`D111_LOO_GAT_ASSET`。数值成员只能是INT8值与固定FP16尺度：

```text
g_q, g_scale
U_q, U_scale
v_g_q, v_g_scale
v_s_q, v_s_scale
B_q, B_scale
epsilon_q, epsilon_scale
class_registry, schema
```

还必须在manifest中绑定checkpoint SHA、method-lock SHA、registry SHA、生成代码／配置SHA、来源Phase1 aggregate SHA、outer signature和formal状态。禁止clean/raw IQ、source单样本feature/logit/cache、physical/sample ID、成员归属、样本数、源路径、receiver/day名称、逐成员radius、FP32源中心、dense source bank导出、query/truth字段和未签名sidecar。

现有D106 wire只有共享rank-3几何，缺少类锚；现有v2组件为`PENDING_OUTER_JOINT_SEAL`，其逐类basis也不能直接当作共同子空间。因此必须机械生成新joint-sealed bundle；这只改变`bundle_id`，不改变Phase2 received-IQ、physical ID、split或`p2_min_v1`，不触发数据重验。

## 3.共同子空间

对每个旧类的聚合rank-3 basis \(B_c\)，定义

\[
P_{\rm cons}=\frac1{C_o}\sum_{c=1}^{C_o}B_c^\top B_c,
\qquad U=\operatorname{Top3}_{\rm canonical}(P_{\rm cons}).
\]

\(U\)对旧类等权、对类置换不变。只有当量化后谱隙满足

\[
\lambda_3(P_{\rm cons})-\lambda_4(P_{\rm cons})>2\delta_q
\]

且解码后\(U^\top U\approx I\)时才生成formal bundle。失败时拒绝生成，不得以逐类basis、D106 sidecar或新rank替代。

## 4.LOO域锚运输

对旧类\(c\)，仅用当前合法support构建

\[
m_c=N\!\left(K^{-1}\sum_kx_{ck}\right),\qquad
r_c=U^\top(m_c-g_c).
\]

用其余\(C_o-1\)个旧类残差的固定32步阻尼Weiszfeld几何中位数得到\(\hat h_{-c}\)，再运输源锚：

\[
a_c=N(g_c+U\hat h_{-c}).
\]

32步只产生候选；资格\(\chi_c=1\)还要求：全局子空间证书通过，数值可复现的primal-dual gap满足\(F_c(h)-D_c\le\epsilon_F\)，且至少3个余下残差落在同一预封存\(B\)包络内。仅KKT残差不足以通过。

\(B\)是Phase1 aggregate LOO残差的固定conformal order statistic加INT8圆整界，不读held accuracy、receiver表现或query。对五点中至少三个真inlier满足\(\lVert r_j-h\rVert\le B\)的条件情形，有

\[
\lVert\hat h_{-c}-h\rVert\le 6B+\epsilon_F.
\]

这只是条件误差界，不是性能保证。资格、证书、谱隙或有限性任一失败时\(\rho_c=0\)。

## 5.同尺度单位质量评分

M0的identity距离、完整Student-t密度常数和每类带宽\(h_c\)保持不变。

\[
L_c^{\rm sup}(q)=\operatorname{LSE}_k\ell_c(q,x_{ck})-\log K.
\]

新类恒取\(\rho_c=0\)，故\(L_c=L_c^{\rm sup}\)。合格旧类只允许

\[
L_c(q)=\operatorname{logaddexp}\!\left(
\log(1-\rho_c)+L_c^{\rm sup}(q),
\log\rho_c+\ell_c(q,a_c)
\right).
\]

support与anchor共用同一\(h_c\)，全部类先验均为\(1/C\)。禁止\(K+1\)裸平均、额外logit、裸max、old bias或提高旧类总质量。

\[
v_{t,c}=\begin{cases}v_s,&K=1,\\
\max\{v_s/K,S_c^2/K\},&K>1,
\end{cases}
\]

\[
v_{a,c}=v_{g,c}+(6B+\epsilon_F)^2/160,quad
e_c=\lVert m_c-a_c\rVert^2/160,
\]

\[
\rho_c=\chi_c\frac{v_{t,c}}{v_{t,c}+v_{a,c}+e_c}.
\]

合格时\(0<\rho_c<1\)，否则严格回退M0。不存在target cap、强度扫描或基于receiver的选择。

## 6.不变性声明边界

对任意\(R^\top R=I\)的联合坐标正交重参数化

\[
(g,m,U,q)\mapsto(Rg,Rm,RU,Rq)
\]

解码后的\(r_c,\hat h_{-c},\rho_c\)和实数分数不变。不声称INT8 bit级旋转不变。对物理模型\(m_c=g_c+Uh+\varepsilon_c\)，只声称共享低秩位移下的局部条件误差有界；span\((U)\)外位移、类特异畸变、多个旧类异常或锚映射失配是明确失败边界。

## 7.K1可识别性与验证边界

K1时\(a_c\)不由本类唯一support复制，因而只具有泛型非恒等性，不承诺argmax必变。实现阶段仅授权bundle，不授权G0/G1。bundle闭合后，若另行批准G0，只在真实588条tap上检查K1/K5/K10的资格、anchor、score/margin和`argmax_changed_count>0`，不读accuracy。任一K为0即拒绝revision，不扫描\(\rho\)、rank或包络。

## 8.资源

- bundle生成：固定\(C_o\)的聚合线性代数，无训练。
- enrollment：\(O(C_oKd+C_o^2r)\)；几何中位数上限32步。
- 每query：现有\(O(CKd)\)加\(O(C_od)\)的旧锚密度；无跨query状态、无Q的二次／三次项。
- 新增部署态：每旧类一个INT8 anchor、一个FP16权重及小型联合封存bundle，目标约1KiB级；以实现收据为准。

## 9.当前裁决

- 独立监督：理论设计`P0=0 / P1=0 / P2=0 / DESIGN_FROZEN`。
- 当前发布：`P0_BLOCKED=MISSING_FORMAL_D111_JOINT_SEALED_ASSET`。
- 当前唯一授权动作：机械实现和验证joint-sealed bundle。
- 未授权：分数core、G0、G1、N607、Target25、参数扫描。
