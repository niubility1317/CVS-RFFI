# D122-RDCE×静态ground head轻型组合设计冻结

## 0.状态与裁决

- 状态：`DESIGN_FROZEN / IMPLEMENTATION_AUTHORIZED / NO_NEW_PERFORMANCE_RESULT`。
- 裁决：`MERGE_D122_DESIGN`。只授权一个独立实现波次、本地真实21包无truth predict smoke和一次63行四臂source-held G1；不发布独立588行G0，不调参，不补seed，不运行Target125。
- 域适应因素逐字复用D106的`RDCE-r3-SCATTER02`；分类头因素复用D112已出现正收益的静态单位质量ground head。D122只解决两个已冻结因素在同一坐标中的数学组合，不重新研发RDCE或ground head。
- 最近真实性能边界：D106/D121中的RDCE主效应为正；D112静态ground head在source-held G1中为正；D121-LBR头在identity和RDCE背景下均为负并已关闭。这些历史结果只支持选择因素，不是D122性能结果。

## 1.历史去重

|路线|已验证机制|与D122差异|
|---|---|---|
|D106|RDCE＋RCMR-2V头|没有静态ground head|
|D112|raw表示的静态ground head＋SEAM运动|没有RDCE；SEAM独立效应为0|
|D113|BCAT共同平移＋ground head|BCAT不是RDCE，且只完成无truth G0|
|D114|HBPD带宽＋ground head|HBPD性能为负并已关闭|
|D121|RDCE＋LBR局部rival头|LBR不是ground head，且性能为负并已关闭|

因此`RDCE＋D112静态ground head`没有完整同row性能结果，不是改名重跑。

## 2.唯一新增的数学闭合

设RDCE的冻结平方根度量为

\[
A=I-B^\top\operatorname{diag}(1-\sqrt{1-a})B,
\qquad T(x)=\frac{Ax}{\lVert Ax\rVert_2},
\]

其中`B`是冻结rank-3正交basis，`a`是由D106冻结规则从当前row support得到的衰减；query不参与`a`。

`M_JOINT`必须用同一个`T`变换全部support、每条独立query和六个旧类Phase1 ground anchor。禁止把raw ground anchor与RDCE空间的support/query混合。

D112封存的`σ0_amb`和`v_g_amb`是每坐标ambient MSE标量proxy。D122不新增方向协方差资产，而是明确采用与该标量表示一致的局部各向同性一阶delta-method输运。归一化线性映射在单位向量`x`处的Jacobian为

\[
J_x=\frac{(I-T(x)T(x)^\top)A}{\lVert Ax\rVert_2}.
\]

定义每坐标方差输运倍率

\[
r_A(x)=\frac{\lVert J_x\rVert_F^2}{160}
=\frac{\operatorname{tr}(A^2)-\lVert A T(x)\rVert_2^2}
{160\lVert Ax\rVert_2^2}.
\]

实现不得假设INT8解码后的`B`严格正交，也不得为D122另行正交化。令`D=diag(1-sqrt(1-a))`、`G=BBᵀ`、`t=T(x)`和`u=Bt`，必须按与`M_DA`逐字相同的解码`B/A`计算

\[
\operatorname{tr}(A^2)=160-2\operatorname{tr}(DG)+\operatorname{tr}(DGDG),
\]

\[
\lVert At\rVert_2^2=1-2u^\top Du+u^\top DGD u.
\]

因此该倍率仍可用rank-3低秩路径计算，不需要160×160矩阵，也不读取source行；测试必须同时用同一解码`B`构造dense `A`做数值审计。

对旧类`c`：

1.从raw合法support计算单位prototype`s_c`，并将`T(s_c)`固定命名为“原型输运点”；它不等于且不得在实现中替换为`norm(Σ_kT(x_ck))`；
2.将support和ground anchor变为`T(x)`；
3.在变换后的K个support上重算经验ambient离散度`hatσ'_c`；
4.输运冻结标量：`σ0'_c=r_A(s_c)σ0_c`，`v'_g,c=r_A(g_c)v_g,c`；
5.令`v'_s,c=(σ0'_c+hatσ'_c)/K`，`e'_c=||T(s_c)-T(g_c)||²/160`；
6.冻结质量权重`ρ'_c=v'_s,c/(v'_s,c+v'_g,c+e'_c)`。

`M_JOINT`在RDCE bank的同一Student-t核、同一`h_c/ν/d_eff`和同一logit原点下，将旧类support密度的`ρ'_c`质量转给`T(g_c)`anchor专家。它不是增加第`K+1`票或old bias。新类没有ground anchor，其本类logit必须与`M_DA`逐bit一致。

任一输入非有限、`||Ax||`为零、`r_A(x)`非正/非有限、质量分母非正或receipt不闭合时，该旧类最终分数必须逐bit返回`M_DA`；不得读取query或truth选择fallback。全局资产绑定失败则整个联合臂fail closed，不得静默改用另一资产。

## 3.冻结四臂

|臂|表示|分类头|回答的问题|
|---|---|---|---|
|`M0`|identity|原Student-t qKNN|基线|
|`M_DA`|冻结RDCE|原Student-t qKNN|`DA_AT_BASE`|
|`M_HEAD`|identity|D112静态ground head|`HEAD_AT_ID`|
|`M_JOINT`|同一冻结RDCE|Jacobian输运后的同坐标静态ground head|`HEAD_AT_DA`及交互|

必须报告`M_DA-M0`、`M_HEAD-M0`、`M_JOINT-M_DA`和

\[
(M_{JOINT}-M_{DA})-(M_{HEAD}-M_0).
\]

不得用`M_JOINT-M0`代替DA或head的独立贡献。

## 4.协议与资源边界

- `protocol_schema=p2_min_v1`；K仍是K个独立物理support样本，RDCE/ground view不增加K。
- fit接口只接收冻结Phase1聚合资产、合法support和方法锁；query零fit、零update、零selection。
- 每条query独立对全部注册类决策；禁止truth、old/new role、类配额、全局重排或跨query状态。
- 类置换语义必须保持；跨类最终score出现bit-exact tie时返回`CLASS_SCORE_TIE_UNRESOLVED`，不得按class ID或registry顺序破平局。
- Phase2运行时不读取clean/source样本；只复用与checkpoint共同封存的int8多样本Phase1聚合资产。
- RDCE每个向量约`2×3×160=960MAC`；六个ground anchor每row注册时同样各映射一次。head每query最多增加六个anchor核，约960MAC；无反传、无query状态、无新可训练参数。

## 5.最小验证与止损

不单独发布G0。实现后只做一个本地真实21包无truth smoke，同时覆盖63行×4臂：

- 63行和252个prediction单元完整；
- 同row的`M_DA/M_JOINT`共用RDCE state receipt；
- 六个ground anchor使用同一`T`且Jacobian低秩式与dense审计一致；
- 所有query的最终新类分数满足`M_HEAD[new]=M0[new]`且`M_JOINT[new]=M_DA[new]`逐bit一致；任何旧类解析回退的最终分数也逐bit等于`M_DA`；
- query truth未打开，query fit/update/selection和target access全为0；
- K1/K5/K10至少记录anchor、`ρ'`、logit和资源receipt；不以argmax变化作为发布gate。

通过后直接发布唯一一次63行四臂source-held G1，先封存完整prediction再由独立scorer打开truth。若`HEAD_AT_DA`在old、seen-new、H或floor上形成实质负收益，或没有独立正确数收益，则关闭D122；不调方差倍率、`ρ`、RDCE衰减、核参数，不补G0/seed/125，直接研发下一方法。

本设计冻结只证明公式可实现、协议可审计且资源轻，不构成D122性能收益、Target25达标或论文新颖性声明。
