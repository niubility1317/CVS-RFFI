# NEXT-R6 FA-RDCE3×D92-Lite-RPPF160冻结设计

状态：`DESIGN_FROZEN / P0=0 / P1=0 / IMPLEMENTATION_PENDING`

## 1.研究判断

NEXT-R6保留NEXT-R4中唯一具有完整正收益的K5 FA-RDCE3，并用`D92-Lite-RPPF160`替代已关闭的CER、D92-Lite160对角OAS和LOO logit混合。RPPF是`Regularized Prototype Polar Frame`：它直接构造一个全类、类置换等变的原型极分解分类头，不把qKNN分数与另一种概率模型相加，因此不存在Q/L共同温标不可辨识问题。

原型分类在少样本场景中以类均值作为简洁归纳偏置；L2归一化能够降低特征范数波动对原型分类的影响。RPPF进一步对注册类原型的Gram相关性作正则极分解，使高度相关的类方向在分类头中分离。该几何动机与prototype classifier及simplex/ETF分类几何一致，但NEXT-R6只声明本项目内的自研推理头，不宣称未经检索证明的算法首创。

## 2.唯一数学定义

设当前状态的注册类按封存registry排列为`c=1,…,C`，160维support为`z_ck`，`K∈{1,5}`。定义：

```text
u_ck = unit(z_ck)
p_c  = unit(K^-1 Σ_k u_ck)
P    = [p_1,…,p_C] ∈ R^(160×C)
G    = 0.5(PᵀP + (PᵀP)ᵀ)
λ    = tr(G)/C = 1
A    = G + λI_C
A^(-1/2) = V diag(eig(A)^(-1/2)) Vᵀ
U    = P A^(-1/2)
s_c(q) = unit(q)ᵀ U_:c
```

`unit(x)=x/||x||_2`。任一输入、均值、范数、Gram、特征值、`U`或query score非有限，或范数为0，均为技术失败。`A`在实数精确计算中的特征值下界为1；实现固定使用FP64对称`eigh`，不构造160×160协方差，不扫描λ，不读取support正确率、query或truth。

## 3.量化与资源

每个`U`列独立量化为`INT8[160]+FP16 scale`，不保存intercept或FP32 sidecar。scale取不小于`maxabs(U_:c)/127`的最小可表示正FP16值，codes使用round-to-nearest-even；若codes超出`[-127,127]`、scale为0/非有限或反量化列非有限，则技术失败。量化前后logits/prediction一致性只形成receipt，不触发性能回滚或Q fallback。

|项目|RPPF160|formal D92参考|
|---|---:|---:|
|部署数值状态|`162C`B|`1152+590C`B|
|C=6|972B|4,692B|
|C=26|4,212B|16,492B|
|query主点积|`160C`MAC|`288C`MAC表示级参考|
|拟合主项|`KC·160+2C²·160+O(C³)`|包含高维协方差/求解|

因此C=6和C=26时数值状态分别减少79.3%和74.5%，160维点积相对288维点积减少44.4%。实际拟合墙钟、峰值工作集、FP64临时量和量化时延必须由同机resource receipt给出；不得从解析MAC直接声称端到端更快。

## 4.DA与注册状态

- K5：FA-RDCE3公式、rank-3 INT8资产、`rho=sqrt(3)`、Wiener系数及6B FP16动态状态保持不变。FA只由REG0旧类support拟合；`DA1_REG1`逐bit复用`DA1_REG0`的FA状态，新类support不重拟合DA。
- REG0：尚无注册新类，历史D92-Full160的old/new机制没有合法定义。F必须按同representation精确alias Q，即`R0F≡R0Q、R1F≡R1Q`；不得把registry尾类伪造为new角色。
- K1：FA严格旁路。对每个REG状态，`R1Q≡R0Q`、`R1F≡R0F≡Q`、`R1L≡R0L`；RPPF仍由K1单物理support原型功能化，但只拟合一次。receipt必须证明相同cache、RPPF state、logits和prediction，资源不得重复计数。
- K5 RPPF：每个`DA×REG`状态从自身support representation cache独立拟合`P/U`。禁止把R0的`U`用于R1 query；若per-state refit使共同正交/尺度DA效应消失，这是RPPF的正确不变性，而不是实现故障。
- REG1中所有旧类与held-proxy类使用同一公式。不得按old/new角色分支、类别ID、query角色、quota或batch统计改变头部。

## 5.六臂与四状态

每个REG状态执行同一六臂：

|表示|Q|F|L|
|---|---|---|---|
|`R0=DA0`|direct qKNN|历史D92-Full160机制对照|RPPF160|
|`R1=DA1`|FA-RDCE3→qKNN|FA-RDCE3→Full160|FA-RDCE3→RPPF160|

F仅作注册后历史机制对照，不属于NEXT-R6部署候选。REG0的F按同representation严格alias Q；K1的F严格alias Q且FA跨representation旁路。artifact主状态统一为`DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1`；REG0的held-proxy准确率和H_proxy为`N/A`。

主比较：

```text
DA_EFFECT_Q       = R1Q - R0Q
LITE_BASE_REG1    = R0L - R0F
JOINT_REPLACE_REG1= R1L - R1F
DIRECT_UTILITY    = R1L - R1Q
DA_HEAD_INTERACT  = (R1L-R1Q) - (R0L-R0Q)
```

## 6.冻结Proxy24矩阵

receiver按既有D130 plan的`receiver_ids`顺序，在排除NEXT-R4已评分的`1-1、18-2`后取前两个：`1-19、14-7`。这是一条prediction前canonical规则，不依据历史性能选择receiver。held-class固定为`14-10、14-7、20-15、20-19、6-15、8-20`。

```text
2 receiver × 6 held-class × K{1,5}
= 24 matched conditions
每条件2 REG状态 × 6 arms
= 288 state-arm prediction surfaces
```

逻辑surface固定为288。去除强制alias后的唯一prediction为168：12个K5条件各10个unique（REG0四个、REG1六个），12个K1条件各4个unique；其余120个surface必须由alias receipt指回同一cache、state、logits和prediction，不能重复计算。

K1 support是K5逐类物理前缀；support/query物理ID互斥。所有臂共享同一received-IQ、checkpoint、R0/R1 representation cache和query键。该矩阵只输出source-held`A_retained、A_held_proxy、H_proxy、F_retained、总正确数`，不能把held class写成正式Y_new或Target性能。

## 7.结果裁决

完整prediction封存后一次性打开truth并评分。K5的`DA_EFFECT_Q`在REG0只比较`A_retained、F_retained`和正确数，在REG1必须使H_proxy和总正确数严格增加且`A_retained、A_held_proxy、F_retained`不下降。`LITE_BASE_REG1、JOINT_REPLACE_REG1`仅在注册后定义，必须使H_proxy和总正确数严格增加且三个guardrail不下降；不得从REG0的F alias生成虚假的历史D92比较。`DIRECT_UTILITY`在REG1必须使H_proxy与总正确数严格增加且三个guardrail不下降。K1要求注册后RPPF相对同row F/Q提高H_proxy与总正确数，并且retained、held-proxy和floor不下降。

任一主比较失败即记为完整负收益并关闭RPPF160；不调λ、rank、receiver、class、K、量化或矩阵，不重跑。通过只产生source-held方向性联合胜者，随后才允许进入真实功能面、fresh source-held与单seed Target；不得直接写成promotable性能。

## 8.实现级反证测试

1.support行置换不变；类别置换仅置换state列与logit列。
2.共同正交变换、输入共同正尺度与registry同步置换保持预测等变。
3.REG0同representation的F/Q alias及K1跨representation四个alias关系逐logit、state SHA、prediction和资源计数闭合；总计168个unique prediction、120个alias surface。
4.K5的每个DA×REG状态独立拟合RPPF；R0 state不得用于R1 query。
5.query零fit、零update、零selection；无truth/role/quota/global reassignment字段或调用。
6.FP64 eig、FP16 scale、INT8 codes和反量化有限；量化receipt不触发fallback。
7.真实checkpoint forward中RPPF与Q至少产生一个非共同logit变化；若全矩阵无query决策功能，记`REJECT_REVISION_NO_FUNCTION`。

## 9.独立设计复核

两位Terra/max方法作者先后拒绝了FCI160：其类相关项展开后与CER的prototype-shape affine residual同构，且与qKNN没有可辨识共同温标。独立监督随后要求RPPF per-state refit、`λ=1`及K1 alias闭合。纳入唯一必修后，设计级结论为`P0=0、P1=0、DESIGN_FROZEN`；实现仍需独立代码级复核。

## 10.理论边界与参考

RPPF沿用少样本原型分类的均值原型与L2归一化思想，并受到simplex/ETF类间几何的启发：[Prototypical Networks](https://arxiv.org/abs/1703.05175)、[A Closer Look at Prototype Classifier](https://openreview.net/forum?id=U_hOegGGglw)、[Neural Collapse under MSE](https://proceedings.mlr.press/v162/zhou22c.html)。这些工作不证明RPPF在CVS上的收益；唯一性能证据必须来自上述冻结同row矩阵。
