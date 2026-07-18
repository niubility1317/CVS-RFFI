# D35-DSWR稠密安全winner条件注册追踪

## 目标与边界

D34证明稀疏old-new可见性图会在held物理support的old winner漂移时产生固定`-2`截断。D35只改Stage2-C注册决策结构：保留D33-FAST的Stage2-B旧score前缀并逐bit审计，所有新类在每个逐样本决策中始终可见；旧winner只选择support-only安全阈值，不再决定新类是否存在。

本轮仍是receiver`20-1`、seed`713101`、K10、5新类、3个LEO_weak场景的开发support-only screen。query保持关闭；不产生正式性能声明。D35成功也不等于最终目标达成，因为注册前FAST旧类held均值仅82.22%，后续仍须单独增强Stage2-B。

## 数学锁

每个已接收LEO_weak IQ只产生一行拼接向量

`z=[z160,FFT96,RF32]∈R^288, ||z||2=1`。

旧类FAST score为`g_i(z)`，Stage2-C不修改该矩阵。新类`j`从独立physical support得到`M_j∈{1,2}`个原型`p_jm`，只保存symmetric int8中心、FP32 scale和inverse norm。令`i*(z)=argmax_i g_i(z)`，对旧winner`i`、新类`j`、原型`m`，只用旧support构造安全阈值

`b_ijm=max_{x∈O_i}[18 cos(z_x,p_jm)-g_i(z_x)]+epsilon+lambda u_i f_i`，

其中`O_i`是冻结旧头winner为`i`的旧support集合；空集合使用全部旧support保守回退；`u_i`是旧类support不确定度；`f_i=2`仅用于低于旧类median support准确率的floor winner，否则为1。

逐样本新类score为

`s_j(z)=max_m[18 cos(z,p_jm)-b_i*(z),jm]`。

最终score为`[g_1,...,g_C,s_1,...,s_N]`并直接argmax。所有新类始终得到有限score，不存在nonedge、quota、batch重分配或role Oracle。

|arm|每新类原型|阈值buffer|floor保护|
|---|---:|---:|---|
|D35-A|1个mean|`lambda=0`|无|
|D35-B|最多2个确定性cluster/medoid|`lambda=0.25`|无|
|D35-C|最多2个确定性cluster/medoid|`lambda=0.5`|低于median的旧winner×2|

K1只能退化为单个物理support原型，旧LOO标记`NOT_EVALUATED_K1`，不得伪造。K5/K10从本K可达support重建全部原型和阈值。

## 需求到实现追踪

|需求|实现面|验证面|证据面|
|---|---|---|---|
|全注册类逐样本有限score|`stage2_d35_dense_safe_registration.py` scorer|core finite/all-class测试|geometry audit|
|旧score前缀逐bit不变|runner共享FAST prefix+D35拼接|bitwise断言|105行training log|
|旧fit support不退化|`b_ijm`max-residual安全阈值|逐类/floor pre-post相等|resource audit|
|held旧类安全|outer 8-shot fit/2-shot held|15折new intrusion=0硬门|selection|
|新类物理LOO可达|移除held physical support后重建原型|每类margin_min>0|geometry audit|
|09f8/f608与旧floor|逐类矩阵|classwise断言/报告|自动化报告|
|极轻部署|int8原型+小型`b_ijm`表；0梯度|MAC/state/latency测试|resource audit|
|协议|support-only、query=0、无clean/source/Oracle/quota/global/dense graph|candidate lock+runtime guards|support audit/RECEIPT|

## 晋级硬门

1. 全部D35 old prefix逐bit不变，fit旧support逐类与floor不退化。
2. 15个outer held折全部旧类new intrusion为0。
3. 三场景full-K10全部5个新类physical LOO margin_min>0，尤其09f8和f608。
4. old/new/H/forgetting/worst joint floor联合达到B3与D33-FAST阈值，H必须严格提高；禁止单指标最大值晋级。
5. 0 optimizer step、active parameter<=50k、状态<=50kB、无dense query图；5新类最坏query MAC低于B3并报告20新类扩展估算。
6. 任一门失败则自动fallback并`selected_positive_route=false`。

## 预期资源

单原型5新类新增约1,500B；双原型约3,000B，再加`6×5×M`小型阈值表。5新类query上界为FAST旧头2,016MAC加`M×5×288`点积；A约3,456MAC，B/C约4,896MAC。20新类双原型约13,536MAC，仍显著低于identity-only K10单qKNN的41,600MAC，但实测延迟和状态必须以artifact为准。
