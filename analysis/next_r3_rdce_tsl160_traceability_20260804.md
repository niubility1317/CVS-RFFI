# NEXT-R3 RDCE×TSL-160设计—实现—验证追踪

状态：`IMPLEMENTING`（科学监督`MERGE`；TSL核心与RDCE四态集成由两个Terra/max agent分文件实现）

## 1. 决策与证据边界

本轮只实现一个联合候选：以已封存的`D106-RDCE/GTSM-r3-SCATTER02`产生`R1`，以`TSL-160`替代D130中已出现held-proxy与floor下降的无约束`D92-Lite160`。D122完成矩阵只给出RDCE的source-held方向性正信号：`DA_AT_BASE`的`ΔH=+0.4447pp`、每行总正确数`+0.9206`、`Δall-floor=+0.4438pp`；它不是Target/125或正式Stage2性能结论。D130的`D92-Lite160`虽将拟合MAC降低99.754%，但未通过留出类与floor不伤害条件，因此关闭原机制，不再调参复活。

## 2. ≤20行可行性摘要

1. `R0/R1`统一为D106的`canonical normalized-ReLU z_id160`，禁止signed-pre-ReLU混用。
2. `R1`复用RDCE rank-3资产；K1固定`a=(0.3,0.3,0.3)`，K5由支持集闭式产生`a`。
3. RDCE资产504B、行状态6B、每query额外960MAC；Phase2零梯度、零query拟合或更新。
4. TSL先验也在同一normalized-ReLU z_id160上构建。
5. 每个receiver-held×class-LOCO资产同时排除held receiver和held class。
6. 先验绑定fold、checkpoint、representation rule、Phase1 physical-ID root和seal。
7. 先验是checkpoint内共同封存的只读INT8多样本聚合知识，不是runtime sidecar。
8. K1的Q/F/L逐logit精确别名qKNN，不宣称head收益。
9. K5的TSL对所有注册类使用同一公式，不读取old/new role。
10. TSL以Phase1物理LOO封存半径限制其相对球形参考头的Frobenius位移。
11. 六臂固定为`R0Q/R0F/R0L/R1Q/R1F/R1L`，共享同row physical IDs与缓存。
12. source-held只报proxy；四态正式命名只用于真正的old/new注册闭环。
13. `REG0`的新类与H为`N/A`，不得记为0。
14. 冻结`S_B`后才追加new support；`DA1_REG1`不得重拟合DA状态。
15. 主比较为`R1Q-R0Q`、`R0L-R0F`、`R1L-R1F`及交互差分。
16. `F160`只是role-structured D92机制的同160维参照，`L-F`不得写成纯head效应。
17. 任一表示绑定、LOCO隔离、量化、无函数或exact top-tie失败均技术关闭。
18. 完成24原子行后立即关闭负候选；不调rho、nu0、rank、表示或矩阵。

## 3. TSL-160数学冻结

K≥5时，`z_ck`为同一R0或R1缓存中的单位向量，`C`为当前注册类数：

```text
mu_c = K^-1 sum_k z_ck
e_ck = ((K-1)/K) [z_ck - (K-1)^-1 sum_(j!=k) z_cj]
v_post,j = [nu0 v0,j + sum_(c,k) e_ck,j^2] / [nu0 + C(K-1)]
v_sph = mean_j(v_post,j)

W_ref,c = mu_c / v_sph
b_ref,c = -||mu_c||^2 / (2 v_sph)
W_hat,c = mu_c / v_post
b_hat,c = -0.5 sum_j mu_cj^2 / v_post,j

D = ||[W_hat-W_ref, b_hat-b_ref]||_F
eta = min(1, rho_h / D)
W = W_ref + eta(W_hat-W_ref)
b = b_ref + eta(b_hat-b_ref)
```

参考头和对角头都先做一次全类中心化，再计算公共位移并施加`eta`；最终沿用共同正二次幂缩放、逐行INT8权重、FP16行尺度与FP16截距的仿射wire。`rho_h`只限制源域物理LOO观察到的函数偏移，不能保证Target floor。

## 4. Phase1先验与封存

每个外层fold只用非held receiver且非held class的Phase1物理cells。每cell在D106同一normalized-ReLU z_id160上计算无偏对角方差；cells等权聚合`log(v0)`，并以`int8[160]+FP16 scale/offset`封存。`nu0`由合法cells的自由度几何均值确定。物理LOO中，以球形参考头的正确pairwise margin求每cell允许的最大公共位移；`rho_h`取有限正半径的固定Type-7 5%分位并向下封存。该构造没有Target输入，也不允许依据24行结果更改。

`v0/rho_h`在R0和R1中都定义为同一160维ambient坐标基上的`pre-adaptation source-anchor`正则，不随RDCE传输，也不声称估计R1真实协方差。R1的support/query必须直接消费RDCE输出的signed unit cache，禁止再次ReLU或重新归一化；先验只把其support估计拉向封存source geometry。receipt必须记录`prior_transported_by_rdce=false`和`r1_covariance_claim=false`。该选择牺牲协方差运输解释，换取同缓存因果一致性，并由24行proxy直接反证其有效性。

最小TSL先验数值payload为170B；TSL仿射状态为`164C B`。C=26时联合TSL数值状态为4434B。TSL拟合解析MAC为`4Nd+8d+2Cd`，零稠密`d×d`矩阵、零谱分解、零线性求解；F160为`Nd²+2d³+Cd²`。两者query仿射均为`Cd`MAC，墙钟与峰值工作集必须在同机实测。

## 5. 最小矩阵与因果读取

冻结2个receiver×6个held class×`K1/K5`，共24个原子行；每行封存REG0/REG1两份2表示×3head bundle。receiver预注册为`1-1`和`18-2`：它们在D122既有联合head证据中分别代表明显正向与明显负向的跨接收机异质性，用于快速反证而非正式性能估计；不得看本轮结果更换receiver。K1用于DA功能性与别名检查；K5用于头与联合收益。source-held held class是directional proxy，不得标成正式新类。进入正式注册闭环后，四态必须使用：

|状态|含义|允许指标|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|旧类、old-floor、总正确数；N/H=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|同一old query上的旧类、old-floor、总正确数；N/H=`N/A`|
|`DA0_REG1`|域适应前/新类注册后|旧类、新类、H、全注册floor、总正确数|
|`DA1_REG1`|冻结同一`S_B`后的域适应表示＋新类注册|旧类、新类、H、全注册floor、总正确数|

24行source-held筛选仍使用上述四个状态字段以避免“before/after”歧义，但artifact和表标题必须统一加`SOURCE_HELD_PROXY`语义，且`formal_new_registration_claim=false`；不得把held-class proxy写成正式新类注册。只有共同定义的指标允许差分中的差分。以Q/L的2×2作为主因果量：`R1Q-R0Q`隔离DA，`R0L-R0Q`测量TSL相对qKNN的增量，`[R1L-R1Q]-[R0L-R0Q]`测量DA×TSL交互。`R0L-R0F`与`R1L-R1F`仅是TSL相对role-structured F160的全管线替换差。

## 6. 实现与验证映射

|设计要求|实现位置|必要验证|
|---|---|---|
|TSL fold先验与receipt|待实现`stage2_next_r3_tsl160.py`|held receiver/class双排除、seal/binding、量化roundtrip|
|TSL K1/K5核心|待实现`stage2_next_r3_tsl160.py`|K1逐logit alias；K5类置换等价、无role输入、资源公式|
|RDCE统一表示桥接|待实现`stage2_next_r3_rdce_tsl_runtime.py`|normalized-ReLU逐字节绑定、无signed fallback|
|六臂共享缓存|复用并窄扩展D129 heads/runtime|2×3arm、同row cache/query-root、零query更新|
|24行执行与评分|复用D129 matrix/scorer，新增单候选entry|24/24覆盖、不可覆盖输出、完整same-row表|

## 7. 发布前唯一硬门

只要求：实际Git入口；query零fit/零update/零selection及禁用clean/source/query truth/role/quota/global reassignment的定向负测；真实checkpoint无query smoke；独立P0/P1=`0/0`；不可变run ID/output；本地commit；N607预检与资源记录。重复数据验证、文档润色、额外签名层、Target125或历史D62/D91/SVRN重跑均不阻塞本轮24行。
