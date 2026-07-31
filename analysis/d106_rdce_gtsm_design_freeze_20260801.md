# D106-RDCE/GTSM-r3域适应设计冻结

状态：`DESIGN_FROZEN / IMPLEMENTATION_NOT_STARTED / SOURCE_HELD_NOT_OPENED / TARGET_NOT_OPENED`

## 1.结论

D106的域适应主候选冻结为`D106-RDCE/GTSM-r3-SCATTER02`。它不使用`z_dom→z_id`公共平移，而是从Phase1合法`L_s`学习`z_id`中的共享receiver-day类中心残差方向，形成对全部注册类一致的低秩非等距SPD度量。

冻结原因：

- 公共cross-cov平移在真实`L_s`训练面机械探针中K5/K10正确数不变，K1减少3个正确样本，符合G0对“共同平移无决策作用”的证伪条件；
- RDCE共同衰减能改变归一化后的角度、邻居和margin，不会被共同平移、正交变换或全局尺度抵消；
- K1使用Phase1预锁非零衰减，不估计不存在的单类方差，也不允许identity/fallback通过；
- 运行时不读取`z_dom`、source样本、ground anchor、query truth、old/new query角色、配额或跨query状态。

`z_dom→z_id`cross-cov只保留为被拒绝的机制参考，不作为rank-matched对照。唯一rank-matched因果对照为相同rank、量化和衰减规则的class内receiver-day cell-shuffle basis，仅在source-held使用。

## 2.合法数据面

D106只使用冻结D104 split：

|角色|行数|TX标签|用途|
|---|---:|---|---|
|`L_s`|588|可见|构造RDCE资产和train-only选择|
|`U_s`|5292|隐藏|只用于闭合`rho_label`，本候选不读其TX标签|
|source validation|2520|scorer-only|独立source-held预测后首次打开truth|

`rho_label=588/(588+5292)=0.1`。`L_s`每个TX×receiver×day cell有2–4个互异物理样本，6个TX×7个receiver×4天共168个cell；每个receiver×TX四天合计14条。

输入`L_s`工件：

- split ID：`d104_source_seed104713_v2`；
- `L_s/features.npz`SHA256：`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`；
- checkpoint SHA256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；
- source validation尚未运行正式prediction矩阵，也未计算性能。

不得复用D105的8400行`source_validation`strict tap。D106需要builder-only全池metadata、冻结D104 split和588个`L_s`physical ID做精确inner join，join完成后才允许TX标签和received IQ进入same-IQ dual forward。

## 3.Phase1数学构造

令`d=(receiver,day)`，`c`为source TX，`z_i=norm(ReLU(pre_relu_i))∈R^160`。对每个合法cell：

```text
m[d,c] = mean_i z_i
mu[c]  = mean_d m[d,c]
delta[d,c] = m[d,c] - mu[c]
g[d] = mean_c delta[d,c]
```

对28个receiver-day单元中心化后：

```text
G[d] = g[d] - mean_d g[d]
S_G  = G^T G / 28
```

取`S_G`前三个特征方向，按特征值降序、固定重根canonicalization和最大绝对坐标为正规则形成`B∈R^(3×160)`。rank固定为3，不能由source-held或Target修改。

每个方向的source参考scatter为：

```text
tau[j] = mean_c Var_i((B z_i)[j] | class=c, i∈L_s)
```

部署payload只允许：

- `basis_qint8[3,160]`及逐行FP16量化尺度；
- `tau_qint8[3]`及FP16量化尺度；
- `spectrum_qint8[3]`及FP16量化尺度；
- checkpoint、runtime、method lock、split、tap、构造代码和content root绑定。

不得持久化FP32 basis、source feature、cell均值、TX/class/receiver/day名字、physical/observation ID、raw/clean/received IQ或可替换sidecar。所有source-held门必须使用INT8解码后的basis与scatter重跑。

## 4.Phase2 support-only适应

对当前row全部注册类使用同一公式。令`K`为每类support数：

```text
a0(K) = min(0.95, 1.5*K/(K+4))
```

K1严格使用：

```text
a[j] = 0.3
```

K≥2时，对每类support在`B`方向计算无偏类内scatter，再对全部注册类等权：

```text
e[j] = mean_c Var_i((B z[c,i])[j])
a[j] = clip(
  a0(K) + 0.2*tanh(log((e[j]+1e-8)/(tau[j]+1e-8))),
  0.05,
  0.95
)
```

不使用old/new角色加权。共享SPD度量与等价特征为：

```text
M_S = I - B^T diag(a) B
phi_S(z) = norm(M_S^(1/2) norm(z))
```

INT8解码后必须验证`B B^T`接近单位阵、`M_S`最小特征值不低于0.05。`M_DA`与`M_JOINT`逐字复用同一`M_S`；query只读`phi_S(z_id)`，不得更新`a`、support bank、温度、bandwidth、head或任何选择状态。

## 5.train-only机械探针

探针只读取588条`L_s`，按held day构造内层折，使用简化归一化centroid-qKNN。它用于机制和公式选择，不是source-held、Target或可晋级性能。

|机制|K1净正确|K5净正确|K10净正确|结论|
|---|---:|---:|---:|---|
|cross-cov公共平移rank1–5|−3|0|0|G0拒绝倾向|
|RDCE-r3，`γ=0`|+4|+2|+2|有功能|
|RDCE-r3，`γ=0.05`|+4|+3|+2|有功能|
|RDCE-r3，`γ=0.10`|+4|+3|+2|有功能|
|RDCE-r3，`γ=0.20`|+4|+4|+2|冻结|

绝对正确数：

|K|旧表示|RDCE-r3-SCATTER02|总数|
|---:|---:|---:|---:|
|1|486|490|588|
|5|509|513|588|
|10|509|511|588|

完整cell effect中，RDCE前5方向能量占比约78.98%；cross-cov operator前5奇异方向能量占比约80.93%。这些数值只证明低秩结构与简化头上的可观测作用，不能替代正式四臂source-held。

## 6.数据与实现P1

正式release前必须关闭：

1.实现D106专用`L_s` IQ join与strict tap；现有D104 `L_s` archive本身没有received IQ；
2.新tap保留`day/scenario/observation`绑定；现有D105 tap缺失这些字段；
3.独立validator重算train/held physical ID交集为0并生成ID-only收据；不能只信两个集合root；
4.forward前同时强制D104 exact split、588/5292/2520、`rho_label=0.1`，并证明U_s/source-validation TX标签未进入方法面；
5.实现INT8 RDCE资产、无wire科学拒绝收据、严格loader、runtime handle与共同封印；
6.实现固定四臂source-held预测、truth-side scorer和cell-shuffle对照。

任一协议/lineage错误为`INVALID / NO_PERFORMANCE_RESULT`。输入合法但rank、SPD、INT8、TX、receiver/class held或四臂因果门失败为`REJECT_SOURCE_HELD / NO_DEPLOYABLE_WIRE / NO_TARGET_RESULT`。

## 7.资源上界

rank3 payload的主要数值成员约480B INT8 basis，加scatter、spectrum和量化尺度后仍低于1KiB。每个support/query向量的basis投影与回写约`2×160×3=960MAC`，另加归一化；不训练神经网络，不产生query batch优化，不持久化GPU状态。

最终资源仍须由正式实现receipt重算，不能用本设计估算替代release证据。
