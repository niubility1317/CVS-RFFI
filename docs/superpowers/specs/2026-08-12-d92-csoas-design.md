# E0 FULL CSOAS设计

状态：`DESIGN_FROZEN`

## 1.目标与边界

`E0_FULL_CSOAS`只在`DA1_REG1`且`K>2`时激活。它复用E0的D81 support变换、288维特征、old/new固定任务平衡、equal-prior单FULL solve和D42部署state，只替换FULL内部的类内协方差估计。`DA1_REG0`、K1和K2保持E0/D92 FULL byte-exact alias。

方法只读取同一outer合法K-shot support、标签及封存D81 ground basis。query及其view不参与fit、更新、选择、回退或停止；禁止clean/source、query truth、role Oracle、class quota和跨query重排。

## 2.冻结统计量

对D81变换后的第`c`类support行`x_ci∈R^288`，分类均值保持E0：

`mu_c=K^-1 sum_i x_ci`。

复用D81在同一类同一support上已经产生的归一Cauchy权重`a_ci`，并只为协方差定义独立加权中心：

`xbar_c^w=sum_i a_ci x_ci`，`sum_i a_ci=1`。

加权无偏scatter与有效样本数为：

`S_c=[1-sum_i a_ci^2]^-1 sum_i a_ci(x_ci-xbar_c^w)(x_ci-xbar_c^w)^T`，

`n_eff,c=[sum_i a_ci^2]^-1`。

不得以E0非加权均值代替`xbar_c^w`计算上述scatter；不得把`mu_c`改为加权分类均值。

## 3.逐类闭式OAS

令`p=288`，`tau_c=tr(S_c)/p`，`alpha_c=tr(S_c^2)/p^2=mean(S_c⊙S_c)`。这里的`mean`对完整`p×p`矩阵取平均，严格匹配冻结的sklearn OAS尺度。逐类收缩系数为：

`rho_c=clip[(alpha_c+tau_c^2)/((n_eff,c+1)(alpha_c-tau_c^2/p)),0,1]`。

分母非正时固定`rho_c=1`。逐类协方差为：

`Sigma_tilde_c=(1-rho_c)S_c+rho_c tau_c I`。

该式逐类保trace，禁止再加入`tr(U)/tr(R)`或任何全局尺度重标定。old组和new组分别对类等权平均：

`Sigma_old=mean_c∈old Sigma_tilde_c`，`Sigma_new=mean_c∈new Sigma_tilde_c`，

`Sigma=0.5 Sigma_old+0.5 Sigma_new`。

随后完全沿用E0的equal-prior单FULL solve、class-common居中和一次D42 codec发布。

## 4.对称性、生命周期与资源

所有类、support行使用同一公式；类别ID只作注册表索引。old/new组内label置换等变，support行置换不变，不按TX、receiver、scene、seed、K或new class count分支。D81权重在当前support上生成后立即消费，不持久化到query state。

K>2的two-state总fit为2、`DA1_REG1`实际FULL fit为1；CSOAS不增加第二fit、BLOCK、LOO、Fisher、迭代或参数扫描。实现必须流式复用单个`288×288`类scatter buffer，不能保留`C×288×288`矩阵。query MAC与持久state bytes必须与E0精确一致。

## 5.回退与真实G0门

任一权重、`1-sum(a^2)`、`n_eff`、trace、OAS、SPD、solve或codec闭包失败时精确回退E0。量化后state与E0 byte-exact也视为机制未激活，不能进入Hard9。

单一真实K10三场景truth-free G0必须同时满足：active、无fallback、实际FULL fit=1、D42 state非E0、support跨组margin变化达到真实codec量子、query全部禁用访问为false、registration wall P90≤150ms且peak≤E0+512KiB。目标wall为≤120ms。

## 6.Hard9与最终裁决

G0通过后，运行与G0不重叠的冻结Hard9+K1，K1只验证alias/liveness。九个performance outer与E0逐row配对，八项均值必须全部严格朝优方向；任一持平、反向、fallback、artifact不完整或硬资源门失败即`REJECT_ROUTE`。只有Hard9全过才允许完整Target125。
