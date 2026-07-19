# D75交叉拟合margin安全nuisance投影追溯

## 要修复的失败

D74证明“类中心正交＋最大类内残差能量”不足以识别域nuisance：15/15fold真实删除rank-1方向、12/15fold改变outer预测，但相对D62的`ΔA/N/H/F=−1.67/−5.33/−3.81/+1.67pp`，最弱新类下降10pp。support内几乎不变而outer退化，说明删除方向必须先通过独立的support-held边界安全证据。

## 数学机制

在D42变换后的全部已注册support上，对每个类内物理rank`r∈{1,…,K}`执行完全类对称的leave-one-physical-rank-out：

1.仅用`S_{−r}`拟合equal-prior automatic-shrinkage LDA`(W_r,b_r)`；
2.仅用`S_{−r}`按D74规则求与中心化类均值span正交的最大类内残差方向`u_r`；
3.在每类恰好一个held support样本上计算base margin
   `m=W_{r,y}x+b_{r,y}−max_{c≠y}(W_{r,c}x+b_{r,c})`；
4.计算固定头投影`W'_r=W_r(I−u_ru_r^T)`后的held margin`m'`与`Δm=m'−m`。

数值容差只取浮点舍入界`τ=64·eps·max(1,max|m|,max|m'|)`，不是可调阈值。仅当以下三项同时成立时接受full-support D74方向：

- 所有实际注册类的`mean_r(Δm_c,r)≥−τ`；
- 全部held样本的`mean(Δm)≥−τ`；
- projected held正确数不低于base held正确数。

通过时冻结D62 final头并编译`W(I−uu^T)`；否则精确回退D62 identity。无rank、强度、阈值、类、角色、场景或结果扫描。

## 与matched baseline的单一主要差异

相对D74只增加一个预登记、nested support-held、全类floor安全门；候选方向、D62 before/final强头、D42 metric、int8编译、query推理和全部协议输入不变。相对D62仅可能增加一个已通过安全门的rank-1非可逆投影。

## 预期可观察结果

- 每个target row记录K个held fold、每类margin delta、总体delta、正确数delta、浮点容差和accept/reject；
- reject row必须与D62 final状态和预测等价，accept row必须投影rank=287且方向已编译进单一int8头；
- 若D74退化确由support-held边界可检测，D75应拒绝伤害fold，同时保留少数可能的正向fold。

## 失败与停止条件

开发正向门固定为：相对D62的`A/N/H/min-A/min-N`均不退化，`F`不升，且至少一项严格改善；场景与混淆不得以一侧显著伤害换另一侧均值。未通过即关闭D75，不扫阈值、margin权重、rank、删除强度、场景/类/角色门，不开第二seed或125矩阵。

## 最小验证矩阵

- 复用D18 `VALIDATED_ONCE` capsule，receiver`20-1`、seed`713101`、K10/new5、3场景×5fold，outer-fit实际K8；
- 7个既有D42 candidate同行Runner，D75只替换target INT8/FP32 final state；
- 全量分析总体、场景、11类、15fold、old→new/new→old/new→new、训练、量化、MAC/状态/显存和artifact闭包。

## 协议与地面组件边界

只读合法support及其标签；query、outer held真值、clean/source、query角色、batch类数、quota、global assignment和dense query graph均为0/false。公式对类标签置换等变，不知道old/new角色。D22地面组件不具正式资格，D75 ground int8输入固定0。
