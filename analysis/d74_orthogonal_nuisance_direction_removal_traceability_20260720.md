# D74类中心正交nuisance方向删除追溯与预注册

## 机制定位

D73在15/15fold真实改变共享对角metric并降低旧/新support loss，但D62 refit把可逆坐标重标度完全吸收，outer prediction与D62逐行相同。D74因此不再做可逆metric、score校准或门控，而检验一个不可由LDA逆变换恢复的最小表示操作：删除一个对类别中心差异无贡献、但承载最大类内残差能量的方向。

## 锁定公式

在D42固定log-diagonal变换后的全部已注册support`z_i`上，计算类均值`mu_c`、全局均值`mu`与中心矩阵`C=[mu_c-mu]`。由SVD取得`C`的机器精度行空间正交基`Q`。类内残差为`r_i=z_i-mu_{y_i}`，其类中心正交部分为：

`r_i^perp=r_i-r_iQQ^T`。

令`u`为`R_perp`最大奇异值对应的唯一首个右奇异向量，并以最大绝对坐标为正固定符号。最终非可逆投影为`P=I-uu^T`。D74只在Stage2-C用`zP`重新拟合一次D62统一头`W,b`，随后编译`W'=WP`，因此query仍在原D42特征上执行一个all-registered int8仿射头。Stage2-B before状态保持D62逐位不变。K1或残差退化时精确回退D62。

该公式固定删除rank-1，不扫描rank、能量阈值、收缩、场景、类、old/new角色或support门。

## 追溯矩阵

|要求|D74实现约束|验证证据|状态|
|---|---|---|---|
|LEO_weak-only|只读取固定单观测support特征|Runner/source audit|PREREGISTERED|
|Stage2-B/C同等|before完整保留D62；final投影由全部已注册类等K残差共同决定|state hash与类平衡audit|PREREGISTERED|
|类对称|同一`Q/u/P`作用于全部类；类置换等变|置换测试|PREREGISTERED|
|非可逆性|`P`对称幂等、rank=D−1且`Pu=0`|线性代数测试|PREREGISTERED|
|中心保护|`u`与中心化类均值span正交，类间中心差异理论不变|orthogonality与pair-distance audit|PREREGISTERED|
|无query泄漏|只用outer-fit support；无outer-held/query/真值/角色|API和geometry字段|PREREGISTERED|
|正式量化态|`WP`编译为一个residual-int8/FP16状态|量化误差和零翻转审计|PREREGISTERED|
|地面组件|D22仍不具正式资格；输入/更新/状态0|resource字段|PREREGISTERED|
|资源|新增1次D62 refit与1次SVD；optimizer/epoch不增加，query额外MAC/state0|resource verifier|PREREGISTERED|

## 与历史路线非重复性

- D21-M6/D36学习低秩可训练metric或identity residual；D74不训练参数，且删除的是类中心span正交的rank-1类内方向。
- D60在full/block协方差之间按leave-one谱稳定度连续收缩，保持SPD和可逆；D74直接形成rank-287非可逆投影，不选择协方差端点。
- D61沿Fisher类间方向增加identity-primary增益；D74反向只删除与全部类间中心差异正交的最大类内nuisance方向。
- D67–D73在score/head混合、门控、聚合或可逆metric层工作；D74改变可辨识表示秩，但query仍编译为单头。

## 开发门与停止边界

固定开发单元为receiver`20-1`、seed`713101`、K10/new5、3场景×5 outer physical-rank held折，outer-fit实际K8。必须完成105/105行和30个目标量化行。只有相对D62的`A/N/H/min-A/min-N`全部不退化且至少一项严格提高，同时`B/F`、三场景和混淆无联合交换伤害，才允许第二seed。失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；停止rank数、稳定度、能量阈值、按block/场景/类方向、多个方向或投影强度扫描，不运行125。
