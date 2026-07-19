# D78地面域切向最差类边界追溯

## 方法定位

D77证明地面84个int8域×类中心能够在严格support-only和单仿射部署约束下参与适配，但对角可靠性预条件器只让OOF CE均值下降`0.000251`，15/15个outer prediction hash均未变化。失败来自两个层面：对角度量丢失跨坐标域形变，且“所有类CE共同下降”在梯度冲突时使11/15个row退化为identity。

D78把地面中心的类均值剥离后，将全部域内残差压成固定低秩域切向基；地面只提供“接收域可能沿哪些方向形变”，不提供类别分数。target support在该子空间内优化class-symmetric的最差类top-2 margin，允许不同类CE发生受控交换，从而直接对准最差类floor和三类部署混淆。

## 固定公式

设地面解量化中心为`p[d,c]∈R^160`，14个domain、6个ground class且84个cell全有效。先按类去均值：

`r[d,c]=p[d,c]−mean_d p[d,c]`。

将84个残差堆叠为`R∈R^(84×160)`，取SVD的前

`q=min(domain_count−1,numerical_rank(R))`

个右奇异向量，得到`U∈R^(160×q)`。本组件预期`q<=13`；这是14个域最多13个独立域对比的代数上限，不是扫描参数。D78只通过投影矩阵`UU^T`使用该基，SVD符号与简并子空间旋转不影响公式；FFT96/RF32方向固定为0。

每个target row沿用D77的8个物理rank crossfit：每折以K−1 support拟合equal-prior automatic-shrinkage LDA，在总计88个held support上取得base score。对样本`i`，真实类为`y_i`，固定top rival为`r_i=argmax_{k≠y_i}s_ik`，base margin为`m_i=s_i,y_i−s_i,r_i`，切向特征为`z_i=x_i U`。

学习零行均值的`A∈R^(C×q)`，使修正margin为：

`m'_i=m_i+(A[y_i]−A[r_i])·z_i`。

逐类top-2 logistic loss：

`l_c(A)=mean_{i:y_i=c} softplus(−m'_i)`。

以初始类均值`tau=mean_c l_c(0)`固定温度，优化平滑最差类目标：

`J(A)=tau log(mean_c exp(l_c(A)/tau))`。

执行固定20个接受步。每步从解析pairwise-logistic Lipschitz初值开始，只做确定性二分回溯直至`J`非增，并把`A`投影到零行均值及类无关Frobenius球：

`||A||_F<=||W_D62||_F/sqrt(CD)`。

最终直接编译`W'=W_D62+A U^T`，intercept不变，不refit、不按support指标选择identity、无rank/温度/step/radius/类权重扫描。

## 需求到实现追溯表

|ID|来源|可验收要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D78-R1|ground tangent|严格校验D19/D22 int8组件；84个cell只读；类去均值SVD；`q=min(13,numerical rank)`|`code/cvsrffi/stage2_d78_ground_tangent_worstclass_margin.py`、probe loader|specified|待SVD重构、类/域置换与符号不变测试|组件当前只具诊断资格|
|D78-R2|support crossfit|8物理rank、88个held support、固定top rival、无outer/query|core|specified|待真实与合成闭包|actual outer-fit K8|
|D78-R3|floor objective|逐类等权top-2 logistic＋固定温度smooth-max，类置换等变|core|specified|待数值梯度、置换与最差类loss测试|无old/new角色|
|D78-R4|固定优化|20个接受步、解析初始步与确定性回溯、零行均值、类无关trust ball|core|specified|待单调目标、20步、确定性测试|无候选/超参扫描|
|D78-R5|D62集成|仅对D62 final rows加`AU^T`，bias不变，不refit，INT8/FP32 matched|`code/scripts/probe_d78_ground_tangent_worstclass_margin.py`|specified|待probe测试与真实量化审计|query仍为单仿射|
|D78-R6|资源闭包|D42 20step＋D78 20step=`40<=50`；epoch20；参数<80k；含ground总状态<256KB；query额外MAC/state0|probe与artifact|specified|待MAC、状态和runner闭包|训练自由度最多143|
|D78-R7|协议闭包|单LEO_weak、support-only、全注册类对称；clean/source/query truth/role/quota/global assignment访问0|probe、测试、RECEIPT|specified|待静态锁、专项与真实闭包|复用D18 VALIDATED_ONCE|
|D78-R8|完整开发实验|receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、7候选105行|run与summarizer|pending|待完整日志解析|先验证离散边界是否改变|
|D78-R9|性能晋级门|相对D62，`A/N/H/min-A/min-N`不退化、`F`不升且至少一项严格改善；混淆无交换伤害|summarizer与报告|pending|待同row判定|失败即关闭，不开第二seed或125|
|D78-R10|正式资格门|只有联合封存且由外部authority签名的ground bundle才能产生formal claim|loader、报告|blocked|当前无外部签名joint bundle；probe强制`formal_candidate=false`|诊断结果不得写成正式性能|

## 创新性与效率

- D66/D77使用的是坐标级地面可靠性；D78首次在当前D42-D62链上利用跨坐标域形变子空间，保留地面原型的相关结构。
- 不把ground class与target class绑定，不生成ground分类分数；类去均值残差只形成共享域切向几何，因此不会天然偏旧类。
- 优化对象从平均CE改为smooth worst-class top-2 margin，直接对应最差类floor和实际argmax错误。
- 训练只维护最多`11×13=143`个切向系数；部署时低秩残差预先并入INT8 affine rows，query没有矩阵链、原型检索或batch优化。

## 停止条件

若性质测试、协议闭包或资源上限失败，先修实现。若105行完成但D78相对D62没有严格联合改善，记录`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不扫描rank、温度、回溯步、trust radius、class weight或margin倍率。
