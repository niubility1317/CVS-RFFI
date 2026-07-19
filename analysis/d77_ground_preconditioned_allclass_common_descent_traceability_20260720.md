# D77地面预条件全类共同下降追溯

## 方法定位

D19/D25把地面旧类中心或半径直接送入分类，造成严重old/new尺度不对称；D30只做old-old重排，跨组错误不变；D36的ground anchor、offset与IRLS只移动错误；D66共享尺度带来after-old`+1.11pp`，却使seen-new`−1.33pp`、min-new`−6.67pp`和joint`−3.33pp`。D77不再让地面原型产生类别分数，也不在地面变换后refit。地面组件只定义优化几何，更新方向完全来自合法target support的全注册类OOF梯度。

## 固定公式

从不可变int8地面域×旧类聚合中心`p[d,c,j]`计算D66同源统计：类内跨域漂移`W_j`、类间信号`B_j`和可靠性`r_j=(B_j+eps)/(B_j+W_j+2eps)`。D77将其改为几何均值归一的正定预条件器：

`m_j=exp(0.5*(log(r_j+eps)-mean_k log(r_k+eps)))`，`j<160`；FFT96/RF32的`m_j=1`。

每个target row按物理rank执行8折K−1 equal-prior automatic-shrinkage LDA，在88个held support上形成11个真实类CE梯度`G_c∈R^{C×D}`。在class simplex中用20次固定Frank-Wolfe求：

`G*=argmin_{α>=0,sum α=1} ||sum_c α_c G_c||_M^2`，其中`||G||_M^2=sum_j m_j ||G[:,j]||²`。

并列vertex按相同最小导数平均，保证类置换等变。令`H=G*diag(m)`，若非退化，则所有类的一阶共同下降内积`a_c=<G_c,H>`为正。解析步长：

`eta=min_c a_c/(L_c ||H||_F²)`，`L_c=0.5 max_{i:y_i=c} ||x_i||²`。

再以`||W_D62||_F/sqrt(CD)`施加类无关trust cap，直接编译`W'=W_D62-eta H`，intercept不变，更新后不refit。无rank、alpha、阈值、场景、receiver、类角色或步长倍率扫描。

## 需求到实现追溯表

|ID|来源|可验收要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D77-R1|地面统计|只读校验D19/D22 int8组件，84个有效cell；产生几何均值归一的160维正定预条件器|`code/cvsrffi/stage2_d77_ground_preconditioned_common_descent.py`、probe loader|pending|待组件SHA、只读、域/类置换测试|当前组件formal资格为false，运行只能是诊断|
|D77-R2|OOF梯度|按8个物理rank拟合K−1 LDA，从88个held support聚合11个类CE梯度|core模块|pending|待fold闭包和有限值测试|query、outer-held、ground class role均不可达|
|D77-R3|M-共同下降|20次固定Frank-Wolfe求M范数最小组合，并列vertex平均|core模块|pending|待确定性、类置换等变和objective trace测试|不扫描迭代数或权重|
|D77-R4|解析安全更新|解析Lipschitz步长、类无关trust cap；每类OOF CE非增，退化点才identity fallback|core模块|pending|待逐类CE与正确数审计|无二元候选门|
|D77-R5|D62 final-row集成|只更新D62最终行系数，intercept不变，不refit，编译INT8/FP32 matched state|`code/scripts/probe_d77_ground_preconditioned_allclass_common_descent.py`|pending|待支持分数和量化审计|before状态保持D62|
|D77-R6|资源闭包|D42 20step＋Frank-Wolfe20step=`40<=50`；epoch20；参数<80k；组件含总状态<256KB；query额外MAC/state0|probe与artifact|pending|待精确MAC、状态和loss trace验证|地面组件25,428B计入主状态|
|D77-R7|协议闭包|单LEO_weak、support-only、全注册类对称；clean/source/query truth/role/quota/global assignment/dense query graph访问0|probe、测试、RECEIPT|pending|待105行闭包|复用D18 `VALIDATED_ONCE/p2_min_v1`|
|D77-R8|完整开发实验|receiver`20-1`、seed`713101`、K10/new5、3场景×5fold、7候选105行|run与summarizer|pending|待完整日志解析|actual outer-fit K8|
|D77-R9|性能晋级门|相对D62，`A/N/H/min-A/min-N`不退化、`F`不升且至少一项严格改善；混淆无交换伤害|summarizer与报告|pending|待同row判定|失败即关闭，不开第二seed或125|
|D77-R10|正式资格门|只有联合封存、外部authority签名通过的ground bundle才能产生formal claim|loader、报告|pending|待正式bundle artifact|当前D19组件不能满足此项|

## 效率与创新性

- 地面84个中心不参与逐query检索；适配完成后query仍只执行单一int8仿射头。
- 预条件器是坐标级只读统计，不产生旧类分数，因此不会像ground anchor那样天然偏向旧类。
- M-共同下降把地面经验与11类target证据结合：地面决定可信坐标，target support决定更新符号与类别平衡。
- D77直接改写最终row boundary，避免D66/D73类可逆坐标变化被后续LDA吸收。

## 停止条件

若专项性质测试、协议闭包或资源上限失败，先修实现；若105行完成但联合性能门失败，记录`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不扫描预条件指数、FW次数、trust cap、类权重或step倍率。
