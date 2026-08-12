# E0 FULL/BLOCK Pareto蒸馏设计

状态：`DESIGN_FROZEN`

## 1.目标与边界

`E0_FULL_BLOCK_PARETO_DISTILL`只在`DA1_REG1`且`K>2`时激活。它复用E0的288维联合特征、ground-spectrum稳健中心、old/new任务均衡协方差和单F0查询头；`DA1_REG0`保持E0，K1/K2保持D92 FULL精确alias。REG0的seen-new与H报告为`N/A`。

方法只读取同一outer的注册support和联合封存的Phase1聚合知识。query及其view不参与fit、目标、修正、回退、停止或选择；禁止clean/source、query truth、role Oracle、class quota和跨query重排。

## 2.共享统计与双几何

对经过一次D81稳健中心变换的support`(X,y)`，只计算一次类别均值及任务均衡协方差：

`Sigma = 0.5 Sigma_old + 0.5 Sigma_new`。

由同一`Sigma`构造两个等先验LDA头：

- `theta_F = G(LDA(Sigma))`；
- `theta_B = G(LDA(blockdiag_160_96_32(Sigma)))`。

`G`减去全类公共系数行与公共截距。BLOCK不得重新估计协方差、稳健中心或support统计。正式收据固定`covariance_estimation_count=1`、`robust_center_transform_count=1`、`full_solve_count=1`、`block_solve_count=1`、`LOO=0`、`Fisher=0`。

## 3.互补方向与尺度

在每个组件上计算组均衡、类中心化support-logit RMS：先对每个真实类的support样本计算去全类均值后的logit平方均值，再在old六类与new类之间等权聚合。记结果为`r_F,r_B`。定义

`D = G((r_F/r_B) theta_B - theta_F)`，

`theta(beta) = theta_F + G(diag(beta) D)`，`0 <= beta_c <= 1`。

`beta=0`严格等于E0 FULL；`beta=1`对应按FULL尺度对齐的BLOCK行。所有old类按同一公式、所有new类按同一公式，类别ID只用于索引注册表，不参与分支。

## 4.一次词典序support求解

先以部署E0 FULL头固定每类bottom-20% support集合；`method=lower`，阈值并列样本全部纳入。每个样本的margin为真实类logit减去最强竞争类logit。求解过程不扫描权重：

1. 最大化共同下界`t`，约束六个旧类各自的固定tail平均margin增益，以及一个将所有新类support合并后冻结的lower-Q20 new→old margin增益，均不小于`t`。新类只形成一个组级约束，不逐新类加门，以保持新类置换等变并避免K5下过度收紧为E0。
2. 在保持第一阶段最优值的条件下，最小化old→new与new→old零阈值hinge均值的最大值。
3. 在前两阶段最优值不退化的条件下，最小化`||G(diag(beta)D)||_F^2`。

前两阶段使用确定性HiGHS线性规划；第三阶段使用确定性凸二次目标并在返回后重新核对全部线性约束。数值失败、不可行、非有限或约束残差越界时exact E0 fallback。

## 5.D42部署

对唯一连续解执行一次正式D42 codec回环，并直接在解码后的`Q_D42(theta(beta))`上重算tail、双向hinge、类别置换闭包和新/旧support指标。只允许一次固定的最小范数code-local修正；不得使用scale列表、回缩搜索或query选择。

出现以下任一情况时不发布Hard10：部署头与E0 byte-exact；全部support跨组margin变化小于一个真实D42量化步；部署support共同下界不为正；双向hinge任一方向劣于E0；永久state或query MAC不等于E0。结构错误直接抛出，合法数值退化返回exact E0并标记`LOCAL_INVALID`。

## 6.资源与裁决

K>2的two-state组件计数为4，`DA1_REG1`实际solve为2；最终只持久化一个D42 F0头。Hard10发布前先做真实checkpoint无query资源smoke。历史未共享双fit约191ms，不构成本方法资源证据；新共享统计路径必须以实测证明wall硬门`<=150ms`且`<=1.50x E0`、peak`<=E0+512KiB`。

正式Hard10仍是唯一性能证伪器。八项均值任一平或反向即`REJECT_ROUTE`；八项全正但幅度、稳定性或目标资源未齐且硬上限未破，裁决`REVISE_ONCE`；全部通过才输出`ADVANCE_TO_TARGET125_CANDIDATE`，不得自动运行Target125。
