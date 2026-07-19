# D68交叉拟合有向冻结registry标定追踪

## 证据动机

D65通过冻结Stage2-B covariance并append新类，把after-old提高到86.11%、forget降到6.11pp，但seen-new仅59.33%。D67日志进一步证明D65原始affine存在系统性行方向不一致：before90行有12行反转，final165行有19行反转；D62为0。D67未做方向校正，D65 final支持风险为D62的7.78倍，导致连续堆叠权重接近0并在outer产生新类交换伤害。

## 唯一方法

D68对D65使用leave-one-rank-out support交叉拟合。每个fold在train rank拟合D65 expert，只在held rank产生score；对每个匿名类以`sign(mean_pos_cv-mean_neg_cv)`锁定方向。full D65 expert再以full support一对多center和`max(within,abs(gap)/2,eps)`统一尺度，最终编译`orientation*(score-center)/scale`的单一affine head。所有注册类使用同一公式，没有角色、class ID、scene、receiver或query分支。

D68不是D67的alpha修补：不混合D62与D65，也不优化连续权重。K1精确回退D62，K≥2按rank留一，支持行exact-once。失败后停止整条方向标定路线，不扫描参数。

## 地面原型边界

D68不读取D22地面组件，因为其manifest当前为`formal_phase2_eligible=false`且provenance为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`。D66已证明84个int8 ground cell可在开发探针中合法只读进入共享尺度，但性能为负向交换；现阶段不能把ground访问次数等同于有效利用，更不能让不具正式资格的组件成为最新目标候选的依赖。

## 实验与判门

开发cell固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8，复用D18`VALIDATED_ONCE/p2_min_v1` enrollment-only support。相对D62必须保持总体、场景、floor、遗忘和混淆无交换并严格改善至少一项A/F/J/floor；量化变化与margin flip必须为0。真实105行后必须保留7候选、11类、15fold、方向、风险、训练、资源和artifact全账。
