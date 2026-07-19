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

## R1运行前修订

首版代码测试揭示Stage2-C若重新标定旧行，会丢失D65的冻结优势。真实运行前将生命周期锁为：Stage2-B有向标定旧行后冻结其字节和共同affine项；Stage2-C同公式只标定并追加新行，旧行逐bit不变。该修订无角色query分支、无参数扫描，且在任何outer性能计算前登记。

R1已实现独立数学core、probe和两组10项专项测试；专项10/10、D42–D68完整链325/325通过，完整链用时81.1s。干净worktree复跑325/325通过，用时82.8s。

## 真实结果与否决

真实105/105行完成。D68 INT8的B/A/N/H/F/J为58.89/51.67/14.00/18.66/7.22/0.00，min-B/A/N为50.00/43.33/0.00，混淆旧→新/新→旧/新→新为20/118/11。相对D62，B/A/N/H/J分别下降33.89/30.56/70.67/63.97/26.67pp；F减少3.33pp仅由注册前B先塌至58.89%造成，不是旧类保护改善。

机制审计显示INT8 before/final分别有14/28个负方向行，有向变换把平均support风险降低约9.86%/9.73%，但每行独立标准化删除了D65原始joint head的绝对跨类尺度。20epoch support acc最终100%，outer却灾难性失败；118/150个新类held样本被判成旧类。matched FP32/INT8仍有1个margin sign flip及3个final outer argmax变化，量化门也失败。

ground实际输入为0。D66此前真实读取84个int8 ground cell，但只形成A小幅改善与N、min-N、J下降的负交换；D65低F来自冻结生命周期，而非ground。D22仍为`formal_phase2_eligible=false`且provenance未按当前协议验证，因此D68不接入它是协议要求，不是遗漏。

最终状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。停止per-row signed calibration、第二seed和125矩阵。下一候选只能保持D62绝对joint尺度，检验冻结D62旧行并追加D62同族新行的单一生命周期组合。
