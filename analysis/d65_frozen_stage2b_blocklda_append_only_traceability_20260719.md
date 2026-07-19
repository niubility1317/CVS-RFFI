# D65冻结Stage2-B Block-LDA追加式注册预注册与追溯

## 1.要修复的D64根因

D64在before保持92.78%，但注册后after降至74.44%、new降至77.33%、遗忘升至18.33pp；2100个二类pair在support上全部100%正确，held却出现37/16/18混淆。核心问题不是量化或拟合不足，而是注册从6类15个pair扩到11类55个pair后重写所有旧类row。D65直接检验“冻结Stage2-B目标域几何并只追加类别row”能否消除这种registry-size漂移。

## 2.冻结数学机制

1. Stage2-B只读取旧类合法support，在D42已学习的目标域特征上估计一次equal-prior auto-shrinkage covariance，并保留D43已验证的`z160/FFT96/RF32`三个对角块；记为`Sigma_B`。
2. 对任意已注册类别`c`，只由该类support均值`mu_c`和同一冻结精度矩阵产生row：`w_c=Sigma_B^{-1}mu_c`，`b_c=-0.5 mu_c^T w_c`。equal-prior公共常数完全省略。
3. Stage2-C不更新`Sigma_B`。旧类support和row保持逐bit不变；新类用完全相同公式计算并追加row。预测阶段不知道注册时序或old/new角色，只对全部注册类做一次affine argmax。
4. K1或Stage2-B residual退化时，`Sigma_B`按既有规则退化为单位阵并同样冻结；不增加K专属参数。

该机制只有“冻结一次、同公式追加”这一条路线，没有freeze强度、协方差混合、角色权重、阈值、温度、pair选择或场景分支。它与D43 3-block的单一主要差异是final不再用all-class support重估协方差；与D64的单一主要差异是没有pair图，任一class row不依赖其他类别的新增或删除。

## 3.协议与类无关边界

- Stage2-B读取旧类support、Stage2-C读取新类support，严格遵守既有生命周期；before物化前不读取新类对象。
- 行公式对所有类别一致且对标签置换等变；query不读取class ID、old/new角色、receiver、scene、fold、query真值、真实batch类数、quota或global assignment。
- 复用匹配`VALIDATED_ONCE/p2_min_v1`的固定接收IQ，不访问clean/source，不生成第二LEO观测；ground int8只读不变。
- target-old/new均编译为同一residual-int8 coefficient＋FP16 intercept状态；query无dense graph、无batch优化、无额外状态。

## 4.可证伪预期与停止门

硬闭包：final前6个旧类的FP32 row、int8 codes/scales和FP16 intercept必须与before逐bit一致；任何不一致立即fail closed。量化before/final argmax变化和margin sign flip均须为0。

性能晋级至少要求：

- before不低于D46的92.22%；after不低于D62的82.22%，new不低于84.67%，H不低于82.62%，forgetting不高于10.56pp；
- min-before不低于80%、min-after不低于53.33%、min-new不低于73.33%，joint不低于26.67%；
- clear、low-elev、rain的after/new/H不得低于D62，forgetting不得高于D62；三类混淆总数均不得高于23/8/15；
- 至少严格改善after、forgetting、min-after或old→new之一。

任一主指标、场景、floor、混淆、量化或逐bit追加闭包失败即记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，停止D65；不得扫描freeze系数、full/block混合或角色化旧row保护。通过也只进入第二development seed，不直接运行125。

## 5.最小真实验证与详细报告

- receiver`20-1`、seed`713101`、K10/new5、实际K8、3场景×5 outer fold；复用D18 capsule。
- 7候选×15fold=105行，INT8/FP32 matched；D46、D61、D62、D63、D64为matched历史对照。
- 必须报告7候选、3场景、11类、15fold、旧row逐bit闭包、协方差condition、三类混淆、量化、epoch1–20、资源、artifact SHA和项目门差距，不能只说明缺陷。

## 6.计划执行面

- 实现：`code/scripts/probe_d65_frozen_stage2b_blocklda_append_only.py`。
- 测试：`tests/test_probe_d65_frozen_stage2b_blocklda_append_only.py`。
- 输出：`automation_reports/CV-SincNet/d65_frozen_stage2b_blocklda_append_only_probe_20260719/frozen_stage2b_blocklda_append_only`。
- 本地`ssr-gpu`从detached clean worktree运行；本轮不访问N607。
