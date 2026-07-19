# D64全pair局部3-block LDA tournament预注册与追溯

## 1.要修复的失败

D46/D62使用全注册类共享协方差，再按类融合full/block分数。D62虽是当前聚合最强开发点，但after82.22%、旧类floor53.33%、new84.67%，O3、N0、N2仍是联合下尾；D61–D63证明共享Fisher变换与support安全门只会在旧类保护和新类可达之间交换。D64检验一个不同假设：持续floor来自局部类别对使用同一全局几何，而不是缺少更复杂的安全门。

## 2.冻结数学机制

对当前fit中的每个无序匿名类别对`(c,d)`：

1. 只读取该pair的合法support，使用D43已验证的`z160/FFT96/RF32`三块auto-shrinkage等先验二类LDA；
2. 得到有向margin`m_cd(x)=(w_c-w_d)^T x+(b_c-b_d)`；
3. 用同一pair support上的`r_cd=sqrt(mean(m_cd(x)^2))`无参数归一化；`r_cd`非有限或不为正时fail closed；
4. 对类别`c`累加`+m_cd/r_cd`，对类别`d`累加`−m_cd/r_cd`；每类除以`C−1`；
5. 删除类别公共仿射分量，编译为一个`C×288` FP32 affine，再走既有target-old/new residual-int8 coefficient＋FP16 intercept生命周期。

该规则等价于连续全pair tournament，但query只执行最终每类一个dot，不保存pair graph、不投票、不联合优化。pair枚举、RMS公式和平均权重固定，无阈值、温度、pair选择、full/block融合或超参数扫描。

## 3.协议与类无关边界

- 所有类别对采用同一公式，对class label置换等变；不读取class ID、old/new角色、receiver、scene、outer fold或历史难类名单。
- before只用旧类support并在新类support读取前物化；final只用当前row全部注册类support。
- 不读取query特征、标签、角色、真实batch类数或quota；无Hungarian、OT、global reassignment、dense query graph或query-dependent optimization。
- 复用匹配`VALIDATED_ONCE/p2_min_v1`固定接收IQ；不访问clean/source，不生成第二LEO观测；ground int8保持只读逐bit不变。
- K1若pair residual rank不足，沿D43既有unit-covariance fallback，不增加专属K1参数。

## 4.预期可观察结果与停止门

相对D46与当前聚合最强D62，D64应通过局部pair几何同时减少old→new、new→old、new→new，并抬升O3旧类floor与N0/N2新类下尾。晋级至少要求：

- 总体before、after、new、H、joint、min-before、min-after、min-new不低于D62，forgetting不高于D62；
- clear、low-elev、rain各自before、after、new、H、joint和三项class floor不退化，forgetting不增加；
- final三类混淆均不高于D62；INT8/FP32 before/final argmax变化与margin sign flip均为0；
- 至少一个final floor或after/new/H严格改善，且至少1/15 final prediction发生真实变化。

任一场景、新类、H、floor、混淆或量化门失败即记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，停止D64；不得扫描pair阈值、pair权重、投票规则、full/diagonal协方差或RMS指数。即使通过也只讨论第二development seed，不直接运行125。

## 5.最小验证矩阵与必须报告

- receiver`20-1`、seed`713101`、K10/new5、实际K8、3场景×5 outer fold；复用D18合法capsule。
- 7候选×15fold=105行，INT8与FP32 matched；D46、D61、D62、D63为历史matched对照。
- 完成后必须报告7候选总体、3场景、11类、15fold、全部pair scale/condition/support margin、三类混淆、量化、epoch1–20、资源、artifact SHA与项目门差距；不能只说明缺陷。

## 6.计划执行面

- 实现：`code/scripts/probe_d64_allpairs_blocklda_tournament.py`。
- 测试：`tests/test_probe_d64_allpairs_blocklda_tournament.py`。
- 输出：`automation_reports/CV-SincNet/d64_allpairs_blocklda_tournament_probe_20260719/allpairs_blocklda_tournament`。
- 本地`ssr-gpu`验证并从detached clean worktree运行；本轮不访问N607。
