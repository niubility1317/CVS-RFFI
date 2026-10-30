# PAMF 层次密度发现与匿名注册

PLANNED，无新性能。Run `20261031-phase3-hier-register-s2026092701-r01`，actualcode `d6f3a19fe06252c2e4b9ed199939896b4504ab62`，完整Phase3目标ACTIVE；唯一主predicted_REG1，pooled/predicted×REG0/REG1四列、三DURA条件不变。方法及冻结矩阵见 `docs/PAMF_PHASE3_HIER_REGISTER_PLAN_20261031.md`，逐row/六类seed/路径唯一维护在experiment.json。

原source F0/default6、r03完整334721、DURAv1/R69、density160 known90/OE32/whitener和两旧6τ保持。新算法为sklearn HDBSCAN固定minsize16/minsamples8/eom/brute/njobs1/alpha1/epsilon0/allowSingleFalse/copyTrue/maxsizeNone，在同一完整U层次密度选簇，不使用sourceε切割/真实K/配额/oldclusterlabels。原qualification size16且predicted拒识strictmajority、全成员Gaussian shrinkn/(n+160)/no1K、共同竞争/oldties和sourceVknown12两globalτ规则保持。

只继承上轮signal-register/e48f3b来源与source聚合和U384公开ids/logits/closed/features；旧assignments/rejection忽略重算，旧cal/newτ/score/NV/Q均不进fit。actual3CP synthetic2先于缓存值，原sourceGaussianexact和sourceNN/G身份/capsule/split匹配，newsourceL/sourceεfit/真实GFsignal全0，旧sourceε仅来源记录。共享engine局部policy+scorer override，不改全局run身份，原默认贯通已单项回归。

源split392005/VALIDATED_ONCE：L6300、U_s56700、OE6134 onlyoutside-six/publicRXday、V27000额外knownlabels（各旧类4500）、NV3150 unknown19-15/19-16/7-1各1050。旧六14-10/14-7/20-15/20-19/6-15/8-20，sourceRX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23，sourceL每cell70，原L/(L+Us)10%。本轮sourceL/OE/Us IQ均0。

U384是RX24-1/index26/03-08原G10星地+DURA，源角色与futureQ物理互斥；futureQ同RX/03-15的864记录，old6各96=576、nonold3各96=288，仅评分器识别真实U到达/后来集合。clean/star同物理Q均接DURA并复用冻结原r03和F0同输入基线。600km/25MHz/2.45GHz residual/post_sync/noEQ/referenceSNR30dB/reflect、原六分量/郊区城区各0.5、receiver2027/scene3001/channel4001，一record一received；不新增物理或clean学习。历史NV/Q、单RXseed/source日期已见、high缺19-15相关指标NA、不formalRX24-18访问。

anonfreeze→sourceV两新globalτ＋旧two6τexact→全阈值冻→NV+两Q完整4x3预测→独立CPU4truth，仅UtruthHungarian映射、不回流、不Q重排。所有真Uarrivals入AN分母，未发现计错，unmatched匿名接受计错。REG0RUallnonold/REG1later分母分开，报actualunregistered及同later固定配对。U纯度非未来AN；预期注册集合old6+全部真arrivals含未发现，实际K另报。

最新v3 gate两futureoverallAN80、oldrelREG0和sameDURAF0abs2pp、old/预期registeredCK90、laterRUclean95/star90及主景完整roster/保护；sceneAN报告不新增80门槛，sourceknown12须过。仅加竞争不能增加old正确或同later拒识，AN单项通过不替代完整晋级，UDA/更优拒识/持续多session仍需完成。源失败/K0也全部测试，不改主或调参数选择性重跑。

新NN/真实GFsignal/sourceεfit0，actualCP synthetic2each；新HDBSCAN U384及实际K匿名聚合、两sourceV校准，K0只零占位。旧sourceL6300signal/U384GFsignal/ε/GMM、F0r03G训练及Q缓存成本继承；NN834665+实际统计/τ字节，不部署source/U行bank，过程时间非B1。

命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_hier_register_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-hier-register-s2026092701-r01-release/runtime.json`。N607解释器 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`、CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-hier-register-s2026092701-r01-release`，唯一owner `codex-phase3-hier-register-20261031`，不可覆盖run `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-hier-register-s2026092701-r01` /logs；singleproducer GPU free≥4096MiB/每卡新增本任务≤2，独立CPU CUDAempty。不得停止/修改/重启其他owner。

10unique+45sub通过（新core/cache边界、旧默认唯一actual回归、新缓存actualproducer→CPU4truth、两改动旧fixture定向、新launcher），10AST/11镜像文件，唯一直接P0/P1 PASS；local sklearn1.7.2/N6071.7.0签名verified/noinstall。单preflight/archive SHA/compile/启动PID-CWD-argv-GPU-log核实；非法角色/祖先/非有限/输出碰撞/无法合法pred保留、不partial truth/覆盖重启，UNKNOWN核原PID/root；科学负结果不停止健康任务。IQ/weights/pred/rawlogs/private runtime不进Git。
