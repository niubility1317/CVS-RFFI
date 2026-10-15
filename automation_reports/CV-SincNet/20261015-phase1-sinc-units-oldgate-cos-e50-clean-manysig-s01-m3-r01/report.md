# 固定 SIG_TIME：Sinc 优化器单位消融预登记

本轮只改变原 Sinc 参数的优化器单位设置，检验历史源训练中“存在梯度和 AdamW 状态，但实际频率步长几乎低于浮点分辨率”的现象能否被纠正。SIG_TIME 网络、原始 Hz 参数化、前向滤波器构造、部署 joint160、默认 cosine 头和原半监督流程保持。数值学习恢复不等于泛化有效；所有正负结果按完整矩阵保留。

## 固定方案与依据

代码提交：`00132f7d54aff45b6e52fc41f89d7705a043bc71`；分支：`codex/phase1-feature-challenge-20261013`。方案、源资格判据及完整实现已在剩余 RSC 新目标数值回收（2026-10-15T00:20:27.595Z）之前固定。依据来自真实 9 份历史源 checkpoint/AdamW/EMA 状态和 3 份现有 SIG_TIME 初末状态记录，详见 [源证据与本地组件验证](../../../analysis/sinc_unit_optimizer_preparation_20261015.md)。没有根据目标分数选择频率倍率、父权重或学习日程。

原 Sinc 低频和带宽各 24 个参数；只在它们的 AdamW 参数组上固定设置：lr_sinc=Fs×lr_native、eps_sinc=eps_native/Fs、wd_sinc=wd_native/Fs，Fs 为原采样率 25000000 Hz。保留每轮原生 cosine 日程；原生代码覆盖所有组 lr 后，在真实 step 前恢复上述固定单位关系。其他参数（含默认 W）保持原 AdamW 设置。保留原 Hz 梯度全局 clip5，因此只声称在相同已裁剪梯度条件下的单位映射，不声称完整优化器坐标不变性。无新参数、损失、网络、前向或额外优化器更新。

| 行 | 模型种子 | 新训练 | 优化设置 | 评价 |
|---|---|---:|---|---|
| OLD_GATE_COS50 | 2026092701 | 0；复用原50轮 | 历史旧门控+ownEMA+cos | 原默认头 clean |
| SIG_TIME | 2026092701 | scratch50 | 原 AdamW | 原默认头 clean |
| SINC_UNITS | 2026092701 | scratch50 | 仅原48个Sinc参数按固定采样率映射 | 原默认头 clean |

两个新模型同种子初态、相同原数据与随机顺序，分别训练 33 label+17 pseudo，11100 更新/模型，U963900 包/模型。原 L128/U256、batch_neighbor、ownEMA0.999、UCE0.16、entropy0.01、smoothing0.01、cos、lr0.0002、wd0.0001、NoExplicitCFO 和无卫星训练保持。默认 6×160/scale30 cosine 头按原流程随骨干联合训练，禁止另训、冻结骨干后重训或替换分类头，也不拟合诊断头。单模型训练不超过 50 epoch。

## 数据、冻结与自动评分

复用既有 VALIDATED_ONCE ManySig source contract 与 clean capsule；split392005、物理角色、TX/RX/day、L6300/U56700（TX隐藏）/V27000保持不变。源 RX1/3/4/6/8、days1/2/3；V只作健康诊断。新模型从零初始化，模型/EMA/优化器不加载任何父权重。真实 checkpoint smoke 只读预定 DRO SIG_TIME 首种子 final50，先核对其合法 source_proof，再用 2 个零 IQ 包作无 query eval，不对该 checkpoint 作 optimizer 更新。

固定 final50/11100 后立即自动 clean168000 预测/行；旧6TX×7目标RX×4days，只有 ORIGINAL_HEAD。全部预测固定后独立 truth-last scorer 连接真值。报告整体准确率、Macro-F1、最差RX/day、所有登记分层和混淆矩阵、配对错误转移；n=1不代替预定三种子≥84%目标。既有目标基准已暴露，不声称新的独立盲测，无 Phase2、新TX或卫星部署声明。

## 唯一源资格与扩种子规则

在读取本候选任何新目标数值前完成并提交/推送源决定：两臂最后5轮原生L CE均≤0.5；原11100更新/U963900/0额外前向、隐藏U、默认头、真实fθ更新和feature-only导出证明完整；候选低频和带宽各至少12/24参数实际改变，student和ownEMA的实际Sinc滤波器相对L2均≥1e−6；候选与控制在真实L720的joint160 L2差≥1e−6。PASS才将两个臂原样扩展至2026092702/03；FAIL保留首种子所有结果，不扩种子，不扫倍率/epsilon/日程/父模型。目标数值不得改变扩展、训练、选择、排序或重跑。

记录 E1/17/33/50 每16批一次的真实 Sinc pre/post step、实际裁剪后总原生梯度和 Adam 一二阶矩，每模型52批/104参数记录；全部11100个实际group lr；原生L/U特征梯度E33/34/50沿用原观察器。Sinc梯度为L+原U/entropy总梯度，不冒充单目标归因。保存初末student/ownEMA滤波器差、真实部署特征、原默认W联合更新、实际epoch时间和全部U预算。只读诊断没有额外前向/更新；全部成本如实报告。

## 发布路径、环境与停止规则

唯一launch owner：codex/root/sinc-units50-20261015。普通N607既有CVS-RFFI环境，FP32、每worker CPU2线程；不改环境/账号/服务。运行根目录 `PROJECT/runs/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r01`，release/CWD `PROJECT/releases/cvs_sinc_units50_20261015_r01`；本地私有产物 `LOCAL_ARTIFACTS/sinc_units50_delivery_r01`。每GPU所有owner总计最多3实验进程，且启动时可用显存≥12000MiB；共享容量锁动态分配GPU并记录实际值。禁止停止、重启或热改其他健康任务。

发布命令：`python -u -m experiments.cvs_sinc_units50.publish --output LOCAL_ARTIFACTS/sinc_units50_delivery_r01`；远端入口 `python -u -m experiments.cvs_sinc_units50.dispatch`。一次普通账号/容量/不可变路径预检；一次release归档与一次本地/远端SHA传输比较；一次远端compile；launcher先执行聚焦负例和真实checkpoint无query smoke，PASS立即继续，不设置smoke令牌或额外审批。启动后独立核对PID/CWD/argv/GPU与日志增长。

只因本run的系统性技术失败停止：数据权限/隐藏U/默认头不符、缺失真实更新或原loss、数值非有限、固定单位映射不成立、不可变产物冲突。保留部分产物，不自动重新提交，不终止无关进程；科学源资格FAIL只阻止扩种子，首矩阵按固定预算完成并自动评分。源/目标分数不作为热改或提前停止依据。

聚焦本地组件7负例、实际原生参数和L路径、STANDARD与原AdamW精确一致、候选第一步全部非Sinc/W精确一致、原EMA和cos覆盖处理已PASS。两个真实新行原生上下文各16个合成联合step和原U loss对象保留检查PASS，见 [组件检查](local_checks.json)、[入口检查](pipeline_checks.json)。这些是合成机械验证，不是训练性能。唯一独立P0/P1审查发现缺少release依赖，已直接补齐并定向确认PASS；无第二轮全审，见 [审查](p0p1_review.json)。完整canonical配置仅恢复PROJECT前缀，3行全等工厂配置且嵌套source_config保留，见 [配置核对](canonical_roundtrip.json)。

预期真实产物以 [experiment.json](experiment.json) 为准，包括初末checkpoint、resolved/native config、冻结记录、Sinc真实step/梯度/矩和初末滤波器证据、实际部署特征、全部prediction、独立scores及same-row报告。代码/配置与本预登记提交推送后才发布；训练完成、评分完成、源资格与84%目标分别报告，不互相替代。所有成功与负结果均上传Git；不提交IQ、权重、原预测、私有机器状态或原始日志。
