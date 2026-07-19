# D86地面半径反事实鲁棒原型研发报告

## 实验身份与目标

|字段|值|
|---|---|
|实验ID|`D86_ADV3B02_GROUND_RADIUS_COUNTERFACTUAL_CENTER`|
|时间|2026-07-20 CST|
|执行者|Codex|
|目标|高效利用D85压缩地面原型的类无关域方向和p90半径，使地面知识直接参与target support的margin稳健原型估计，同时保持旧类、新类完全相同的公式|
|比较目标|D81、D83、D84、D85同receiver20-1、seed713101、K10实际K8、new5、3场景×5fold结果|
|当前状态|`IMPLEMENTED_LOCAL_VALIDATED_PENDING_105_ROW_RUN`|

## 方法、创新点与假设

D86不把地面旧类中心映射为target类分数。它先按D84从14个ground domain×6个旧类中心提取跨类共识单位方向`u_d`，形成方向后丢弃ground类身份；再从D85真实v2组件读取每个domain×class的p90余弦半径，固定令

`r_d=median_c(r_d,c)`，`a_d=sqrt(2*r_d)`。

对每个target support样本`z_i`和每个竞争类`j`，计算平方距离margin，并减去地面半径对称反事实`z_i±a_d*u_d`在全部14个domain上的最坏解析敏感度。得到最近竞争类的最坏margin`m_i`后，使用

`risk_i=softplus(-m_i)`，`w_i∝1/(1+risk_i/mean_class(risk))`

形成类内稳健权重，只把该target类的z160公共中心平移到加权均值；类内残差、FFT96和RF32严格不变。该规则对target-old/new完全一致、对ground方向正负号和target类排列等变，没有学习率、半径倍率、rank或强度扫描。K1/K2严格回退D62。

与既有路线的区别：D83是统一二次协方差加载；D84/D85只用统一中心平移且15/15预测不变；D86让真实半径直接控制边界样本对prototype的贡献，是非二次、逐support样本、零优化器的鲁棒原型估计。假设是边界附近且对合法ground域漂移敏感的support样本更可能是不可靠的域偏差样本，降低其类中心影响可改善离散old/new联合分类。

## 协议与数据边界

|项目|锁定值|
|---|---|
|协议|`p2_min_v1`；复用匹配`VALIDATED_ONCE`D18 capsule，不触发数据重验|
|数据|每个物理样本一个固定`leo_*_weak`观测；support/query物理ID不相交|
|反事实语义|固定received IQ特征的数学view，不增加K，不作为新物理样本|
|适应数据|support-only；query使用量0；无clean/source访问|
|决策|逐query独立，单一INT8 affine head；无role Oracle、class quota、全局重分配|
|ground组件|D85 v2只读组件，状态仍为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，因此本实验强制非正式、不可推广|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|组件NPZ SHA256|`1ac2424fee2ef804d83d7c8faca8d27c7c0267c0d9d7c8b97af0cf053bfb4ea6`|
|组件manifest SHA256|`6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112`|
|class binding语义SHA256|`76735ae6d9b2d7e58f683635ca2644e00fbd27a515246aab9d47488c1ab5111f`|
|pre-sign content root|`098badd1e82c05c1029cb02c024fe7d3c433488e8ab22e5c6e2ba0516b8d0055`|

## 实现、版本与本地验证

根目录`E:\type10-7`不是Git仓库；实现先进入干净Git工作树`E:\type10-7\code\snapshots\d81wt`，再以精确提交镜像到主发布仓库。没有修改D42/D62 runner，也没有远端专用修改。

|文件|作用|SHA256|
|---|---|---|
|`code/cvsrffi/stage2_d86_ground_radius_counterfactual_center.py`|ground方向/半径解析与support鲁棒中心变换|`853255bf7cb1a517e1f56fa93a50639e417d17d23f81b42c19e7a55891529096`|
|`code/scripts/probe_d86_ground_radius_counterfactual_center.py`|复用D85 v2 loader和D84/D62完整runner闭包|`5b09872d36877ddebadcff56b166c368000d1763ec2c98b71757dd8db385d30e`|
|`tests/test_stage2_d86_ground_radius_counterfactual_center.py`|公式、不变量、类置换、方向符号、K1/K2回退|`0ebbae211f3f49efb8ed0cfe3aa42f52020e0d5082c0d32208384d837c61b239`|
|`tests/test_probe_d86_ground_radius_counterfactual_center.py`|D62 full/block OOF闭包和资源计数|`e10e878648dd14d9f13ff89849ac65b067a45fbf0b57439ca6776250aae526b9`|

验证命令：

`python -m py_compile code/cvsrffi/stage2_d86_ground_radius_counterfactual_center.py code/scripts/probe_d86_ground_radius_counterfactual_center.py`

`python -m pytest -q tests/test_stage2_d86_ground_radius_counterfactual_center.py tests/test_probe_d86_ground_radius_counterfactual_center.py tests/test_stage2_d84_ground_crossclass_consensus_center.py tests/test_probe_d84_ground_crossclass_consensus_center.py tests/test_probe_d85_ground_radius_calibrated_consensus.py`

结果：22项全部通过，退出码0。合成D62闭包确认每次full/block fit保持原support行数和K，输出仍是单一有限affine head。

## 预登记矩阵、资源与停止门

本轮仅运行一个development cell：receiver`20-1`、seed`713101`、K10实际K8、seen-new5、clear/low-elev/rain×5fold、7候选，共105行。环境为本地`ssr-gpu`，工作目录`E:\type10-7\code\snapshots\d41wt`，代码入口来自`E:\type10-7\code\snapshots\d81wt`，输出为`E:\type10-7\automation_reports\CV-SincNet\d86_ground_radius_counterfactual_center_20260720\ground_radius_counterfactual_robust_center`；stdout/stderr保存在同一报告目录。无需N607，本轮也不访问N607。

资源目标：ground预处理<0.5M MAC；额外适配MAC<5%D62；optimizer step=0；query额外MAC/state=0；组件含总持久状态≤14,399B；无dense query graph。

相对D81/D85基线`B/A/N/H/F/J=92.78/82.78/84.67/82.9366/10.00/26.67%`，只有同时满足以下条件才允许seed2：

1. `B、A、N、H、J、min-class、row-floor`及三场景同项均不下降，`F≤10%`，三类混淆均不增加；
2. `A、H、minA、J、rain-A、old→new`至少一项严格改善；
3. 至少1/15个outer row出现离散预测变化，且旧类、新类paired transition均无净负迁移；
4. INT8/FP32 outer argmax差异和margin sign flip均为0；
5. 资源目标全部满足。

任一条件失败即标记详细负结果，不扫描半径倍率、风险权重、rank或回退强度，不启动seed2或125。完成后必须报告七候选同row表、三场景表、15个outer row、逐类准确率与遗忘、混淆、D85→D86预测转移、连续机制、量化和资源。

## 运行记录与结果

待105行运行完成后更新。
