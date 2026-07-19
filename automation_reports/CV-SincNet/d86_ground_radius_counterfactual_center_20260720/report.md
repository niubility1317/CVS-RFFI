# D86地面半径反事实鲁棒原型研发报告

## 实验身份与目标

|字段|值|
|---|---|
|实验ID|`D86_ADV3B02_GROUND_RADIUS_COUNTERFACTUAL_CENTER`|
|时间|2026-07-20 CST|
|执行者|Codex|
|目标|高效利用D85压缩地面原型的类无关域方向和p90半径，使地面知识直接参与target support的margin稳健原型估计，同时保持旧类、新类完全相同的公式|
|比较目标|D81、D83、D84、D85同receiver20-1、seed713101、K10实际K8、new5、3场景×5fold结果|
|当前状态|`COMPLETED_DIAGNOSTIC_PERFORMANCE_NEUTRAL_QUANTIZATION_UNSTABLE_NOT_PROMOTABLE`|

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

本地PID=`9408`于2026-07-20 07:50:02 CST启动，126.18秒完成105/105行、7候选×15 outer rows；`stderr=0B`，无Traceback/OOM/NaN/Inf。入口为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe E:\type10-7\code\snapshots\d81wt\code\scripts\probe_d86_ground_radius_counterfactual_center.py`；按预登记传入D85 v2组件、D18 before/after封装、class binding、`runtime-root=d41wt`、`probe-root=d81wt`、`device=auto`、`mode=development_select_unverified_component`和`candidate-set=d42_v1`。完整原始输出和细化JSON保存在根报告目录；本报告保留完整同row性能与判定。

### 七候选联合指标

|candidate|B|A|N|H|F|J|min class B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D42-USLDA-INT8|92.78|82.78|84.67|82.94|10.00|26.67|80.00/53.33/73.33|73.33/50.00/46.67|22/8/15|
|D42-USLDA-FP32-MATCHED|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|53.33/33.33/36.67|33/22/19|
|D42-D40-HNBR-INT8-NEGATIVE|85.56|85.00|15.33|25.16|0.56|0|66.67/63.33/0|40.00/40.00/0|2/0/0|
|D42-D41-BEC-INT8-NEGATIVE|86.11|20.56|78.67|31.50|65.56|0|76.67/0/36.67|46.67/0/26.67|142/0/32|
|D42-PROTOnet-CDA-ZID160|71.11|48.33|52.67|48.97|22.78|0|33.33/13.33/3.33|13.33/0/0|0/0/0|
|Z0_SUPPORT_ONLY|71.11|48.33|52.67|48.97|22.78|0|33.33/13.33/3.33|13.33/0/0|0/0/0|

数值均为%；unknown拒识、coverage、rollback、defer不属于本锁定K10/new5已注册类筛选，记为`N/A`。

### 目标INT8逐场景指标

|场景|B/A/N/H/F/J|min class B/A/N|row floor B/A/N|o→n/n→o/n→wrong-n|
|---|---|---|---|---|
|clear|98.33/91.67/98.00/94.441/6.67/50.00|90/70/90|90/60/90|2/1/0|
|low-elev|91.67/80.00/76.00/76.922/11.67/20.00|80/60/50|70/60/20|7/5/7|
|rain|88.33/76.67/80.00/77.447/11.67/10.00|60/30/70|60/30/30|13/2/8|

### 15个outer row

|scene/fold|B/A/N/H/F/J|floor B/A/N|o→n/n→o/n→wrong-n|
|---|---|---|---|
|clear/0|100/100/90/94.74/0/50|100/100/50|0/1/0|
|clear/1|100/83.33/100/90.91/16.67/0|100/0/100|0/0/0|
|clear/2|91.67/83.33/100/90.91/8.33/50|50/50/100|1/0/0|
|clear/3|100/100/100/100/0/100|100/100/100|0/0/0|
|clear/4|100/91.67/100/95.65/8.33/50|100/50/100|1/0/0|
|low/0|100/75/80/77.42/25/50|100/50/50|3/1/1|
|low/1|83.33/58.33/70/63.64/25/0|50/50/0|1/0/3|
|low/2|83.33/91.67/70/79.38/-8.33/0|50/50/0|0/2/1|
|low/3|100/100/70/82.35/0/0|100/100/0|0/1/2|
|low/4|91.67/75/90/81.82/16.67/50|50/50/50|3/1/0|
|rain/0|83.33/83.33/60/69.77/0/0|50/50/0|2/0/4|
|rain/1|100/66.67/90/76.60/33.33/0|100/0/50|4/1/0|
|rain/2|91.67/83.33/80/81.63/8.33/50|50/50/50|1/0/2|
|rain/3|83.33/75/90/81.82/8.33/0|50/0/50|3/0/1|
|rain/4|83.33/75/80/77.42/8.33/0|50/50/0|3/1/1|

### 逐类性能

|TX|角色|B|A/N|遗忘/缺陷|
|---|---|---:|---:|---|
|14-10|旧|96.67|93.33|3.34pp|
|14-7|旧|80.00|53.33|26.67pp，总体最弱旧类|
|20-15|旧|96.67|90.00|6.67pp|
|20-19|旧|93.33|93.33|0pp|
|6-15|旧|93.33|73.33|20.00pp|
|8-20|旧|96.67|93.33|3.34pp|
|1-16|新|—|93.33|良好|
|1-18|新|—|73.33|最弱新类|
|18-10|新|—|90.00|良好|
|14-11|新|—|76.67|偏弱|
|8-3|新|—|90.00|良好|

### 机制、量化和资源

- 1,080个D62 component fits、2,160次support-center transform；14个domain、84个ground cell全部进入，组件入口/出口哈希一致，query使用0行。
- before类中心平移L2 min/mean/max=`0.002556/0.004054/0.005066`，final=`0.001593/0.003131/0.004743`；最低权重`0.115814`，最低有效样本数`7.9921/8`。类内残差误差≤`2.78e-17`，FFT96/RF32误差0。
- 相对D81/D85，15/15 outer prediction hash相同，所有D85→D86离散转移均为0。matched FP32在`low-elev/fold0`有1个outer argmax和margin sign flip，且方向为多1个old→new错误；FP32 A=82.22%、F=10.56%。INT8/FP32 score最大误差`0.0018687`。
- 继承D62训练20步×15 row，loss min/mean/max=`0.07560/0.30719/1.11738`，support accuracy=`89.58/98.77/100%`；D86额外optimizer step/参数=0/0。

|资源|D85|D86|变化|
|---|---:|---:|---:|
|ground统计MAC|216,724|216,724|0|
|新增适配MAC|22,107,284|98,551,444|+76,444,160|
|总适配MAC|24,913,331,254|24,989,775,414|+0.307%|
|D86新增占总适配|—|0.394%|低于5%门|
|query MAC/额外MAC|6,624/0|6,624/0|0|
|总状态|14,399B|14,399B|0|
|参数/epoch/step|2,016/20/20|2,016/20/20|0|
|peak CUDA/dense query graph|22,886,912B/0|22,886,912B/0|0|

## 缺陷与最终判定

D86仍缺项目K10/new5目标`A=9.22pp`、`minA=34.67pp`、`N=7.33pp`。权重几乎均匀，真实半径只产生`0.0016..0.0051`中心移动，被INT8边界吸收；唯一FP32边界变化还是负迁移。继续扫描半径倍率/权重违反锁定设计并有old/new交换风险。

资源门通过，但严格改善、`outer prediction≥1/15`、INT8/FP32一致性和绝对性能门失败。停止seed2/125，状态为`COMPLETED_DIAGNOSTIC_PERFORMANCE_NEUTRAL_QUANTIZATION_UNSTABLE_NOT_PROMOTABLE`。下一路线必须从prototype重加权转为带target identity保护的非二次margin head残差，同时保持old/new同式、物理rank交叉拟合、query0和单INT8 affine部署。

证据SHA256：receipt=`bc47a39dda3ad8cb124c1e2baf2101b9e163ec7e96597cc75ee338a483a3d4da`；training log=`239a2786113fc6d7393e834d6b82157e852abf30c71e62525f3a4753fd0c9cb0`；D86 metadata=`2df3a01fdaddec3d2e7a56cf447dd89bd475ad45286d2bd85cc0f61daffd8253`；完整summary=`58a6174339c270d4cfbc4daa8340db4ccd424dd0117d5466a29865e89a8baada`。
