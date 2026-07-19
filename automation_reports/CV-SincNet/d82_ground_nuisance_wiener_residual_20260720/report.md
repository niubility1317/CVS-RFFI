# D82地面干扰谱稳健中心与Wiener残差收缩

## 实验登记

- 实验ID：`d82_ground_nuisance_wiener_residual_20260720`；登记时间：2026-07-20 05:49 HKT；操作者：Codex。
- 目标：解决D81仅在1/15 outer folds产生变化、rain场景无改善的问题，使地面压缩原型提供样本级域适应先验，同时不压制新类。
- 对照：同一`rx20-1/seed713101/K10/new5/3场景×5fold`上的D62与D81；所有比较使用同一row完整指标。
- 状态：`PREREGISTERED_LOCAL_VERIFIED_QUERY_NOT_OPENED`。

## 单一机制差异与数学锁

D82保留D81的一步Cauchy稳健类中心，并新增一个参数无关的地面干扰方向残差收缩。84个只读int8地面域×类中心先逐类去中心，得到rank由`ceil(participation-ratio effective rank)`唯一决定的地面干扰基`U`和归一化谱`π`。设rank为`r`，固定信号尺度`s=1/r`，第j方向的Wiener保留率为：

```text
retention_j = s / (π_j + s)
z'_i = robust_center_y + [I - U diag(1-retention) U^T](z_i - mean_y)
```

- 无收缩强度、rank或门控扫描；超参数数为0。
- 每个旧类和新类使用完全相同的公式，不读取具体TX/class ID、old/new角色、receiver或scene。
- K≤2逐位恒等；FFT96/RF32不变；query不参与fit、不更新状态、不增加评分计算。
- 地面原型不进入query类分数，也不与target类做身份匹配；只读谱不写回，最终仍编译为单个INT8 affine head。

## 协议、数据与停止条件

- `protocol_schema=p2_min_v1`；复用D18 `VALIDATED_ONCE` capsule，receiver=`20-1`、seed=`713101`、K10（实际K8）、new5；不重建、不重验数据。
- 固定单次`LEO_weak`接收IQ；support-only适配；query逐样本面对全部注册类；无clean/source/query truth/role Oracle/类配额/global assignment/dense query graph。
- ground NPZ SHA256=`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`；manifest SHA256=`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`；组件仍为`UNVERIFIED`，结果只能是开发诊断证据。
- 成功门：相对D81，B/A/N/H/F/J、三场景、全部逐类和mean-row floors、old→new/new→old/new→new均不回退，且A/H/F、rain或新类至少一项严格改善。任一联合项回退即判负并停止确认seed/125。

## 本地版本与验证

- `E:\type10-7`根目录不是Git仓库；代码和报告在隔离Git worktree`E:\type10-7\code\snapshots\d81wt`开发，之后精确提交并cherry-pick到`E:\type10-7\github_publish\CVS-RFFI-repo`。
- 新增核心：`code/cvsrffi/stage2_d82_ground_nuisance_wiener_residual.py`，SHA256=`8e8870a6a00d99f4ce64f26deb177253bc9a1b12209fe26e5ebe47363db202d1`。
- 新增执行器：`code/scripts/probe_d82_ground_nuisance_wiener_residual.py`，SHA256=`a4c7c79953a6e5a4d180de4416a13e5473b644a24a09eb26e5c46299539f5f1e`。
- `ssr-gpu`环境：D82专项13/13 PASS；D62/D80/D81/D82相邻链48/48 PASS；`py_compile`与`git diff --check`PASS。

## 运行计划与资源

- 本地执行，不占N607 GPU；复用D18 capsule和runtime authorization，输出到本报告目录下`ground_nuisance_wiener_residual/`，启动前必须不存在。
- runtime root：`E:\type10-7\code\snapshots\d41wt`；probe root：`E:\type10-7\code\snapshots\d81wt`。
- capsule：`E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5`。
- ground：`E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component`。
- class binding：`analysis/d19_adv3b02_class_binding_20260717.json`，SHA256=`39cbf3355c221d604eb005624bffc2595cbdb3c499634274103c7663acb9740b`。
- 预期：105-row完整training log、30个target rows、`RECEIPT.json`、D82 metadata；params≤80k、epochs≤30、steps≤50、state≤256KB、query extra MAC=0。
- 已知风险：固定Wiener先验可能过度压缩target身份残差，从而同时降低旧类和新类；该风险只能由锁定query一次评估判断，不能结果后调整系数。

## 完成结果

状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。retry1完整运行105/105 rows，runner耗时138.13秒、外层墙钟147.1秒；stderr 0B、完整日志无Traceback/OOM/NaN/Inf。D82未通过相对D81联合门，不运行独立seed或125。

### 总体同row性能

|版本|B旧类注册前|A旧类注册后|N新类|H_old_new|F遗忘|J|min class B/A/N|mean-row floor B/A/N|old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D62|92.78%|82.22%|84.67%|82.6238%|10.56%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|23/8/15|
|D81|92.78%|82.78%|84.67%|82.9366%|10.00%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|22/8/15|
|D82|83.33%|72.22%|75.33%|72.7571%|11.11%|23.33%|70.00/56.67/53.33%|40.00/30.00/40.00%|34/24/13|
|D82−D81|−9.44pp|−10.56pp|−9.33pp|−10.1794pp|+1.11pp|−3.33pp|−10.00/+3.33/−20.00pp|−33.33/−20.00/−6.67pp|+12/+16/−2|

D82相对目标的缺口为`A −19.78pp`、`minA −31.33pp`、`new5 −16.67pp`。15/15 outer rows的预测hash均相对D81变化，属于系统性负迁移，不是单个fold噪声。

### 逐场景性能

|场景|版本|B|A|N|H|F|J|min class B/A/N|mean-row floor B/A/N|混淆old→new/new→old/new→new|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|clear|D81|98.33%|91.67%|98.00%|94.441%|6.67%|50.00%|90/70/90%|90/60/90%|2/1/0|
|clear|D82|90.00%|75.00%|84.00%|78.387%|15.00%|30.00%|70/50/60%|60/30/50%|10/3/5|
|low-elev|D81|91.67%|80.00%|76.00%|76.922%|11.67%|20.00%|80/60/50%|70/60/20%|7/5/7|
|low-elev|D82|80.00%|68.33%|78.00%|72.690%|11.67%|20.00%|70/40/50%|30/20/40%|12/8/3|
|rain|D81|88.33%|76.67%|80.00%|77.447%|11.67%|10.00%|60/30/70%|60/30/30%|13/2/8|
|rain|D82|80.00%|73.33%|64.00%|67.194%|6.67%|20.00%|70/50/40%|30/40/30%|12/13/5|

rain的旧类遗忘和after-old floor局部改善，但seen-new下降16pp、最低新类下降30pp、new→old增加11，联合性能仍显著恶化；不能用局部旧类改善掩盖注册失败。

### 逐类性能

|TX|角色|D81 B|D82 B|D81 A/N|D82 A/N|A/N变化|
|---|---|---:|---:|---:|---:|---:|
|14-10|旧类|96.67%|86.67%|93.33%|60.00%|−33.33pp|
|14-7|旧类|80.00%|83.33%|53.33%|63.33%|+10.00pp|
|20-15|旧类|96.67%|86.67%|90.00%|90.00%|0|
|20-19|旧类|93.33%|70.00%|93.33%|73.33%|−20.00pp|
|6-15|旧类|93.33%|83.33%|73.33%|56.67%|−16.67pp|
|8-20|旧类|96.67%|90.00%|93.33%|90.00%|−3.33pp|
|1-16|新类|—|—|93.33%|93.33%|0|
|1-18|新类|—|—|73.33%|53.33%|−20.00pp|
|18-10|新类|—|—|90.00%|66.67%|−23.33pp|
|14-11|新类|—|—|76.67%|83.33%|+6.67pp|
|8-3|新类|—|—|90.00%|80.00%|−10.00pp|

### 内部候选、训练、量化与资源

|candidate|B|A|N|H|F|min A|min N|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8|83.33%|72.22%|75.33%|72.76%|11.11%|56.67%|53.33%|D82目标，负|
|D42-USLDA-FP32-MATCHED|83.33%|72.22%|75.33%|72.76%|11.11%|56.67%|53.33%|与INT8 outer argmax一致|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22%|60.00%|40.00%|诊断baseline|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56%|63.33%|0.00%|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56%|0.00%|36.67%|旧类崩溃|
|ProtoNet-CDA/Z0|71.11%|48.33%|52.67%|48.97%|22.78%|13.33%|3.33%|弱baseline|

- 20步trace×15 rows，共300条训练记录；loss min/mean/max=`0.07560/0.30719/1.11738`，support accuracy=`89.58/98.77/100%`。support拟合良好没有转化为held query泛化。
- FP32/INT8的before与outer argmax变化均为0；margin sign flip=1，但最终预测仍一致；score绝对误差min/mean/max=`0.000768/0.001141/0.001683`。负结果不是量化造成。
- params=2,016、epochs=20、steps=20、state=34,011B、peak CUDA=22,886,912B、query=6,624 MAC且D82额外query MAC=0；dense query graph=0。ground统计90.52M MAC、support-Wiener 41.51M MAC、新增132.03M MAC、总适配25.023B MAC；资源上限均通过。

### 地面机制与数值审计

- 84个ground cells、rank14、保留79.7586%地面漂移谱；Wiener方向保留率固定为`0.2308..0.7395`，target类内地面方向能量实际只保留`16.54%..44.25%`。
- robust center位移最大范数`0.03989..0.06401`；有效样本数下界6.965；Wiener公式误差≤`3.34e-17`，中心公式误差≤`2.78e-17`，FFT96/RF32误差0；新旧类同式、query-free。
- block3协方差共进入1,080次D82数值修复审计；最大`jitter/lambda_max=6.54e-14`，属于机器精度量级，未形成可调正则。ground NPZ与manifest进出hash逐位不变。

### 证据文件

- training log：18,087,788B，SHA256=`eae94fdb8b06084af8a62eafb8bfa663502cf6f11bb439f55d5d9bb976c4baba`。
- metadata：8,232B，SHA256=`87814b4e39388783173577301cc56cf5027a3c4972dfa638d06cfa9221c68ff2`；receipt SHA256=`e3ed6c7a5e30991661f2ad7ba22f85ce26b0aed69054fb99922664a123ebf7f5`。
- 完整汇总：85,676B，SHA256=`daf5e32710638428596b994669854ccf4fa68622cfef91e7c508c8d9858fb326`；summary脚本SHA256=`2d7141a8c12262b0f237a1f2e254ea12b841ccc796d250265e551d6e294d1c8c`。

## D80-D82三轮技术复盘

已重新读取活动目标与`项目.md`，刷新conversation index至1,008条并搜索ground prototype/covariance/domain-adaptation路线；复核D80、D81、D82报告与三份105-row完整日志/汇总。

|版本|地面原型用法|同row结果|结论|
|---|---|---|---|
|D80|地面跨域common-mode协方差去噪|B/A略升，但N−0.67pp、F+0.56pp、min-N−3.33pp|直接改变target协方差且缺少稳健中心，旧/新失衡|
|D81|只用类中心化地面谱估计Cauchy稳健中心，类内协方差不变|两个seed均A+0.56pp、F−0.56pp、old→new−1，无联合回退|最强合法版本，但每seed只修正1个预测，覆盖率不足|
|D82|沿地面干扰方向收缩support残差|A−10.56pp、N−9.33pp、15/15 rows变化|路线淘汰；方差收缩使LDA逆协方差反而放大干扰方向|

复盘结论：地面压缩原型确实有正价值，但应作为“弱、只读、类无关的精度正则”，不能强融合身份中心，也不能只压support残差。下一轮D83的唯一主要差异应是：保留D81稳健中心，在D62共享协方差中沿`U`增加由target support总方差闭式定标的ground nuisance loading，使逆协方差降低这些方向权重；新旧类同式、K1恒等、无scan、query零额外路径。D83必须同时报告注册前后旧类、新类、逐类floor和遗忘；若N或任一floor回退立即停止。

## 启动异常与最小修复

- 封装尝试1在Python启动前失败：嵌套PowerShell变量被外层提前展开；输出目录不存在、query未打开、无性能数据。改为单层PowerShell，实验参数不变。
- 封装尝试2在authority preflight fail-closed：误把`apply_staging_authority.json`传给`--before/after-formal-policy`，其中本地路径字段被runtime正确拒绝；输出目录不存在、query未打开、无性能数据。改用D18的path-free`formal_execution_policy.json`，seal和签名授权不变。
- 尝试3通过authority并进入第一个support-only fit，但在query评分前因`D43 structured covariance is not positive definite`停止；只产生不完整输出，不能报告性能。根因是Wiener残差压缩使block3协方差出现机器舍入量级的非正定漂移。
- 最小修复只对D82的block3协方差启用闭式机器精度SPD修复：`jitter=max(0,d·eps·lambda_max-lambda_min)`；若负能量超过`sqrt(eps)·lambda_max`仍fail-closed。该修复参数数0、不读held/query、不扫描、不改变D81/D62。
- 修复后probe SHA256=`0a3233e602c28f6cd14b2dbe78fb3a1fc73f047bd8011c8387e15b1a822f1c27`；D82专项14/14、D62/D80/D81/D82相邻链49/49 PASS，`py_compile`与`git diff --check`PASS。下一次输出使用`ground_nuisance_wiener_residual_retry1/`，保留失败目录不覆盖。
