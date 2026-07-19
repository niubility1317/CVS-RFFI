# D83地面干扰谱精度加载

状态：`COMPLETED_DIAGNOSTIC_NEUTRAL_NO_GAIN_NOT_PROMOTABLE`。实验ID：`d83_ground_precision_loading_20260720`；时间：2026-07-20 06:15 HKT；操作者：Codex。

目标是修复D82“压小support方差后，LDA逆协方差反而放大干扰方向”的机制错误。D83保留D81的类对称一步Cauchy稳健中心，类内残差不变；随后仅在每个D62 full/block support-only fit的共享协方差中加入rank14地面干扰loading：

```text
tau = trace(Sigma_target_z) / 160
L = (rank * tau / K) * U diag(pi) U^T
precision = inverse(Sigma_target + L)
```

该式使14个地面干扰方向的平均loading恰为target平均方差的`1/K`；K≤2不加载，K1严格回到基线。无超参/强度/rank扫描，新旧类同式，无class ID/role/scene分支，query不更新且额外评分MAC为0。与D80的主要差异是只加载D81筛出的低秩干扰谱，不按ground域自由度强混合完整协方差。

开发单元锁定`rx20-1/seed713101/K10(actual K8)/new5/3场景×5fold`，复用D18 `VALIDATED_ONCE` capsule、runtime authorization和D22只读84-cell int8 ground组件；不重建、不重验数据。ground仍为`UNVERIFIED`，所以只能形成开发诊断证据。成功门：相对D81的B/A/N/H/F/J、全部场景、逐类与mean-row floors、三类混淆均不回退，且A/H/F、rain或新类至少一项严格改善；否则立即判负，不启确认seed/125。

本地Git worktree：`E:\type10-7\code\snapshots\d81wt`；修复后核心SHA256=`5ed98a00d098c51c079f6c3f77b1c02f2328edc11730fc791ce456d912d56b1d`；probe SHA256=`e6626edd22a745747ed09a752692c331fdfa42db2fad94e6c337135a59f59f4a`。首次预注册验证为专项12/12、D62-D83相邻链61/61 PASS；状态兼容修复后D42/D62/D80-D83相邻链85/85 PASS，并增加真实`_compile_state`集成断言，`py_compile`与`git diff --check`PASS。

本地RTX5070Ti执行，不占N607。输出：本报告目录`ground_precision_loading/`；预计105-row日志、receipt、metadata、完整逐类/场景/资源汇总。资源门仍为params≤80k、epochs≤30、steps≤50、state≤256KB、dense query graph=false。

## 首次执行失败记录

2026-07-20 06:15 HKT首次真实运行在首个fold的support拟合后、`_compile_state`阶段fail-closed，`stdout.log`为空，`stderr.log`记录`D42UnifiedShrinkageLDAError: D42 state drift`，没有完整候选、query评分或性能指标，因此不得作性能结论。根因是D83把机制描述`sklearn_lsqr_auto_plus_rank14_ground_loading`写入D42封闭的注册状态存储策略字段`covariance_policy`；该字段仅接受基础解码策略，不代表D83加载机制。最小修复保持`covariance_policy=sklearn_lsqr_auto_shrinkage_equal_prior`，并新增独立审计字段`d83_covariance_policy=sklearn_lsqr_auto_plus_rank14_ground_loading`。公式、support输入、预注册判据和query边界均不改变。原失败目录永久保留，修复后的执行必须写入`ground_precision_loading_retry1/`。

## 完成结果

状态：retry1完整运行105/105 rows，runner耗时111.30秒，外层进程约118秒；`stdout_retry1.log`和`stderr_retry1.log`均为0B，完整日志无Traceback/OOM/NaN/Inf。D83对D81没有任何预测或指标增益，未通过“至少一项严格改善”的预注册门，不运行独立seed或125。

### 总体同row性能

|版本|B旧类注册前|A旧类注册后|N新类|H_old_new|F遗忘|J|min class B/A/N|mean-row floor B/A/N|old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|D62|92.78%|82.22%|84.67%|82.6238%|10.56%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|23/8/15|
|D81|92.78%|82.78%|84.67%|82.9366%|10.00%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|22/8/15|
|D83|92.78%|82.78%|84.67%|82.9366%|10.00%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|22/8/15|
|D83−D81|0|0|0|0|0|0|0/0/0|0/0/0|0/0/0|

D83相对项目K10/new5目标的缺口仍为`A −9.22pp`、`minA −34.67pp`、`new5 −7.33pp`。D83与D81的15/15 outer prediction hash完全相同；相对D62只在low-elev fold0修正1个旧类预测，因此D83的全部可见收益仍来自D81稳健中心，而不是新增precision loading。

### 逐场景性能

|场景|版本|B|A|N|H|F|J|min class B/A/N|mean-row floor B/A/N|混淆old→new/new→old/new→new|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
|clear|D81/D83|98.33%|91.67%|98.00%|94.441%|6.67%|50.00%|90/70/90%|90/60/90%|2/1/0|
|low-elev|D81/D83|91.67%|80.00%|76.00%|76.922%|11.67%|20.00%|80/60/50%|70/60/20%|7/5/7|
|rain|D81/D83|88.33%|76.67%|80.00%|77.447%|11.67%|10.00%|60/30/70%|60/30/30%|13/2/8|

三种场景的均值、逐类最低值、mean-row floors和三类混淆均逐项相等；没有用局部场景变化掩盖总体中性结果。

### 逐类性能

|TX|角色|D81 B|D83 B|D81 A/N|D83 A/N|变化|
|---|---|---:|---:|---:|---:|---:|
|14-10|旧类|96.67%|96.67%|93.33%|93.33%|0|
|14-7|旧类|80.00%|80.00%|53.33%|53.33%|0|
|20-15|旧类|96.67%|96.67%|90.00%|90.00%|0|
|20-19|旧类|93.33%|93.33%|93.33%|93.33%|0|
|6-15|旧类|93.33%|93.33%|73.33%|73.33%|0|
|8-20|旧类|96.67%|96.67%|93.33%|93.33%|0|
|1-16|新类|—|—|93.33%|93.33%|0|
|1-18|新类|—|—|73.33%|73.33%|0|
|18-10|新类|—|—|90.00%|90.00%|0|
|14-11|新类|—|—|76.67%|76.67%|0|
|8-3|新类|—|—|90.00%|90.00%|0|

### 七候选、训练与量化

|candidate|B|A|N|H|F|min A|min N|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8|92.78%|82.78%|84.67%|82.94%|10.00%|53.33%|73.33%|D83目标，中性无增益|
|D42-USLDA-FP32-MATCHED|92.78%|82.78%|84.67%|82.94%|10.00%|53.33%|73.33%|与INT8 outer argmax一致|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22%|60.00%|40.00%|诊断baseline|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56%|63.33%|0.00%|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56%|0.00%|36.67%|旧类崩溃|
|ProtoNet-CDA/Z0|71.11%|48.33%|52.67%|48.97%|22.78%|13.33%|3.33%|弱baseline|

- 20步trace×15 rows，共300条训练记录；loss min/mean/max=`0.07560/0.30719/1.11738`，support accuracy=`89.58/98.77/100%`。
- FP32/INT8的before、support和outer argmax变化均为0，margin sign flip=0；score绝对误差min/mean/max=`0.000347/0.000920/0.001954`。中性结果不是量化抹平造成。

### 地面机制、数值与资源

- 84个ground cells、rank14、覆盖79.7586%地面漂移谱；实际执行1,080个full/block component fits和2,160次support-center transforms。ground NPZ/manifest进出hash逐位不变。
- 30个outer before/final fits的`loading/target mean variance`固定为`1/K=0.125`；loading trace范围`0.44560..0.50204`，每个保留方向平均loading范围`0.03183..0.03586`。后验最小特征值`1.92e-6..4.83e-6`，方程残差≤`1.11e-15`，数值稳定。
- 新增loading确实改变了连续头：相对D81，before系数最大绝对差的15-row min/mean/max=`0.00106/0.56388/3.20022`，final为`0.000869/0.006283/0.017124`；三类最小margin在15/15 rows均发生微小变化，但最终argmax完全不变。缺陷不是“机制未执行”，而是D62融合后有效扰动过弱且方向不命中剩余错误。
- params=2,016、epochs=20、steps=20、state=34,011B、peak CUDA=22,886,912B、query=6,624 MAC且D83额外query MAC=0；dense query graph=0。ground统计90.52M MAC、稳健中心21.89M MAC、协方差loading 1.84M MAC，新增114.26M MAC、总适配25.005B MAC。资源门均通过，但相对D81增加计算而零预测收益，效率判负。
- 公式对所有新旧类相同，不读class ID、old/new角色、receiver、scene、held/query；query/clean/source/role/quota/global assignment访问均为0。ground组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`且`formal_phase2_eligible=false`，所以即使有增益也只能是开发诊断证据。

### 缺陷与下一轮决策

D83证伪了“把地面谱作为统一弱协方差loading即可扩大D81收益”的假设。统一loading改变连续分数，却被D62的classwise融合和argmax margin吸收，额外114.26M MAC没有换来一个预测变化；继续放大loading会重新逼近D80/D82的系统性负迁移风险，不能在同一query结果后调强度。D81仍是当前最强合法开发版本。

下一轮应停止继续改全局协方差，转向更高信息密度的地面原型使用：只从ground域×类中心提取“跨域位移的方向一致性/不确定度”，再用target support交叉拟合证据决定是否对D81的候选类边界施加极稀疏、类对称的margin校正。必须预注册为无扫描、K1恒等、support-only；只有在不回退N和任一floor的前提下扩大D81的预测覆盖才可晋级。

### 证据与版本

- training log：17,751,707B，SHA256=`a1771d71bbe5f1b6167599237d931515a6a3da8238aebfbbc8265482965d5d24`；receipt状态`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，105 rows。
- 完整汇总：`d83_full_performance_summary.json`，SHA256=`eefa0f7a2060c2c3629255badfc2768c6ce2b6c6d3631a2679fe0cdb9cae6b85`；汇总器SHA256=`31d010dd57593f5a06486a7fbc66b94e8856797d5786c843e3e46ee0b38e1238`；D83 metadata、receipt、support/resource/geometry audit均在`ground_precision_loading_retry1/`。
- 状态兼容修复提交：worktree`fb77ce21`，发布仓`7dae6dc7`。完整汇总脚本为`code/scripts/summarize_d83_performance.py`，以实际执行验证全日志。
