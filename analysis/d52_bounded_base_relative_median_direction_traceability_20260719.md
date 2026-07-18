# D52有界base-relative median方向追踪

|需求|设计位置|验证|状态|
|---|---|---|---|
|继承D45稳定底座|先拟合D45，再加唯一系数修正|D45–D52联合116项、D40–D52闭包256项|VERIFIED_PRE_RUN|
|保留D51有效方向|`coordinate_median(x_c)-mean(x_c)`|rank不变、class等变、outlier定向测试|VERIFIED_PRE_RUN|
|消除小RMS放大|`||DeltaW_c||=(1-rho_c)||W_c-mean(W)||`|逐类范数等式与finite测试|VERIFIED_PRE_RUN|
|无可调尺度|无alpha、threshold、clip、scan|公式常量与audit字段测试|VERIFIED_PRE_RUN|
|K1/K2回退|D45精确fallback|K1/K2参数化测试|VERIFIED_PRE_RUN|
|协议闭合|support-only、单affine、无role/query分支|105行closure复算，query/role等0/false|VERIFIED|
|量化和资源|int8系数＋FP16截距，新增资源单列|量化0/0/0，额外227,520 MAC-equivalent|VERIFIED|
|完整性能|总体、场景、逐类、15fold、训练、混淆、量化、资源|summary及报告第9–18节|VERIFIED|

边界：D52不是D51系数扫描，也不使用场景、receiver、类ID、old/new角色或query选择修正尺度。`coordinate median`只在冻结坐标系内定义，不声称旋转等变。当前仅为开发探针，未达到formalization或125准入。

完成结论：D52总体before/after/new/H为`90.56/81.67/80.00/79.96%`，forget`8.89pp`、joint`26.67%`、min-after/new`66.67/66.67%`、混淆`19/15/15`。相对D46，min-after`+13.33pp`且forget`-1.67pp`，但new`-4.67pp`、H`-2.37pp`、min-new`-6.67pp`、new→old`+7`。最终`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

机制结论：预注册范数等式闭合到`1.33e-15`，但final correction L2平均/最大`1.149/3.128`，仍高于D51的`0.736/2.508`。base-relative上界消除了小RMS除法，却未提供保守决策尺度；下一路线应把median位移映射到D45判别几何，而非再对修正系数扫描或clip。
