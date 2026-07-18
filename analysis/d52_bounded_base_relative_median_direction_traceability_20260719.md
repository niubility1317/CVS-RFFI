# D52有界base-relative median方向追踪

|需求|设计位置|验证|状态|
|---|---|---|---|
|继承D45稳定底座|先拟合D45，再加唯一系数修正|D45–D52联合116项、D40–D52闭包256项|VERIFIED_PRE_RUN|
|保留D51有效方向|`coordinate_median(x_c)-mean(x_c)`|rank不变、class等变、outlier定向测试|VERIFIED_PRE_RUN|
|消除小RMS放大|`||DeltaW_c||=(1-rho_c)||W_c-mean(W)||`|逐类范数等式与finite测试|VERIFIED_PRE_RUN|
|无可调尺度|无alpha、threshold、clip、scan|公式常量与audit字段测试|VERIFIED_PRE_RUN|
|K1/K2回退|D45精确fallback|K1/K2参数化测试|VERIFIED_PRE_RUN|
|协议闭合|support-only、单affine、无role/query分支|运行后105行closure复算|PENDING_RUN|
|量化和资源|int8系数＋FP16截距，新增资源单列|运行后artifact verifier|PENDING_RUN|
|完整性能|总体、场景、逐类、15fold、训练、混淆、量化、资源|完成后生成full summary与报告|PENDING_RUN|

边界：D52不是D51系数扫描，也不使用场景、receiver、类ID、old/new角色或query选择修正尺度。`coordinate median`只在冻结坐标系内定义，不声称旋转等变。当前仅为开发探针，未达到formalization或125准入。
