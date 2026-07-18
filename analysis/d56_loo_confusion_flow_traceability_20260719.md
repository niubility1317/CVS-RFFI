# D56 LOO混淆流平衡追踪

|需求|实现|验证|状态|
|---|---|---|---|
|D46底座|继承类级LOO full/block融合|D56＋D46回归23/23|VERIFIED_PRE_RUN|
|support混淆图|inner-held独立argmax生成`y→p`|分区、rank覆盖、图重算|VERIFIED_PRE_RUN|
|零和修正|`Delta b=(out-in)/(K*C)`|边流守恒与数值和为0|VERIFIED_PRE_RUN|
|类对称|无class ID/role/scene/receiver分支|类标签置换测试|VERIFIED_PRE_RUN|
|K1/K2|精确D46 fallback|参数化测试|VERIFIED_PRE_RUN|
|query与数据协议|query/clean/source不可达，复用固定capsule|receipt/resource audit|VERIFIED_COMPLETE|
|完整性能|7候选、3场景、11类、15fold、训练/量化/资源/artifact|summary/report第9—16节|VERIFIED_COMPLETE|
|性能判定|after83.33%、new80.67%、H80.95%、forget8.33pp|min-new60%，D46同折5/15变化|DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE|

D56是support-only、强制nonpromotable开发探针；完整结果显示old改善以new退化为代价，没有formal/125权限，也不扫描流强度。
