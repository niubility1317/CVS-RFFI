# D53谱收缩median transport追踪

|需求|实现|验证|状态|
|---|---|---|---|
|D45底座|先拟合D45，再加单affine修正|联合回归|VERIFIED_PRE_RUN|
|稳健位移|`U=coordinate_median-mean`|rank不变、outlier测试|VERIFIED_PRE_RUN|
|判别几何映射|`G=U M0^T/||M0||2^2`，`DeltaW=diag(gamma)GW0`|class置换等变、谱界测试|VERIFIED_PRE_RUN|
|无可调尺度|谱范数由support唯一决定|无pinv/ridge/alpha/threshold/clip|VERIFIED_PRE_RUN|
|K1/K2回退|在谱退化检查前精确D45 fallback|K1/K2及退化类均值测试|VERIFIED_PRE_RUN|
|协议闭包|support-only、无角色/query分支|105行closure、访问项0/false|VERIFIED|
|完整性能|总体、场景、逐类、15fold、训练、量化、资源|summary与报告第8–14节|VERIFIED|

边界：class-class transport只用于support系数构造，不是query图、全局assignment或quota；coordinate median不声明旋转等变。D53是开发探针，不具有formal/125权限。

完成结果：before/after/new/H=`92.22/81.67/83.33/81.28%`，forget`10.56pp`、joint`23.33%`、min-after/new`53.33/73.33%`、混淆`26/8/17`。相对D46，after/forget/floor持平，但new`-1.33pp`、H`-1.05pp`；5/15预测变化。谱收缩把final correction L2降至mean`0.1248`、max`0.7794`，却没有形成联合收益。最终`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
