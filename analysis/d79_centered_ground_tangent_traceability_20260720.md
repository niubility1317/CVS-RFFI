# D79中心化地面切向旋转追溯

## 方法定位与唯一变化

D78的13维ground tangent使rain场景旧类`A+6.67pp`、min-A`+30pp`，证明地面压缩原型的跨坐标域方向有效；但它同时使rain新类`N−8pp`、min-N`−20pp`，混淆由4次old→new等量交换为4次new→old。D78的`DeltaW x`没有对target域均值去中心，因而低秩旋转含有类相关常量项。

D79唯一改变为：令全注册类support的变换后均值为`mu`，D78优化中的切向特征改为

`z=(x−mu)U`，

并把学习到的`DeltaW=A U^T`连同

`Delta b=−DeltaW mu`

一起编译，使部署修正严格为`DeltaW(x−mu)`。因此在target support均值处每类残差logit都为0，只保留域切向旋转，不改变全局类别先验。`mu`由全部注册类等K support构成，类置换不变；不读取old/new角色或query。

除中心化与bias补偿外，D78全部锁定：同一84-cell只读ground组件、rank13、8物理rank/88 held、固定top rival、初始类均值温度、smooth worst-class top-2 loss、20接受步、相同trust ball、D62 final rows、INT8/FP32 matched和单仿射部署。

## 需求到实现追溯表

|ID|要求|目标文件|状态|验证/停止条件|
|---|---|---|---|---|
|D79-R1|复用D78真实ground tangent，84 cell、14有效域、rank13、只读|D79 core/probe|specified|入口/出口SHA、projector SHA与D78一致|
|D79-R2|全support均值中心化，`mean_i z_i=0`，类/row置换不变|`code/cvsrffi/stage2_d79_centered_ground_tangent.py`|specified|合成精确中心、置换与确定性测试|
|D79-R3|编译`Delta b=−DeltaW mu`，均值点残差logit严格为0|core与probe|specified|FP64/FP32误差、单仿射等价测试|
|D79-R4|D78优化完全冻结：20步、rank13、同目标/trust、无扫描|core wrapper、source lock|specified|D78依赖SHA与audit差异锁|
|D79-R5|资源：40step、20epoch、<80k、含ground<256KB、query额外MAC/state0|probe/artifact|specified|bias补偿仅增加`C×D`MAC上界|
|D79-R6|协议：support-only、全类对称，无clean/source/query truth/role/quota/global assignment|probe/RECEIPT|specified|禁止访问全0|
|D79-R7|完整开发实验：20-1/new5/K10/713101、3场景×5fold、105行|run/summarizer|pending|完整日志、逐类/场景/混淆/量化/资源|
|D79-R8|晋级门：相对D62，`A/N/H/min-A/min-N`不退化、`F`不升且至少一项严格改善；无混淆交换|summarizer/report|pending|失败即关闭，不开seed2/125|
|D79-R9|formal ground bundle需联合封存及外部authority签名|loader/report|blocked|当前只能development diagnostic|

## 停止条件

不扫描中心化比例、bias倍率、rank、温度、步数、trust radius或类权重。若D79仍产生old/new交换伤害或没有严格联合改善，记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并关闭当前ground-tangent边界路线。
