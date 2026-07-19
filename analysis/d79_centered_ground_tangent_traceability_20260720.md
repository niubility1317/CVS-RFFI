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
|D79-R1|复用D78真实ground tangent，84 cell、14有效域、rank13、只读|D79 core/probe|verified|105行真实run中84 cell、14有效域、rank13；ground NPZ/manifest入口出口SHA完全一致|
|D79-R2|全support均值中心化，`mean_i z_i=0`，类/row置换不变|`code/cvsrffi/stage2_d79_centered_ground_tangent.py`|verified|精确中心、确定性与全局特征平移不变测试通过|
|D79-R3|编译`Delta b=−DeltaW mu`，均值点残差logit严格为0|core与probe|verified|直接中心式/单仿射等价、均值点零logit、K1双零测试通过|
|D79-R4|D78优化完全冻结：20步、rank13、同目标/trust、无扫描|core wrapper、source lock|verified|30个target fit、20步、rank13；全部残差命中相同trust radius；无参数扫描|
|D79-R5|资源：40step、20epoch、<80k、含ground<256KB、query额外MAC/state0|probe/artifact|verified|2,159参数、40step、20epoch、34,011B；query额外MAC/state0|
|D79-R6|协议：support-only、全类对称，无clean/source/query truth/role/quota/global assignment|probe/RECEIPT|verified|RECEIPT SHA=`347a82be...`；105行、query0、协议违规计数全0|
|D79-R7|完整开发实验：20-1/new5/K10/713101、3场景×5fold、105行|run/summarizer|verified|完整解析105行、全部stdout/stderr、逐类/场景/混淆/量化/资源|
|D79-R8|晋级门：相对D62，`A/N/H/min-A/min-N`不退化、`F`不升且至少一项严格改善；无混淆交换|summarizer/report|rejected|`A+2.22pp`但`N−2.00pp`、min-N`−3.33pp`，new→old`+3`；不启seed2/125|
|D79-R9|formal ground bundle需联合封存及外部authority签名|loader/report|blocked|当前只能development diagnostic|

## 停止条件

不扫描中心化比例、bias倍率、rank、温度、步数、trust radius或类权重。D79仍产生old/new交换伤害，最终记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，关闭ground-tangent边界路线。追溯状态计数：7项`verified`、1项`rejected`、1项`blocked`。
