# D103-R1-RXID-DUALSPLIT-MB4可行性复核

状态：`FEASIBILITY_PROBE_COMPLETE / NON_PERFORMANCE / INDEPENDENT_REVIEW_PENDING`

日期：2026-07-24

## 1.版本与边界

|字段|值|
|---|---|
|candidate|`D103-R1-RXID-DUALSPLIT-MB4`|
|重入预注册commit|`dc672f8b`|
|tap SHA256|`c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1`|
|dual SHA256|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|固定inner held receiver|`14-7`|
|固定episode receiver|`18-2`|
|outer结果读取|`false`|
|target/capsule/formal query访问|`false`|
|BA/TX probe/LOCO性能计算|`false`|
|deployment asset保存|`false`|

R1代码先验证tap/dual的physical ID、label、receiver、day和class IDs逐数组相等，再验证`z_id max_abs≤1e-5`，最后使用dual中的真实`z_dom`。`tap_dual_row_binding_verified=true`、`z_id_parity_verified=true`。

dual archive只从N607已有只读run-root取回；远端和本地SHA256一致，大小6,449,034B。SCP结束后本地`ssh.exe=0`、N607 TCP22 ESTABLISHED=0。没有启动或修改N607任务。

## 2.可构造性和K1机械检查

|检查|结果|语义|
|---|---:|---|
|source-val row|8,400|仅资源/shape|
|物理ID唯一|8,400/8,400|通过|
|receiver/day/TX|7/4/6|通过|
|每receiver×TX跨day可构造|42/42|通过|
|balanced batch|288行|每cell2个互异物理样本|
|K1 support|6个物理样本、6类各1个|view不增加K|
|真实`z_dom`的K1 rank|4|机械门通过|
|真实`z_dom`的min singular value|1.073563|高于预注册0.05|
|`condition(Λ0+A_data)`|1.289306|低于预注册10|

这些数值只证明正确输入链路下4维系统可以构造且数值非奇异，不证明K1统计可识别、泛化或性能。prior fraction、系数活动、view稳定性和独立OOF尾部仍必须由正式source-held证伪。

## 3.资源实测

设备为RTX5070Ti，PyTorch2.10.0+cu128。代表meta step包含rank-5线性TX零空间、32维encoder、288行balanced batch、多尺度MMD、跨day/cross-TX receiver contrastive、VICReg、K1/K5/K10三个episode、forward/backward和Adam。

|字段|实测|
|---|---:|
|warmup/计时|3/3step|
|计时总长|0.770438s|
|平均/meta step|0.256813s|
|峰值allocated|22,796,800B|
|峰值reserved|27,262,976B|
|inner-fold学习参数shape|5,976|
|代表零值临时checkpoint|77,168B|
|临时目录峰值|77,168B|
|临时文件返回前删除|是|
|临时state含学习值|否|
|loss finite|全部|

## 4.完整流程外推

R1已删除重复的inner leave-one-receiver；7个receiver outer本身承担receiver-held审计。每个outer只增加4个leave-one-day稳定性fit：

`(7 receiver outer+42 receiver×class双留出outer)×5fit+1 final fit=246fit`。

每fit400step，共98,400step：

|计算|结果|
|---|---:|
|本机实测GPUh|`98,400×0.2568127/3600=7.019548`|
|3倍N607安全因子|21.058643GPUh|
|再加35%完整流程开销|28.429168GPUh|
|按下一个6GPUh取整|30GPUh|

冻结资源候选上限：

|资源|上限|依据|
|---|---:|---|
|总GPU时|30GPUh|预注册公式向上取整|
|单fit峰值显存|4GiB|大于实测reserved8倍且满足至少4GiB|
|完整run-root磁盘|20GiB|大于`246×77,168B`且满足至少20GiB|
|Phase2 state|<80KiB|设计门|
|post-backbone MAC/query|≤262,144|项目活动资源门|

30GPUh只是冻结run上限，不是N607实际耗时承诺。双probe、量化、M0/D102 matched评估、I/O和失败artifact必须全部计入；超限即`NO_PERFORMANCE_RESULT`，不得减少fold、K、day审计或LOCO覆盖。

## 5.Phase1 split与声明

微探针使用source-val只做资源/shape且不保存资产。正式实现必须：

- `L_s=0.07`独占TX监督、`P⊥`、TX-MMD、带TX标签meta和类平衡bank；
- `U_s=0.63`不读取或恢复TX，只进入receiver/day自监督和VICReg；
- source-val=`0.30`完全不进梯度、资产或量化，只在状态冻结后证伪；
- source-held old/new/H只标匿名生命周期proxy，不是实际`Y_new`或Stage2-C证据。

## 6.结论

|问题|裁决|
|---|---|
|tap/dual真实`z_dom`链路是否闭合|是|
|跨day/cross-TX正对是否可构造|是|
|K1机械系统是否数值非奇异|是，仅机械证据|
|资源是否明显不可行|否|
|是否已有性能证据|否|
|是否可接N607正式训练|否，等待设计独立终审和正式实现|
|是否可打开Target25|否|

当前只声明`FEASIBILITY_PROBE_COMPLETE / NON_PERFORMANCE`。
