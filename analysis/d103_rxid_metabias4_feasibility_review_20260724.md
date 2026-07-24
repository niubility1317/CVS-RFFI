# D103-RXID-Episodic-MetaBias4-qKNN可行性复核

状态：`FEASIBILITY_PROBE_COMPLETE / K1_MECHANICS_INVALID_ZID_MISBOUND / RESOURCE_SHAPE_ONLY`

日期：2026-07-24

## 1.边界

本次只使用冻结D102 Phase1 source-held tap：

- tap SHA256：`c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1`；
- 8,400个互异物理样本；
- 7个source receiver、4个day、6个TX；
- 固定inner held receiver=`1-1`、episode receiver=`1-19`；
- outer结果读取=`false`；
- target/capsule/formal query访问=`false`；
- BA、TX probe准确率、LOCO性能均未计算；
- 未保存`U/B/bank`或deployment bundle。

微探针代码在本地Git commit`2317e321`后执行，没有上传或访问N607。

## 2.可构造性

|检查|结果|裁决|
|---|---:|---|
|每receiver覆盖day|4/4|通过|
|每receiver覆盖TX|6/6|通过|
|每receiver×TX跨day可构造|42/42|通过|
|balanced batch|288行|通过|
|每receiver×day×TX物理样本|2|通过|
|balanced batch物理ID唯一|是|通过|
|K1 support物理样本|6个，6类各1个|通过|
|view增加K|0|通过|
|K1机械rank|4|无效：`z_id`误绑定为`z_dom`|
|K1机械最小奇异值|1.187441|无效：`z_id`误绑定为`z_dom`|
|K1机械condition|1.165228|无效：`z_id`误绑定为`z_dom`|

微探针代码把tap成员`z_id`装入了名为`z_dom`的字段；真实`z_dom`只存在于需逐row绑定的dual archive。因此上述K1数值不能证明D103声明的domain输入可形成有效系统，全部从可行性结论撤回。修复版必须hash绑定tap和dual archive，并验证physical ID、metadata和`z_id`逐字节parity后再使用真实`z_dom`。

## 3.资源实测

设备为RTX5070Ti，PyTorch2.10.0+cu128。代表meta step误用了160维`z_id`代替160维`z_dom`；因为shape相同，下列耗时和显存仅保留为160维代表计算的资源近似，不是正确D103链路实测。该step同时执行：

- rank-5 TX线性零空间与32维row-orthogonal encoder；
- 288行balanced batch；
- 多尺度RBF MMD；
- 跨day/跨TX同receiver contrastive；
- VICReg；
- K1/K5/K10三个MetaBias4+qKNN episode；
- forward、backward和Adam step。

|字段|实测|
|---|---:|
|warmup|3step|
|计时|3step|
|计时总长|0.830749s|
|平均/meta step|0.276916s|
|峰值allocated|22,796,800B|
|峰值reserved|27,262,976B|
|临时学习参数|5,976|
|warmup/timed loss|全部finite|

## 4.完整流程资源外推

冻结训练计划为每fit20epoch×20meta step=400step。每个outer fold含4个leave-one-day fit和1个outer fit：

- `(7 receiver outer+42 receiver×class双留出outer)×5=245fit`；
- 最终全source重训1fit；
- 合计246fit、98,400meta step；
- 按本机实测为7.569GPUh；
- 采用3倍N607设备/实现安全因子后为22.707GPUh；
- 再加35%的双probe、量化、M0/D102 matched评估、I/O和失败artifact开销，合计30.655GPUh。

建议冻结上限：

|资源|上限|
|---|---:|
|总GPU时|36GPUh|
|单fit峰值显存|4GiB|
|完整run-root磁盘|20GiB|
|Phase2 state|<80KiB|
|post-backbone MAC/query|≤262,144|

该外推只用于资源可行性；不能证明N607实际耗时、收敛或性能。正式训练若超上限，必须以`NO_PERFORMANCE_RESULT`封口，不能减少fold、K、epoch或LOCO覆盖换取完成。

## 5.结论

|问题|结论|
|---|---|
|跨day/跨TX正对可构造吗|是|
|真实`z_dom`的K1系统可形成rank-4吗|未知，原证据无效|
|本地GPU资源是否明显不可行|否|
|D103是否已有性能证据|否|
|是否可进入N607|否|
|是否可打开Target25|否|
|是否可冻结设计|待独立复核修订稿、数值门和资源外推|

本次结论为`FEASIBILITY_PROBE_COMPLETE / K1_MECHANICS_INVALID_ZID_MISBOUND / RESOURCE_SHAPE_ONLY`。D102仍是`PHASE1_HELD_FALSIFIER_REJECT`；D103 Revision2为`P0=0、P1=6 / NO_GO_TO_DESIGN_FROZEN / TARGET25_NO_GO`。
