# D7a类条件固定接收IQ representation head追踪

日期：2026-07-17

## 实现边界

D7a只允许三个预登记operator：`base`、`dc_rms`、`dc_rms_spec15`。第三项对已经DC/RMS处理的固定接收IQ执行15%轻度频谱幅度收缩并保留相位；没有CFO估计或去旋，也不调用LEO信道模拟器。三个view共享同一物理sample ID和父接收IQ SHA，不增加K。2026-07-17阻断修复后，fit/register不再接受普通预提取feature mapping，只接受由模块从固定`received_iq`逐样本构建的validated artifact；每个view记录并校验`physical_sample_id`、`parent_received_iq_sha256`、`operator_id`、固定`view_seed=0`和feature SHA，payload与binding由artifact seal闭合。

每个注册类分别使用该类物理support的leave-two-out证据选择operator。候选必须同时满足：

- 该类删除法准确率不低于base；
- 一类替换后的support总体准确率相对base下降不超过1pp。

所有类选择组合后还必须通过总体与最低类非退化门禁；否则整个scenario回退all-base。

## 跨operator校准

每个operator只从注册support计算一个统一的off-class cosine中心和尺度。类分数为：

```text
(cosine(query_operator_feature, class_operator_prototype) - operator_center)
/ operator_scale
```

query仅计算当前registry实际使用到的operator，最多3次固定IQ representation前向；模块对每个物理query逐样本、逐operator调用feature extractor，callback不会同时接收其它query行，随后逐样本在全部注册类上argmax。没有query拟合、角色Oracle、类别quota、batch-global assignment或模块内q-q图。

## 注册前后

after保持before旧类operator、prototype和operator calibration bitwise不变，只为registry中缺失的新类执行support-onlyoperator选择和原型注册。新旧类别由registry成员关系判断，不存在old/new角色参数。

K1/K5不再重复运行operator选择。新增锁定策略入口要求先从恰好K10物理support锁定operator/calibration，再在K1/K5对应物理support上只重建prototype；operator与calibration保持不变。该入口使K1可以运行，但不改变K10开发选择、K1/K5仅压力评估的协议边界。

## 资源

闭式0epoch，adapter可训练参数为0。持久状态仅包含每类一个prototype、operator ID和三个operator校准标量；运行时强制低于256KB。query无dense graph，backbone/representation view数等于registry中operator去重数。

## 证据边界

模块和直接测试覆盖实际IQ operator、validated artifact seal与逐view绑定、随机feature篡改拒绝、leave-two-out类条件选择、组合非退化回退、跨operator support校准、query逐样本按需view、batch-coupled callback隔离、K10锁定后的K1/K5 prototype-only重建、after旧状态锁定和单观测lineage fail-closed。

当前`receiver=20-1,seed=713101`的40样本/类/scenario已经全部分配给K10 support、原20条query和后续10条fresh query，均已被评分。D7a不复用这些query声称独立性能。本轮只允许对注册support产生operator锁定诊断；`20-19`和`1-18`需要在support映射中单列，但真实性能floor必须等待新的未评分development seed。

## 真实K10 support锁定诊断

只打开before/after enrollment support与sealed runtime，未打开apply/query、既有prediction、score或truth。三个scenario的before组合最低类/总体非退化门均通过。operator使用情况及状态：

|scenario|before使用operator|after使用operator|before状态|after状态|
|---|---|---|---:|---:|
|clear|base|base/DC-RMS/spec15|7,368B|13,508B|
|low|base/DC-RMS/spec15|base/DC-RMS/spec15|7,390B|13,521B|
|rain|base/spec15|base/spec15|7,404B|13,542B|

adapter训练参数为0，query最多3个实际representation view。

重点floor类support leave-two-out：

|类|clear|low|rain|
|---|---|---|---|
|`20-19`|base：60%→60%|DC/RMS：70%→90%|spec15：70%→70%|
|`1-18`|spec15：40%→50%|base：20%→20%|base：60%→60%|

箭头左侧为该类base support删除法准确率，右侧为锁定operator准确率。`20-19`在low场景显示出明确DC/RMS support收益，但clear仍只有60%；`1-18`在low场景仍只有20%，说明类条件view不能单独解决其结构性floor。新类选择严格使用before冻结的operator中心/尺度；before旧类operator、prototype和calibration均未更新。

support锁定artifact：`automation_reports/CV-SincNet/d4a_single_observation_smoke_20260717_010128/dev_k10_new5_r2/d7a_support_lock_r2/support_lock.json`。

该证据只支持“D7a合法完成support锁定且识别出类/场景差异”，不支持query性能改善或候选晋升。下一未评分development seed应预先锁定本实现，再报告`20-19`、`1-18`真实query floor。
