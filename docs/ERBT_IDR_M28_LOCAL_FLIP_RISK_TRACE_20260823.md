# ERBT-IDR M2.8局部共形翻转风险实现追踪

## 设计输入与问题定位

M2.5完整125证明B3相对去RF32 D92 E0存在小而稳定的正收益，但仍有98条harm。M2.7证明MGD96有正交信息：screen中其接受的6次B3翻转全部有益；问题是整行可靠性门只召回28次有益翻转中的6次，并把receiver`3-19`全部拒绝。Phase32在12个场景拟合中没有可靠场景，不继续作为主候选。

M2.8把优化对象从“整行表征是否可靠”收紧为“当前query的当前类别对翻转是否值得接受”。

## 需求—实现—证据映射

|需求|冻结实现|验证证据|
|---|---|---|
|包含target receiver域偏移|旧类逐类中位中心后再做类平衡均值，得到共享MGD96目标域中心|顺序不变、单类离群鲁棒性和audit字段测试|
|去RF32|B0/B3继续使用物理IF256；辅助表征只读取`blocks[:,160:256]`的FFT96|配置锁、状态维度和row receipt|
|改进FFT96利用|使用MGD96的趋势残差、局部斜率和fftshift镜像不对称，并做target-centered LOO校准|表征fit audit与中心角距证据|
|避免M2.7整行否决|按rank、目标类和`源类→目标类`构造分层Beta-Binomial后验|pair后验、证据层级和接受率诊断|
|控制过拟合|所有校准事件均为support leave-one-out；query只读|`query_rows_used=0`、batch/单样本一致性测试|
|保留主决策|每个query只复制完整B0或完整B3分数行|逐行精确相等测试与`row_source_allowlist`|
|小K失败闭合|`K<5`和rank事件不足时精确B0|K1测试、fallback audit|
|可比较实验|冻结B0/B3/C1/C2同row screen；门槛通过才扩展完整125|matrix index、独立scorer和汇总gate|

## 不采用的路线

- 不放宽M2.7全局LOO阈值：它仍是整行决策，无法解释局部help/harm。
- 不用历史query truth训练风险模型：这会把测试标签经验带入方法选择并增加过拟合风险。
- 不恢复RF32：历史证据已表明RF32是无效零填充维度。
- 不继续Phase32主分支：M2.7 screen没有支持其可靠性。

## 当前状态

- 分支：`codex/m28-local-conformal-risk-20260823`
- 基线提交：`e55d3d49f47bd0eeded5565a63e4ebf1b6783bf8`
- run ID：`erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1`
- 状态：`LOCAL_VERIFIED_RELEASE_PENDING`

## 本地闭环

- RED：缺失M2.8模块导致预期collection失败。
- GREEN：M2.8聚焦测试8项通过。
- 相邻回归：M2.5/M2.7/M2.8共40项通过。
- M2.4编译/集成与M2.8共29项通过。
- 正式模块入口smoke通过；`git diff --check`通过。
- 唯一一次独立P0/P1审查：`PASS`，直接阻断问题为0。
