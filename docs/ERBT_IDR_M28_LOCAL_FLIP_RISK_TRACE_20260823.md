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
- 状态：`ANALYZED / SCREEN_NEGATIVE_NO_FULL125`

## 本地闭环

- RED：缺失M2.8模块导致预期collection失败。
- GREEN：M2.8聚焦测试8项通过。
- 相邻回归：M2.5/M2.7/M2.8共40项通过。
- M2.4编译/集成与M2.8共29项通过。
- 正式模块入口smoke通过；`git diff --check`通过。
- 唯一一次独立P0/P1审查：`PASS`，直接阻断问题为0。

## N607正式实验闭环

- release archive本地/远端SHA-256一致：`9531eafc11ca1265003aabbd270ea6fd860ce71f34def7ec5bb151ec9557a01b`。
- 真实checkpoint无query smoke通过：195个tensor严格加载，75个tensor输出有限，query输入0，truth未打开。
- prediction只启动一次，父PID`1810409`；16/16行和48个场景单元完成，B0/B3/C1/C2各4行。
- truth打开前`matrix_index.status=PREDICTIONS_COMPLETE_TRUTH_UNOPENED`，4个paired identity闭合，R1注册前/后分歧均为0。
- 独立truth-last scorer完成16行评分；汇总状态为`ANALYZED`。
- 首次交互PTY smoke等待stdin关闭，只终止精确连接进程`1807229`并确认消失；正式stdin重定向smoke通过。该事件未触发prediction重启。

## 需求—结果闭合

|需求|正式结果|裁决|
|---|---|---|
|包含target receiver域偏移|target-centered MGD96已进入状态，中心角距发生非等价变化|工程通过；不能单独证明分类收益|
|去RF32|B0/B3/C1/C2均使用物理IF256，无RF32恢复|通过|
|FFT96改进|MGD96、共形与径向证据均生成|表示有效；效用校准失败|
|逐query局部风险|每条query只从完整B0/B3分数行选择|通过|
|提高M2.7召回|C1/C2接受0次实际B3翻转|失败|
|超过B0和B3|C1/C2 H=0.610486，与B0相同，较B3低0.005385|失败|
|完整125晋级|预登记门槛未通过|不启动full125|

## 性能摘要

|arm|H|F|min-old|min-new|相对B0 H|help/harm vs B0|
|---|---:|---:|---:|---:|---:|---:|
|B0|0.610486|0.092643|0.268919|0.251577|0|基线|
|B3|0.615871|0.088626|0.281306|0.256532|+0.005385|28/5|
|C1|0.610486|0.092643|0.268919|0.251577|0|0/0|
|C2|0.610486|0.092643|0.268919|0.251577|0|0/0|

C1/C2的预测标签及逐类性能与B0完全一致。注册后B3相对B0共有49次argmax翻转，其中28 help、5 harm、16 neutral；C1/C2全部否决，相对B3的help/harm变为5/28。

## 失败定位

实现中的support训练事件使用`source=argmax(B0 support score)`和`candidate=MGD LOO top1/top2`，成功定义为MGD candidate等于support标签；query阶段却把该后验用于实际`B0预测→B3预测`类别对。它学习的是MGD候选是否正确，而不是B3相对B0是否有益。

12个场景拟合中的rank1和rank2成功事件全部为0；因此分层后验、共形、径向和类别稳定门共同把所有B3翻转拒绝。C1/C2阈值虽不同，但在零成功事件下结果相同。问题不是target receiver域中心未构造，而是监督目标错位。

## 下一候选约束

下一候选应直接在support leave-one-out上构造B0与B3两套折内预测，并以`B3正确/B0错误`、`B3错误/B0正确`和neutral形成真实翻转效用标签。类别对必须来自实际B0→B3预测；MGD96、共形、径向、margin和残差强度只作为协变量。

默认决策改为保留B3，只在harm后验达到高置信阈值时回退B0。无事件、稀疏pair和低K时不再默认B0。这样才能利用M2.5 full125中352/98和本screen中28/5的正先验，同时专门压缩harm。
