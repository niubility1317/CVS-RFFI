# D43结构化协方差探针追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D43-01|复用D42同一固定received IQ、15个support-held折与old-only B20，不打开query|probe/receipt|pending|待运行三个预锁arm|
|D43-02|只比较full-centered对照、3-block与diagonal；不扫描threshold/rank/lr/类专属参数|probe/tests|verified|`ARMS`/CLI封闭；目标测试30/30通过|
|D43-03|3-block严格按z160/FFT96/RF32置零跨块协方差；diagonal只保留对角|probe/tests|verified|矩阵结构单测通过|
|D43-04|去除所有类共有的coefficient/intercept均值；FP64代数等价，FP32 support argmax fail-close并报告pairwise drift|probe/tests|verified|代数等价、D42 full-control及FP32 support断言通过|
|D43-05|旧类before必须在首次new support读取前物化，Phase1 ground int8保持只读|D42 lifecycle|pending|复用D42 Runner真实entry/exit证据待执行|
|D43-06|所有类等先验；query逐样本面对完整registry；无role/quota/global assignment|D42 score/receipt|pending|复用D42 fail-closed面，真实receipt待执行|
|D43-07|对每个arm保存并报告全部匿名类的before-old/after-old/seen-new、三类最低值、H、forgetting、joint floor、逐场景及量化翻转|probe analysis|pending|三个artifact待生成|
|D43-08|从D42原始SHA锁定全精度基准；聚合、三类floor、遗忘与逐场景均不得退化，至少一个final floor严格提高|decision|pending|容差和并列规则已在报告预锁，结果待执行|
|D43-09|D43探针不是正式候选，不得直接进入full-K10或N607|scope|implemented|selector强制identity、full-K10 refit禁用及evidence核验已实现；真实receipt待验证|
|D43-10|只有完整独立确认矩阵全门达标才能完成goal|confirmation|deferred|D43 development不等于目标完成|

当前计数：`pending=5`、`implemented=1`、`verified=3`、`deferred=1`、`rejected=0`、`blocked=0`。
