# D43结构化协方差探针追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D43-01|复用D42同一固定received IQ、15个support-held折与old-only B20，不打开query|probe/receipt|verified|三个arm各105/105行；query0，physical-token闭包一致|
|D43-02|只比较full-centered对照、3-block与diagonal；不扫描threshold/rank/lr/类专属参数|probe/tests|verified|`ARMS`/CLI封闭；目标测试30/30通过|
|D43-03|3-block严格按z160/FFT96/RF32置零跨块协方差；diagonal只保留对角|probe/tests|verified|矩阵结构单测通过|
|D43-04|去除所有类共有的coefficient/intercept均值；FP64代数等价，FP32 support argmax fail-close并报告pairwise drift|probe/tests|verified|代数等价、D42 full-control及FP32 support断言通过|
|D43-05|旧类before必须在首次new support读取前物化，Phase1 ground int8保持只读|D42 lifecycle|verified|三arm lifecycle/ground gate全真，ground entry/exit逐bit|
|D43-06|所有类等先验；query逐样本面对完整registry；无role/quota/global assignment|D42 score/receipt|verified|三arm source/state gate全真，receipt/support均query0|
|D43-07|对每个arm保存并报告全部匿名类的before-old/after-old/seen-new、三类最低值、H、forgetting、joint floor、逐场景及量化翻转|probe analysis|verified|完整315行、三组逐类/场景/混淆/量化/资源已入报告|
|D43-08|从D42原始SHA锁定全精度基准；聚合、三类floor、遗忘与逐场景均不得退化，至少一个final floor严格提高|decision|verified|block因最低新类与场景门失败；diagonal多门失败；无结构入选|
|D43-09|D43探针不是正式候选，不得直接进入full-K10或N607|scope|verified|三arm均强制identity，full-K10未执行，N607未访问|
|D43-10|只有完整独立确认矩阵全门达标才能完成goal|confirmation|deferred|D43 development不等于目标完成|

|D43-11|3-block结构进入正式实现|mechanism result|rejected|聚合/H/floor正信号，但最低seen-new0.6667<0.7、low-elev new退化、rain old/forgetting退化|
|D43-12|纯diagonal结构进入正式实现|mechanism result|rejected|before/after/new/H和三类floor全面退化，且before量化argmax变化1|
|D43-13|full-centered对照作为D43结构候选晋级|control result|rejected|按预注册不参与选择；只证明公共项去除可实现0量化翻转|

当前计数：`pending=0`、`implemented=0`、`verified=9`、`deferred=1`、`rejected=3`、`blocked=0`。
