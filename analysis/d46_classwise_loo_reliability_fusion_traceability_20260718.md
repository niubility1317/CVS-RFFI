# D46类级head-only LOO可靠度融合追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D46-01|复用D42–D45同一fixed received-IQ development cell、old-only B20和15个outer held折，不打开query|probe/receipt|implemented|探针强制identity、development seed713101和query0|
|D46-02|full与3-block组件在RMS、inner CE和类级融合前统一进入“逐特征系数类均值为0且截距类均值为0”的canonical gauge|fit/tests|verified|任意类公共仿射偏移消除测试和逐fit数值闭环通过|
|D46-03|每类仅以相同公式`w_g,c=softmax_g(-K×CE_g,c)`使用合法support内部head-only LOO证据；实际outer/inner fit的K作为指数|fit/tests|verified|公式、极端有限输入、非有限fail-close和K指数tamper测试通过|
|D46-04|类标签置换等变；不按class ID、old/new角色、receiver、handle或场景分支；不读outer-held/query|fit/tests|verified|标签置换逐行等值；审计字段与source closure锁定|
|D46-05|K1固定1:1等价回退；K2同CE证据严格闭合为1:1，否则fail closed|fit/tests|verified|K1状态和K2公式闭环测试通过|
|D46-06|资源保留D45的`4K+4`精确LDA inventory，另计逐类可靠度打分与仿射融合MAC；只持久化一个query state|resource/tests|verified|资源重算与inventory tamper测试通过|
|D46-07|D42–D46定向回归全部通过，且独立代码复核无P0|tests/review|verified|独立复核无P0；K1资源P1及两项证据P2已修复/加固；82项通过|
|D46-08|真实105行development输出的协议、lifecycle、source、ground、state、resource与artifact闭包全部通过|probe|pending|待同一D18 cell运行|
|D46-09|相对D42聚合、floor、场景、forgetting、joint和混淆全部通过预锁门，且final floor至少一项严格改善|decision|pending|待真实结果|
|D46-10|before/final int8-FP32 argmax变化与margin翻转均为0|decision|pending|待真实结果|
|D46-11|D46相对D45的15个final held预测至少改变1个；若0个则拒绝该机制|decision|pending|待prediction SHA/逐样本比较|
|D46-12|探针不生成125、不访问N607；仅在另行正式化候选且开发门全过后生成125 handles|scope|verified|当前仅本地support-held开发流程|

当前计数：`pending=4`、`implemented=1`、`verified=7`、`deferred=0`、`rejected=0`、`blocked=0`。
