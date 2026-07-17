# D27逐新类安全bias追溯表

日期：2026-07-18

状态：核心、runner、launcher和65项相关回归完成，待N607 90行support-only执行。

|ID|需求|实现|状态|验证|
|---|---|---|---|---|
|D27-01|单IQ高维拼接|`z160+FFT96+RF32`同一接收IQ一条288D行|verified|复用D25 capsule/operator闭包|
|D27-02|轻型域适应|shared diagonal+逐旧类weight，Stage2-B 15步|verified|核心loss/resource测试|
|D27-03|独立新类注册|每个新类独立FP32 bias，0/10/15步suffix|verified|向量状态与append测试|
|D27-04|旧类遗忘保护|逐新类安全cap保留所有old-only正确support行及逐类准确率|verified|构造碰撞与随机安全测试|
|D27-05|floor优先|support LOO坐标选择按min-class、overall、margin排序|verified|坐标选择测试|
|D27-06|K1|直接安全cap，无伪LOO|verified|K1测试|
|D27-07|逐样本全类决策|全部注册类FP32 score+一次argmax|verified|predict/API测试|
|D27-08|query/clean/source隔离|无query/truth/role/quota/source/clean拟合入口|verified|API/CLI与resource审计|
|D27-09|资源上限|≤2,016活动参数、≤30step、状态≤256KB、无dense query图|verified|5/20新类资源测试|
|D27-10|90行与选择锁|Z0/B3/C0+D27A/B/C，C0双floor与H/forgetting门|verified|runner lock/selector测试|
|D27-11|冻结旧头|旧weight、diagonal、raw score prefix不变|verified|fold/full-K10测试|
|D27-12|证据闭包|runner/core/operator SHA与Git提交独立记录|verified|candidate lock/launcher门|

实际性能仍待N607执行，不在本地实现PASS含义内。
