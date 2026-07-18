# D34-FCLER设计追踪

|目标/约束|实现/证据面|
|---|---|
|域适应与新类注册同等优先|FAST Fisher旧头+FCLER注册在同一fold输出注册前/后old、new、H、forgetting|
|旧类遗忘与floor保护|旧score prefix逐bit冻结；old support逐类non-degradation和old LOO零侵入为硬门|
|新类注册|每新类独立support mean/medoid原型、int8+scale+inverse norm、至少一条support-built碰撞边|
|无角色Oracle/配额|逐样本先算旧winner，再为全部注册类生成有限score并argmax；无query batch信息|
|query只测试|开发screen中query rows/features/labels均为0|
|唯一LEO_weak观测|只复用密封单观测support；z160/FFT96/RF32来自同一接收IQ，不增加K|
|轻量部署|0 optimizer step、稀疏边、int8新类中心；审计平均/最坏degree和MAC/state/latency|
|K1/5/10|K10锁arm；K5复用锁定配置；K1无LOSO/派生shot并用固定安全margin|
|晋级边界|必须联合优于D33-FAST并通过旧类安全门；否则保持开发负证据|

`项目.md`无需修改：D34不改变既有场景、数据、K-shot、Stage2-B/C、query或clean/source权限，只在既有合法边界内更换Stage2-C注册机制。
