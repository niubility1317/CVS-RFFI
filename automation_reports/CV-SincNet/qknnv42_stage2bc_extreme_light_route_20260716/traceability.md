# qKNNv4.2正式Stage2-B/C极轻路线追踪表

此文件镜像根目录同名追踪表，Git侧状态以逐项提交更新为准。

|ID|要求|状态|关键证据|
|---|---|---|---|
|R01|Phase2`LEO_weak-only`且clean/clean-derived物理不可达|verified|125个v2 row的execution receipt逐项核验完整协议字段；support/query均来自三种密封`leo_*_weak`包|
|R02|逐样本全部注册类、无role/真实批次数/quota/global assignment|verified|strict summary复核125×2个execution receipt；query fit rows=0，dense query graph=0|
|R03|prediction与independent scorer隔离|verified|125对before/after不可变prediction；truth仅在两份prediction完成后由独立scorer连接|
|R04|Stage2-B注册前与Stage2-C注册后同row|verified|125个pair job、375个scenario pair、750个scenario-state指标|
|R05|真实嵌套5/10/20 seen-new TX覆盖|verified|K10覆盖new5/10/20；K5和K1覆盖new20；75个K1/K10 support-query嵌套审计PASS|
|R06|K10开发锁定、K5独立matched确认|verified|development seed713101未进入125矩阵；candidate锁定为`d1_historical_diag_fftrf`|
|R07|多receiver、多seed、多场景确认|verified|5 receiver×5独立seed×5 slice；每row覆盖3种LEO场景|
|R08|K10 old>=92%、旧类floor>=88%|failed|B old为92.47%，但pooled B floor为85.67%；C old与C floor全部失败|
|R09|K10 seen-new 5/10/20>=92/90/86%|failed|84.24%/80.97%/86.29%，仅new20达到线|
|R10|K5较K10下降<=3pp|failed|matched new20的C old/new/H平均下降4.19/3.86/4.07pp，全部比较未统一通过|
|R11|注册后旧类遗忘控制|failed|K10 old adaptation gain为-4.64至-8.12pp；K1总体-18.07pp且5个receiver均为负|
|R12|adapter<=50k、<=20epoch、无dense query图|verified|max7,802参数、20epoch、33,055B状态、dense query graph=0|
|R13|identity-only及三种方法Pareto|incomplete|本轮已完成D1实际MAC、时延、显存和状态审计；完整direct ADV3B02 K1及三方法同矩阵Pareto仍缺|
|R14|完整日志或闭式求解诊断|verified|125份stdout/stderr/events/summary、逐row loss/resource receipt；0非空stderr|
|R15|合法TX/receiver/support-query清单|verified|密封cache、authority bundle、row manifest、truth sidecar及support/query opened-member SHA闭环|
|R16|自动化报告和Git提交|verified|代码提交`eaabeed`、`7c97720`、`446b16d`；根报告第42–45节与本Git镜像|
|R17|每3个turn回顾目标和对话|verified|本轮重读目标和`项目.md`边界后直接推进125重跑，未用确认query调参|

当前结论：`STRICT_V2_125_COMPLETE_PERFORMANCE_FAIL_ROUTE_EXPLORATION_ACTIVE`。旧v1保留为诊断基线；严格v2的125个score与v1逐job精确一致，但补齐support/query provenance闭环。D1不能晋级，后续必须修复多新类下旧类遗忘，并补direct ADV3B02 K1配对证据。
