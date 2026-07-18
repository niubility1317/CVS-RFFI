# D45冻结outer-B20的head-only LOO可靠度融合追溯表

状态词仅使用`pending/implemented/verified/deferred/rejected/blocked`。

|ID|Requirement|Target|Status|Verification/stop evidence|
|---|---|---|---|---|
|D45-01|复用D42–D44同一fixed received-IQ development cell、old-only B20和15折，不打开query|probe/receipt|verified|105/105行；query0；seed713101|
|D45-02|只保留D42锁定原始full解的类公共项中心化与3-block两组件；完整fit仍用support RMS归一化后合成单state|fit/tests|verified|67项D42–D45定向回归通过；full不另行重构sklearn系数|
|D45-03|B20只在outer fit训练一次并冻结；inner仅按support-row rank重拟合LDA/RMS，held row不参与对应head；这不是全链路nested无泄漏泛化估计|fit/tests|verified|互斥、exact-once覆盖和逐fold CE测试通过|
|D45-04|可靠度为逐fold/逐类等权inner-held CE；以稳定log-domain固定计算`log_evidence=-C×macroCE`和两组件softmax；无clip/temperature/scan|fit/tests|verified|数值、类别置换和weight closure测试通过|
|D45-05|只读合法support feature/label；不读outer-held/query、old/new角色、handle或场景ID|fit/audit|verified|verifier锁定无外部held/query/branch访问字段|
|D45-06|K1固定1:1等价回退；K2的每个inner-train为K1，必须由证据得到数值等价1:1|fit/tests|verified|K1状态测试与K2 unit-component证据测试通过|
|D45-07|资源按before/final各2个main fit＋每阶段`2K`个inner fit计数；总数`4K+4`，K8 outer共36次LDA|resource/tests|verified|逐fit inventory与MAC closure测试通过|
|D45-08|聚合、逐类、逐场景、forgetting、joint与三类混淆相对D42全部不退化且final floor严格改善|decision|rejected|aggregate forgetting10.00pp>8.89pp；rain after76.67%<78.33%、forgetting13.33pp>10pp|
|D45-09|before/final int8-FP32 argmax变化与margin翻转均为0|decision|verified|真实15折为`0/0/0`，max score error0.0016141|
|D45-10|探针强制identity、禁用full-K10；门全过后才另行正式化并运行125|scope|verified|D43复用guard测试通过；不消耗confirmation seeds|
|D45-11|verifier从held索引、逐fold/逐类/macro CE、log-evidence和权重重算闭包，并按四个fit组重算资源；类别置换由函数单测验证；before在读取new前物化且不可变|audit/tests|verified|4类持久化证据tamper与lifecycle closure测试通过|
|D45-12|D45完成后执行D43–D45强制技术复盘|retrospective|verified|重读goal/项目/三报告/完整日志并刷新conversation index；下一轮D46类级同式可靠度融合|

当前计数：`pending=0`、`implemented=0`、`verified=11`、`deferred=0`、`rejected=1`、`blocked=0`。
