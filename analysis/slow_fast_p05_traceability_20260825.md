# Slow-Fast P0.5需求追踪

来源：用户指导报告《对提交e39c28c9的分析与优化建议》；设计：`docs/superpowers/specs/2026-08-25-slow-fast-p05-calibration-design.md`。

|ID|来源章节|验收要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|P05-01|§1、§6|不能把失败简化为trust过严，显式处理support/query校准不足|calibration/report|implemented|51项Slow-Fast聚焦测试|真实source结果待N607|
|P05-02|§2|FILM主候选、LOWRANK单点、COMMON单点负对照|config/calibration|verified|P0.5只允许FILM；旧V2保留LOWRANK/COMMON诊断|收缩资源|
|P05-03|§3|核验`+0.83pp`的净正确样本数和多重比较边界|scorer/report|verified|旧类positive/negative flip手算测试|不得写成稳定收益|
|P05-04|§4.1|使用Q90 move与宽松hard max双层trust|selection|verified|分位数/hard cap fixture|不再单独依赖max|
|P05-05|§4.2|实现margin-normalized相对移动风险|selection|verified|边界距离手算测试|余弦原型几何|
|P05-06|§5|使用support统计归一化lambda强度|selection|verified|support归一化强度测试|目标support-only|
|P05-07|§6.1|记录逐fold风险增益、标准差、正fold数和LCB|selection|verified|逐fold稳定性测试|默认5/6|
|P05-08|§6.2、§11A|实现source receiver-held-out gate校准|calibration/CLI|verified|K10物理互斥/receiver-held-out测试|真实校准待N607|
|P05-09|§7.1|生成每场景query收益响应面|diagnostics/report|verified|完整状态轴测试|truth-last诊断|
|P05-10|§7.2|计算support/query Spearman相关性|diagnostics|verified|含平分rank的手算测试|阈值判断随最终报告|
|P05-11|§7.3|生成Q90 move—query收益数据|diagnostics|verified|move/query响应面测试|图表可后处理|
|P05-12|§8.1|拆分old changes、positive/negative flips和new changes|scorer|verified|旧/新query fixture|只在truth打开后|
|P05-13|§8.2|使用raw cosine计算margin、score L2和新类侵入|scorer|verified|raw-cosine手算fixture|REG0新类准确率仍N/A|
|P05-14|§9|拆分部署、cross-fit、legacy、shadow和总更新量|runner/receipt|verified|`21+183+76=280`fixture|避免混淆星上开销|
|P05-15|§10.1|runner显式传入row seed作为crossfit seed|runner|verified|seed392003传播测试|同scene候选公平|
|P05-16|§10.2|`loo_fit_count=0`并记录selection protocol|selection|verified|audit schema测试|修复语义|
|P05-17|§10.3|decision rule升级V2并记录三个schema|runner|verified|正式P0.5 receipt测试|消除V1误标|
|P05-18|§10.4|去重cross-fit划分并直接检查physical ID互斥|selection|verified|唯一split与ID互斥测试|不做额外hash|
|P05-19|§10.4|保存fold physical-ID hash|无|rejected|`REJECTED_EXTRA_GATE`|项目规则禁止逐support-token hash|
|P05-20|§11B|只使用新的receiver／seed capsule进行一次目标确认|config/N607|pending|capsule可用性审计|旧query不得重跑|
|P05-21|§11C|设置P0停止条件并条件触发P1|report|pending|最终同row结论|P1不提前实现|
|P05-22|§12|增加worst-scene、scene/class和新类侵入晋级条件|scorer/report|pending|阈值边界测试|配对统计同报|
|P05-23|§最终判断|P1因子化慢基和P2中间层仅在P0失败后实施|无|deferred|独立目标结果触发|避免机制混叠|

当前统计：pending=3，implemented=1，verified=17，deferred=1，rejected=1，blocked=0。唯一独立P0/P1审查发现Phase2曾打开完整source校准JSON；已改为只把纯deployment参数写入row config，并经原问题定点复审确认`FIXED`。最高剩余风险是合法新receiver／seed capsule可能尚不存在；这不会阻止地面校准，但会阻止独立目标性能结论。
