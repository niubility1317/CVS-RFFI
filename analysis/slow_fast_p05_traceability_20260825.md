# Slow-Fast P0.5需求追踪

来源：用户指导报告《对提交e39c28c9的分析与优化建议》；设计：`docs/superpowers/specs/2026-08-25-slow-fast-p05-calibration-design.md`。

|ID|来源章节|验收要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|P05-01|§1、§6|不能把失败简化为trust过严，显式处理support/query校准不足|calibration/report|verified|N607完成28个source receiver-held-out episode|最差均值-0.128205pp，不能宣称正收益|
|P05-02|§2|FILM主候选、LOWRANK单点、COMMON单点负对照|config/calibration|verified|P0.5只允许FILM；旧V2保留LOWRANK/COMMON诊断|收缩资源|
|P05-03|§3|核验`+0.83pp`的净正确样本数和多重比较边界|scorer/report|verified|旧类positive/negative flip手算测试|不得写成稳定收益|
|P05-04|§4.1|使用Q90 move与宽松hard max双层trust|selection|verified|分位数/hard cap fixture|不再单独依赖max|
|P05-05|§4.2|实现margin-normalized相对移动风险|selection|verified|边界距离手算测试|余弦原型几何|
|P05-06|§5|使用support统计归一化lambda强度|selection|verified|support归一化强度测试|目标support-only|
|P05-07|§6.1|记录逐fold风险增益、标准差、正fold数和LCB|selection|verified|逐fold稳定性测试|默认5/6|
|P05-08|§6.2、§11A|实现source receiver-held-out gate校准|calibration/CLI|verified|K10物理互斥测试+N607 28个episode闭合|选中FOLD_LCB配置|
|P05-09|§7.1|生成每场景query收益响应面|diagnostics/report|verified|完整状态轴测试|truth-last诊断|
|P05-10|§7.2|计算support/query Spearman相关性|diagnostics|verified|含平分rank及rho=0.1阈值测试|P0停止阈值固定为<0.2|
|P05-11|§7.3|生成Q90 move—query收益数据|diagnostics|verified|move/query响应面测试|图表可后处理|
|P05-12|§8.1|拆分old changes、positive/negative flips和new changes|scorer|verified|旧/新query fixture|只在truth打开后|
|P05-13|§8.2|使用raw cosine计算margin、score L2和新类侵入|scorer|verified|raw-cosine手算fixture|REG0新类准确率仍N/A|
|P05-14|§9|拆分部署、cross-fit、legacy、shadow和总更新量|runner/receipt|verified|`21+183+76=280`fixture|避免混淆星上开销|
|P05-15|§10.1|runner显式传入row seed作为crossfit seed|runner|verified|seed392003传播测试|同scene候选公平|
|P05-16|§10.2|`loo_fit_count=0`并记录selection protocol|selection|verified|audit schema测试|修复语义|
|P05-17|§10.3|decision rule升级V2并记录三个schema|runner|verified|正式P0.5 receipt测试|消除V1误标|
|P05-18|§10.4|去重cross-fit划分并直接检查physical ID互斥|selection|verified|唯一split与ID互斥测试|不做额外hash|
|P05-19|§10.4|保存fold physical-ID hash|无|rejected|`REJECTED_EXTRA_GATE`|项目规则禁止逐support-token hash|
|P05-20|§11B|只使用新的receiver／seed capsule进行一次目标确认|config/N607|pending|N607只读审计完成|`MISSING_INDEPENDENT_TARGET_CAPSULE`，未复用rx20-1|
|P05-21|§11C|设置P0停止条件并条件触发P1|report|verified|正式报告冻结均值/floor/Spearman条件|等待新capsule后执行，不提前实现P1|
|P05-22|§12|增加worst-scene、scene/class和新类侵入晋级条件|scorer/report|implemented|scorer与正式报告阈值|缺少source class-heldout真新类侵入阈值，不能完全验证|
|P05-23|§最终判断|P1因子化慢基和P2中间层仅在P0失败后实施|无|deferred|独立目标结果触发|避免机制混叠|

当前统计：pending=1，implemented=1，verified=19，deferred=1，rejected=1，blocked=0。唯一独立P0/P1审查发现Phase2曾打开完整source校准JSON；已改为只把纯deployment参数写入row config，并经原问题定点复审确认`FIXED`。地面校准已完成并选中`P05_RELATIVE_K08_FOLD_LCB`；N607只读审计确认不存在新的独立目标capsule，因此没有启动目标确认，也没有把旧receiver=`20-1`结果包装为独立收益。

## 对提交8f76032f的深度优化建议追踪

来源：ChatGPT对提交`8f76032f1bd99b8baa6e3bd14953e15fc3e4a8e4`的深度优化建议。该建议按外部review处理，先核验代码和协议，不作为直接指令。

|ID|来源章节|验收要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|SFR-01|§3.3|加入`ALWAYS_DA0`空策略，非零策略不优于DA0时输出`CALIBRATED_TO_ABSTAIN`|calibration/tests|verified|abstain先红后绿测试|DA0以0变化、0侵入、0选择计算参与同一排序|
|SFR-02|§3.5|cross-fit每个fold只能用train half估计强度normalizer|selection/tests|verified|fold scope与normalizer数量测试|validation half不参与本fold强度估计|
|SFR-03|§3.7|互补2-fold先按repeat聚合，再做稳定性判断|selection/tests|verified|5/6正fold但仅2/3正repeat fixture|稳定性门控使用repeat gain与repeat LCB|
|SFR-04|§3.8|加入有方向margin保护：正确样本保留margin，错误样本要求margin改善|selection/tests|verified|正确/错误各一条手算fixture|固定正确margin保留比例0.5；错误样本要求严格改善|
|SFR-05|§3.4|拆分cross-fit、full-support、committed和total selection updates|selection/runner/report|verified|`18+3=21`及总诊断280测试|query inference更新数固定为0|
|SFR-06|§3.1、§3.2|嵌套receiver留出：外层慢基完全未见、内层策略选择|Phase1.5/calibration|pending|P0.6 abstain已触发|需要为每个外层receiver重训轻型慢基，不能与本次门控修复混为一个因果变更|
|SFR-07|§3.9|source class-heldout pseudo-new侵入校准|calibration/scorer|deferred|后续独立地面候选|需要冻结新类侵入阈值，当前代理不得冒充真新类侵入|
|SFR-08|§5—§7|实现receiver rank4+LEO rank4因子化慢基|Phase1.5|pending|P0.6输出`CALIBRATED_TO_ABSTAIN`|已成为下一主候选，星上快参数预算8|
|SFR-09|§13|新增逐文件checkpoint/prototype/code SHA|无|rejected|`REJECTED_EXTRA_GATE`|项目规则禁止额外成员SHA；Git提交和一次release归档SHA已固定代码|
|SFR-10|§17A|复用旧rx20-1 query搜索非零上界|无|rejected|协议与用户约束|旧truth不得反馈调参或冒充独立确认；等待新合法capsule|

本轮实施边界：先完成SFR-01至SFR-05并重新执行source-only最小校准。若`ALWAYS_DA0`胜出，正式结论为`CALIBRATED_TO_ABSTAIN`，这构成进入SFR-06/SFR-08的科学触发条件，但不授权复用旧目标query。

本轮验证：36项Slow-Fast聚焦回归通过，三个修改模块语法编译通过，`git diff --check`通过。唯一独立P0/P1定点审查结论为`NO_P0_P1`；审查只覆盖SFR-01至SFR-05，未增加白名单外gate。

N607 P0.6结果：28个source episode全部闭合，最终`CALIBRATED_TO_ABSTAIN/P05_ALWAYS_DA0`。四个非零门控均在12/28个episode提交，平均mean变化约+0.05pp，但worst receiver mean和worst episode floor均为0，且最大置信侵入代理为0.00027455；DA0以零侵入和零选择计算胜出。该结果验证SFR-01至SFR-05并将SFR-06、SFR-08从deferred转为pending，但不构成Phase2目标性能证据。
