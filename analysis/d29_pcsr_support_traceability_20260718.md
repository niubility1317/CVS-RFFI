# D29逐新类安全释放追溯表

日期：2026-07-18

状态：实现与本地验证进行中。

|ID|需求|预定实现|状态|验证|
|---|---|---|---|---|
|D29-01|单IQ LEO_weak|复用唯一接收IQ的288D拼接与D29 method-bound sealed support，不重建IQ或新增overlay|pending|capsule/operator/pre-open审计|
|D29-02|域适应+注册|完整候选为D27-B target-old域适应+逐新类PCSR注册；同run保存注册前/后状态|pending|相邻回归及before/after artifact|
|D29-03|旧类零翻转|每新类闭式`A_safe`并联合复验|pending|数学/随机测试|
|D29-04|弱新类floor|逐类`T/A`允许改变new-new排序|pending|gate-layer 5-fold shot-rank OOF与held逐类结果|
|D29-05|K1|无LOO时精确透传D27-B|pending|K1测试|
|D29-06|无Oracle/quota|逐样本全类一次argmax，无batch统计，使用当前精确`phase2_query_*`字段|pending|API/CLI/row permutation与batch-composition不变性审计|
|D29-07|极轻资源|2标量/new；PCSR状态含32B头；D27-B 25step；无dense图|pending|5/10/20类72/112/192B状态与组合资源测试|
|D29-08|完整证据|D27-B loss、PCSR闭式诊断、逐类/场景、资源、receipt、Git提交|pending|N607完成后审计|
|D29-09|support-only非正式边界|记录三个claim/authority false、query opened/rows/labels为0及`SUPPORT_ONLY_NO_QUERY_CLAIM`|pending|support audit/selection/receipt一致性|
|D29-10|精确clean/source/query边界|补全query四false、clean dataset/cache/control-flow不可达、source六false及pretrained artifact policy|pending|pre-open allowlist/SHA/provenance与runtime access audit|
|D29-11|Phase1/int8/method lock绑定|support打开前锁checkpoint SHA、int8 SHA、Phase1 TX→class handle逐列映射和D29候选/method lock|pending|candidate lock、detached seal与mapping审计|
|D29-12|注册前后同run证据|同query/View保存old-only before与all-class after，报告old/new/H/floor/forgetting及混淆|pending|正式prediction artifact与隔离scorer；本support screen不填结果|
|D29-13|组合资源与Pareto|报告D27-B+PCSR组合state、head/端到端MAC、平均/P95前向、FFT、时延、峰值RAM/显存|pending|相对identity-only单qKNN同硬件审计|
|D29-14|正式独立确认矩阵|锁定K10唯一候选后覆盖5 receiver×至少5 seed×3场景×真实5/10/20新类及K1/5/10/20|pending|独立密封package、不可变prediction与确认矩阵；不由本轮support screen替代|
|D29-15|三轮retrospective|D29完成后、D30前重读目标/协议/历史，审阅D27-D29完整日志并记录经验、拒绝路线和下一决定|pending|D29 report retrospective章节与Git状态|
