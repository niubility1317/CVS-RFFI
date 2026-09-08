# A1-Fast第一轮设计落地追踪

最新授权：不使用历史checkpoint；使用CORE90须在相同实验配置下重训。执行新CORE90从零E200→本轮checkpoint→两组A1各E200，保留RC4锚点语义，不采用临时拟议的无锚点方案。

原A1实际release的1338个Python/shell/JSON文件均匹配`7504f6669fbc0a02b9b7446f463f561ecbcef6de`，证据见a1_release_provenance.json。该基线只有7个DAOT分量，没有72360版本的clean-anchor等后续组件。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|R01|2,9.1|原源码和本轮初始化|a1_release_provenance.json；matrix config|verified|1338文件一致；同配置测试|新CORE90，不宣称复现历史成绩|
|R02|3|尺度批量读回|orbit_teacher.py|verified|loss/grad/状态/NaN/Inf/inactive/L→U测试|保留Python float和state schema|
|R03|4|逐视图身份教师|model_dual_cvsincnet.py；train_ssdg.py|verified|CPU真实S/M输出/BN/RNG/参数/EMA比较|保留学生前向|
|R04|4|批量教师|train_ssdg.py|deferred|CPU输出及梯度；拒绝train/无running-stat BN|GPU/AMP在发布smoke核验；容差不代替性能|
|R05|5|mean无用计算门控|train_ssdg.py|verified|实际L/U step比较|nuisance零权重有零grad与日志消费者，保留；prototype缺输入原已跳过|
|R06|6|显式教师执行层和同一步复用|_forward_daot_teacher_views|verified|L两fresh；U不重算clean；student_strong不变|无跨步缓存，不拼学生batch|
|R07a|7.1|日志批量汇总|post_stage_common.py|verified|混合dtype/异常/缺项/顺序测试|保留原字段和Python累加|
|R07b|7.1|验证复用|基础和tail验证|deferred|基础无domain参数、tail有；卫星随机输入不同|第一轮未证明等价，保留频率与计算|
|R08|6.4|搬运及数据顺序|原loader/move_batch|verified|已有pin_memory/non_blocking|不改worker/shuffle/增强，不冒充新增收益|
|R09|7.2|完整E20前缀|checkpoint/resume|deferred|用户要求从开始训练，本轮不复用E20|只共享本轮完整CORE90，不声称exact resume|
|R10a|8.2|执行与教师健康诊断|train_ssdg.py|verified|字段可达且CORE90关闭DAOT安全|正确率/置信度/identity-only/原梯度与step日志|
|R10b|8.4|EMA缓存回归和修复分离|原CosFace/_update_ema_model|deferred|确认.data更新后自动与强制刷新输出不同|已知旧缺陷；两组保持，修复归Stable|
|R11|9.2|计时及阶段覆盖|check_a1_fast_execution.py；原epoch资源日志|implemented|E1/E21/E161状态；B128/B256微基准|十项全程profile未闭合，不报总加速|
|R12|9.3|重训CORE90与配对E200|run_a1_fast_matched_core90.py|implemented|配置/旧checkpoint负测/独立P0/P1审查|启动和正式结果待读回|
|R13|8,9|学生合并、分支控制、Stable|独立后续实验|deferred|报告明确与Fast分离|本轮不启动|

本轮为执行优化与匹配重训流水线，不是长期重构全部完成。最高风险是保留的原EMA缓存缺陷和批量kernel差异；科学结论等待配对E200，不允许默认掉点，不用target反馈调参或重跑。

GPU补充：逐视图FP32/AMP exact；batch FP32偏差超预设容差，排除r2正式矩阵。r1仅smoke失败未训练；r2为匹配CORE90+两个A1。
