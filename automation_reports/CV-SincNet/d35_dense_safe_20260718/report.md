# D35-DSWR稠密安全winner条件注册实验

## 登记

- 实验ID：`d35_dense_safe_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`LOCAL_IMPLEMENTATION_IN_PROGRESS`。
- 假设：D34的主要失败来自稀疏可见性而非原型量化；把所有新类改为全winner可达，并用旧support最大残差阈值保护每个winner，可去除约`-2`的新类LOO截断，同时维持旧fit support不退化。
- 比较：Z0、D25-C0、B3、D33-FAST、D35-A/B/C；7候选×3场景×5个独立held-rank折=105行。
- 本轮K10-only；K1/K5只是接口边界，不是执行或性能证据。

## 机制与协议

完整公式和实现追踪见`analysis/d35_dense_safe_registration_traceability_20260718.md`。同一已接收LEO_weak IQ仅生成一行288维`[z160,FFT96,RF32]`拼接描述；不新增信道overlay、physical sample、support row或K。query完全关闭；clean/source样本及未授权衍生信号不可达；无角色Oracle、真实batch类数、quota、global assignment或dense query图。

D35的所有新类对每个旧winner始终有有限score；winner只索引一个由旧support构造的安全阈值。A使用单mean原型；B/C使用最多2个确定性原型；C对旧floor winner加倍不确定度buffer。所有新类原型为int8+FP32 scale/inverse norm；旧FAST score前缀不修改。

## 成功标准

- full-K fit旧support逐类/floor不退化；15个outer held折旧类new intrusion全部为0。
- 三场景所有新类physical LOO margin_min>0，重点检查09f8和f608。
- 联合指标达到B3与D33-FAST门槛，且不转移旧floor损失。
- 0 optimizer step、active<=50k、状态<=50kB；相对identity qKNN、B3、D33报告MAC/延迟/状态Pareto。
- 即使D35注册成功，FAST注册前旧类82.22%仍低于正式92%目标；不得把Stage2-C成功描述为最终路线完成。

## 执行计划

复用D34同一密封support与receiver/seed/scenario，不新增数据准备。完成core、runner、launcher和测试后先Git提交，再执行N607直接preflight、live inventory、最小同步、SHA闭合和唯一输出检查；计划GPU0，唯一输出`runs/d35_dense_safe_20260718/output/support_screen_v1`。完成后回填105行、逐类/场景矩阵、old intrusion、新类LOO、完整日志、资源审计、RECEIPT和Git提交。
