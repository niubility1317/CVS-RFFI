# D22极轻型资源上限同步记录

日期：2026-07-17

## 用户授权

当前活动目标把一般极轻量adapter资源上限更新为：可训练参数不超过80,000、适配不超过30epoch、无dense query图。

## 项目同步

`项目.md`第10.3.1节已同步为一般adapter不超过80,000参数、30epoch和256KB持久状态。用户随后再次明确“5epoch太少”，因此第7.1节的`support-only sparse key-layer delta`也同步采用不超过80,000可训练参数、30epoch和50 optimizer step；仍优先SGD无momentum，FP16 patch与head合计不超过256KB。

用户进一步允许为了探索最佳性能将资源上限放宽50%。因此新增独立`PERFORMANCE_EXPLORATION_150PCT`档：不超过120,000参数、45epoch、75 optimizer step和384KB状态。该档只能用于机制上界和压缩目标，不能直接进入正式125确认矩阵或部署声明；正式晋升仍需压回80,000/30epoch/50step/256KB，或再次取得用户明确授权。

本次只修改资源口径，不改变以下硬约束：

- Phase2输入仍为单一`LEO_weak`接收观测，clean与未授权clean/source-derived信号不可达；
- query仍只用于锁定方法后的隔离测试，不得进入适配、选参、早停、回滚或排名；
- 每个query仍逐样本面对全部已注册类，无角色Oracle、真实批类别数、类别配额或global assignment；
- 域适应与新类注册仍须在同一row等权报告注册前后指标、逐类floor和遗忘。
