# D22极轻型资源上限同步记录

日期：2026-07-17

## 用户授权

当前活动目标把一般极轻量adapter资源上限更新为：可训练参数不超过80,000、适配不超过30epoch、无dense query图。

## 项目同步

`项目.md`第10.3.1节已同步为一般adapter不超过80,000参数、30epoch和256KB持久状态。用户随后再次明确“5epoch太少”，因此第7.1节的`support-only sparse key-layer delta`也同步采用不超过80,000可训练参数、30epoch和50 optimizer step；仍优先SGD无momentum，FP16 patch与head合计不超过256KB。

本次只修改资源口径，不改变以下硬约束：

- Phase2输入仍为单一`LEO_weak`接收观测，clean与未授权clean/source-derived信号不可达；
- query仍只用于锁定方法后的隔离测试，不得进入适配、选参、早停、回滚或排名；
- 每个query仍逐样本面对全部已注册类，无角色Oracle、真实批类别数、类别配额或global assignment；
- 域适应与新类注册仍须在同一row等权报告注册前后指标、逐类floor和遗忘。
