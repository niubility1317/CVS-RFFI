# 在轨部署Phase

部署阶段的目标域是`R_t`。`R_t`必须与地面训练接收机域`R_s`不相交。Phase2主线要求target-old和target-new的support/query都来自同一个目标接收机域定义，并在同一个satellite/LEO target view下报告。open-set/unknown拒识自2026-07-07起作为Phase3备用项，不是Phase2主线。

## Stage2-A：zero-label deploy

```text
support: empty
query:
  target-old query from Y_old on R_t
  target-new query from Y_new as non-enrolled reference on R_t
  optional Phase3-backup unknown query from Y_unknown on R_t
```

允许声明：

- old-class target recognition
- target-new non-enrolled reference
- optional Phase3-backup unknown diagnostic

禁止声明：

- new identity recognition
- target-label threshold fitting
- seen-new accuracy
- Phase2 open-set success

Stage2-A是目标域LEO参考底线，不是新类注册，也不是Phase3 open-set结果。

## Stage2-B：old-class few-shot calibration

```text
support:
  K shots per old TX from Y_old on R_t
query:
  held-out target-old query from Y_old on R_t
  target-new query from Y_new as non-enrolled reference on R_t
  optional Phase3-backup unknown query from Y_unknown on R_t
```

允许声明：

- target-old full accuracy
- target-old accepted accuracy and coverage
- old_acc_delta_pp
- rescue / harm / net_gain
- target-old adaptation under LEO target view
- optional Phase3-backup unknown diagnostic

禁止声明：

- seen-new identity accuracy
- target-new support使用
- unknown query参与阈值拟合
- low unknown FAR / high AUROC作为Phase2主线成功

## Stage2-C：old + seen-new enrollment

```text
support:
  K shots per old TX from Y_old on R_t
  K shots per seen-new TX from Y_new on R_t
query:
  held-out target-old query from Y_old on R_t
  held-out seen-new query from Y_new on R_t
  optional Phase3-backup unknown query from Y_unknown on R_t
```

允许声明：

- target-old performance
- seen-new identity accuracy
- `H_old_new`
- old/new confusion and defer/uncertain behavior

禁止声明：

- 把`Y_unknown`当seen-new识别
- 用unknown query调阈值
- 把clean-view success写成deployment success
- 把Phase3 open-set指标写成Phase2主线门槛

Stage2-C是Phase2主线目标。它只有在`Y_new`与`Y_old`不相交、`R_t`与`R_s`不相交，并且target-old与target-new support/query都来自`R_t`且处于satellite/LEO target view时才成立。

## Phase3：open-set backup

Phase3是备用项，只在Phase2旧类适应和新类学习已经具备合法结果，或用户明确要求安全扩展时启用。

允许声明：

- unknown FAR / unknown rejection
- FPR95、AUROC
- open-set confusion
- reject / uncertain / defer behavior
- 加入拒识门控后的old/new性能保留率

禁止声明：

- 用unknown query反向调Phase2阈值、adapter、prototype或主排序
- 用open-set指标替代Phase2的`old_acc`、`seen_new_acc`和`H_old_new`
- 把Phase3备用项写成当前主线

## 指标

| 阶段 | 主指标 |
|---|---|
| 地面训练 | strict UDU、receiver floor、pseudo-label precision/coverage、`z_id -> receiver` leakage probe、satellite stress mean/floor |
| Stage2-B | target-old full accuracy、accepted accuracy、coverage、old_acc_delta_pp、rollback trigger |
| Stage2-C | seen_new_acc、old_acc、`H_old_new`、new_acc_drop_pp、old->new/new->old confusion、latency/memory/prototype storage |
| Phase3 | unknown FAR、unknown rejection、FPR95、AUROC、open-set confusion、old/new性能保留率 |

报告必须保留同一candidate/run的完整指标上下文。不能把来自不同row的最大/最小值拼成一个“最佳实验”。
