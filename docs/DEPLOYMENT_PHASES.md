# 在轨部署Phase

部署阶段的目标域是`R_t`。`R_t`必须与地面训练接收机域`R_s`不相交。target-old、target-new和unknown的support/query都必须来自同一个目标接收机域定义，并在同一个satellite/LEO target view下报告。

## Stage2-A：zero-label deploy

```text
support: empty
query:
  target-old query from Y_old on R_t
  non-old/unknown query from Y_new or Y_unknown on R_t
```

允许声明：

- old-class target recognition
- non-old rejection
- unknown FAR、FPR95、AUROC

禁止声明：

- new identity recognition
- target-label threshold fitting
- seen-new accuracy

Stage2-A是部署安全底线，不是新类注册。

## Stage2-B：old-class few-shot calibration

```text
support:
  K shots per old TX from Y_old on R_t
query:
  held-out target-old query from Y_old on R_t
  target-new/unknown rejection query from Y_new/Y_unknown on R_t
```

允许声明：

- target-old full accuracy
- target-old accepted accuracy and coverage
- old_acc_delta_pp
- rescue / harm / net_gain
- unknown FAR、FPR95、AUROC不恶化

禁止声明：

- seen-new identity accuracy
- target-new support使用
- unknown query参与阈值拟合

## Stage2-C：old + seen-new enrollment

```text
support:
  K shots per old TX from Y_old on R_t
  K shots per seen-new TX from Y_new on R_t
query:
  held-out target-old query from Y_old on R_t
  held-out seen-new query from Y_new on R_t
  unseen-new/unknown query from Y_unknown on R_t
```

允许声明：

- target-old performance
- seen-new identity accuracy
- `H_old_new`
- unknown FAR under constraint
- old/new/unknown confusion and defer/uncertain behavior

禁止声明：

- 把`Y_unknown`当seen-new识别
- 用unknown query调阈值
- 把clean-view success写成deployment success

Stage2-C只有在`Y_new`与`Y_old`不相交、`R_t`与`R_s`不相交，并且target-old与target-new support/query都来自`R_t`时才成立。

## 指标

| 阶段 | 主指标 |
|---|---|
| 地面训练 | strict UDU、receiver floor、pseudo-label precision/coverage、`z_id -> receiver` leakage probe、satellite stress mean/floor |
| Stage2-B | target-old full accuracy、accepted accuracy、coverage、old_acc_delta_pp、unknown FAR、FPR95、AUROC、rollback trigger |
| Stage2-C | seen_new_acc、old_acc、`H_old_new`、unknown FAR、new_acc_drop_pp、old->new/new->old/unknown->new confusion、latency/memory/prototype storage |

报告必须保留同一candidate/run的完整指标上下文。不能把来自不同row的最大/最小值拼成一个“最佳实验”。
