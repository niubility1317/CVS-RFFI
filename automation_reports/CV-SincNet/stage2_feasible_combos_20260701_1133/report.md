# Stage2可行TX组合审计报告

- 实验/审计ID：`stage2_feasible_combos_20260701_1133`
- 时间：2026-07-01
- 操作者：Codex
- 目标：全面统计当前Phase2协议下，ManyTx中可作为seen-new与unknown的TX组合规模，并说明若修改源域训练类别，应如何选择更合理的组合。

## 审计口径

- 数据源：N607上的`Dataset_WigSig/ManyTx.pkl`
- target receiver：`20-1,3-19,7-14,7-7,8-8`
- 当前旧类`Y_old`：`14-10,14-7,20-15,20-19,6-15,8-20`
- 当前固定seen-new：`1-16,1-18`
- Stage2-C默认组合大小：`|Y_new|=2`，`|Y_unknown|=2`
- K网格：`2,3,5,10`
- query-per-TX：`30`
- 可行性定义：
  - seen-new TX：每个target receiver下样本数`>=K+30`
  - unknown TX：每个target receiver下样本数`>=30`
  - TX不能属于`Y_old`
  - `Y_new`与`Y_unknown`必须互斥

## 工件

| 文件 | 含义 |
|---|---|
| `tx_availability_by_domain.csv` | 每个target receiver下每个非旧TX的样本数与可用性 |
| `seen_new_pairs_by_domain_k10.csv` | K=10时每个target receiver可行的全部seen-new二元组合 |
| `common_seen_new_pairs_k10.csv` | K=10时五个target receiver共同可行的seen-new二元组合 |
| `unknown_pairs_fixed_new_by_domain.csv` | 固定seen-new=`1-16,1-18`时每个target receiver全部unknown二元组合 |
| `common_unknown_pairs_fixed_new.csv` | 固定seen-new=`1-16,1-18`时五个target receiver共同可行的unknown二元组合 |
| `combo_count_summary.csv` | 按domain/K汇总的组合计数 |
| `feasible_combo_summary.json` | JSON摘要，含协议、各域计数、共同可行TX列表 |

## K=10组合规模

| scope | domain | seen-new可用TX | seen-new二元组合 | unknown可用TX | 固定seen-new下unknown二元组合 | seen-new/unknown均可变且互斥的组合数 |
|---|---|---:|---:|---:|---:|---:|
| 单域 | `20-1` | 144 | 10,296 | 144 | 10,011 | 103,073,256 |
| 单域 | `3-19` | 137 | 9,316 | 138 | 9,180 | 85,520,880 |
| 单域 | `7-14` | 135 | 9,045 | 136 | 8,911 | 80,599,995 |
| 单域 | `7-7` | 138 | 9,453 | 139 | 9,316 | 88,064,148 |
| 单域 | `8-8` | 135 | 9,045 | 136 | 8,911 | 80,599,995 |
| 五域共同 | `ALL` | 122 | 7,381 | 125 | 7,503 | 55,379,643 |

结论：从样本可用性看，Stage2-C可行组合极多。若只固定当前seen-new=`1-16,1-18`，仍有7,503个五域共同可行unknown二元组合。若seen-new也允许变化，五域共同可行的seen-new二元组合有7,381个；seen-new/unknown二者均可变且互斥时，K=10下有55,379,643个组合。

## K变化

五域共同范围内，K=2/3/5/10的计数相同：

| K | seen-new可用TX | seen-new二元组合 | unknown可用TX | 固定seen-new下unknown二元组合 | seen-new/unknown均可变且互斥的组合数 |
|---:|---:|---:|---:|---:|---:|
| 2 | 122 | 7,381 | 125 | 7,503 | 55,379,643 |
| 3 | 122 | 7,381 | 125 | 7,503 | 55,379,643 |
| 5 | 122 | 7,381 | 125 | 7,503 | 55,379,643 |
| 10 | 122 | 7,381 | 125 | 7,503 | 55,379,643 |

这说明当前K≤10时，样本数量不是主要限制；主要限制是开放集几何难度和训练/评估计算量。

## 已测试unknown组合的位置

| seen-new | unknown | 结论 |
|---|---|---|
| `1-16,1-18` | `10-1,10-10` | 比较容易；q95 strict FAR为41.25%，仍偏高 |
| `1-16,1-18` | `1-10,1-12` | 更难；q95 strict FAR升至70.83%，说明同一`1-*`族更容易被seen-new/old接受区吸收 |

因此，选择unknown不能只看是否互斥。`Y_unknown`与`Y_new`如果来自相近TX族，会显著增加FAR。

## 如何选择组合

### 1. 如果目标是可比主线

保持旧类`Y_old`不变，先固定seen-new=`1-16,1-18`，然后从`common_unknown_pairs_fixed_new.csv`里挑选unknown组合做sweep。这样旧类和seen-new不变，Unknown FAR变化可以直接归因于unknown几何位置。

推荐至少三档：

| 难度 | 组合原则 | 已有例子 |
|---|---|---|
| easy/control | unknown与seen-new不同前缀族 | `10-1,10-10` |
| hard/stress | unknown与seen-new同前缀族 | `1-10,1-12` |
| balanced | unknown跨两个不同前缀族 | 待从CSV中抽样 |

### 2. 如果seen-new也要变化

使用`common_seen_new_pairs_k10.csv`作为seen-new候选池，使用`common_unknown_pairs_fixed_new.csv`或由`tx_availability_by_domain.csv`重建unknown池。不要直接枚举55M个完整四元组到报告正文；应以pair池+互斥规则表示：

```text
Y_new = any pair from common_seen_new_pairs_k10.csv
Y_unknown = any pair from common eligible unknown TX
constraint: intersection(Y_new, Y_unknown) = empty
```

### 3. 如果修改源域训练类别

当前ManySig源域旧类只有六个固定旧TX。若要修改源域训练类别，本质上不是只改Stage2参数，而是要重定义`Y_old`并重新训练地面主干。此时“最好”的组合不应按样本数选，因为样本数在K≤10已不是瓶颈，应按特征几何和任务目的选。

建议按以下原则：

1. `Y_old`覆盖部署中最可能反复出现的TX族。当前`Y_old`覆盖`14-*、20-*、6-*、8-*`，没有覆盖`1-*`和`10-*`，所以`1-*`/`10-*`进入Stage2时更依赖后置表，拒识更不稳。
2. 如果目标是降低FAR，不要让`Y_new`和`Y_unknown`来自同一个前缀族。例如seen-new=`1-16,1-18`时，unknown=`1-10,1-12`明显比unknown=`10-1,10-10`更难。
3. 如果目标是做强压力测试，应该故意选择同族unknown，例如`Y_new=1-*`、`Y_unknown=1-*`，但报告中要标为hard open-set stress。
4. 如果目标是部署性能，建议把预期会被识别/注册的同族TX尽量纳入`Y_old`或`Y_new`，不要把同族相邻TX留作unknown，否则拒识边界会非常难。
5. 重训前应先做feature几何预审：类中心距离、p95/p99半径、receiver shift、leave-one-TX-out伪未知FAR。仅凭TX label和样本数不能确定最优`Y_old`。

可操作的下一步：

- 保持当前`Y_old`，先做unknown sweep，找出最易/中等/最难的unknown组合。
- 再做seen-new sweep，固定几个unknown难度档，评估seen-new组合本身是否扩大接受区。
- 最后才改`Y_old`重训；否则无法判断失败来自地面主干、seen-new注册还是unknown本身。
