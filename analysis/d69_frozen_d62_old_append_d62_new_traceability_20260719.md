# D69冻结D62旧行并追加同族新行追踪

## 要修复的失败

D65以Stage2-B冻结旧行把A提高到86.11%、F降到6.11%，但其block-LDA新行与旧行竞争失衡，N只有59.33%。D68试图逐行修正D65符号和尺度，却在Stage2-B就把B从D62的92.78%压到58.89%，最终N仅14.00%。这证明不能删除D62/D65原始head的绝对跨类尺度。D69只检验生命周期组合，不再做逐行标定。

## 单一数学机制

Stage2-B在6个旧类K-shot support上按D62原公式拟合一个head：

```text
(W_B,b_B)=D62(S_old)
```

冻结`W_B,b_B`的6个旧行。Stage2-C在全部11类support上另按完全相同的D62公式拟合联合head：

```text
(W_C,b_C)=D62(S_old union S_new)
```

最终registry只取`W_C,b_C`的5个新类行，按注册顺序追加到冻结旧行：

```text
W_final=concat(W_B_old,W_C_new)
b_final=concat(b_B_old,b_C_new)
```

没有逐行center/scale、方向翻转、全局温度、old/new offset、角色query分支或参数扫描。所有类在各自D62拟合中使用相同公式；Stage2-C的生命周期只决定哪些已注册行不可改写，query仍逐样本面对全部11类。

## 与matched baseline的唯一差异

相对D62，唯一差异是Stage2-C不采用重新拟合后的6个旧行，而是逐bit保留Stage2-B的6个旧行；5个新行与D62 final完全一致。相对D65，行生成机制从block-LDA换成当前最强D62同族机制；相对D68，不改变任一行的绝对尺度。

## 预期可观察结果与停止条件

- Stage2-B INT8/FP32预测、B、class floor和state必须与D62逐bit/逐row一致。
- Stage2-C旧行FP32及编译后的INT8/FP32旧state字段必须逐bit不变；新行必须与同row D62 final FP32行一致。
- 若A、N、H、J、min-A或min-N相对D62发生交换伤害，直接记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 若新→旧显著增加，说明D62 final新行不能直接与D62 before旧行共用尺度，停止append-only组合；不追加offset、temperature或角色校准。
- 即使首seed通过，也只进入第二development seed，不直接启动125。

## 最小验证矩阵与协议边界

先复用receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8的D18`VALIDATED_ONCE/p2_min_v1`support capsule，执行完整105行和15个D69 INT8＋15个matched FP32目标row。必须报告7候选、3场景、11类、15fold、训练、量化、资源和D62/D65/D66/D67/D68同排差异。

D69不读取ground：D22仍为`formal_phase2_eligible=false`和`UNVERIFIED_UNDER_CURRENT_PROTOCOL`。D66已真实读取84个int8 ground cell但产生N/floor负交换，不能把ground访问次数当作性能贡献。D69的ground、query、clean/source、role、quota和跨query优化访问必须全部为0。
