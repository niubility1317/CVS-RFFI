# CVS论文级对比实验协议

## Phase1域泛化

统一使用ManySig、`rho_label=0.1`、source receivers与held-out receivers不相交、训练期source-only。主比较方法为CVS、CVCNN-CE、RIEI-FD和DRIFT。

CVCNN-CE锁定为三层常见复值CNN：通道`32/64/128`，每层`ComplexConv1d-BN-ReLU-AvgPool`，128维embedding，线性TX分类头；AdamW，学习率`2e-4`，weight decay`1e-4`，batch size64，200epochs，cosine schedule至`1e-6`。关闭Sinc、伪标签、receiver head和训练期卫星增强。

主结果使用`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`独立随机压力评估。clean仅作control。验证集只用于选模；held-out receiver/day及LEO测试只在预登记检查点执行，论文主统计不得挑选测试最佳epoch。

## Phase2监督域适应

统一比较CVS、ProtoNet CDA、MRIOR-SDA和DADDA-SDA。三种对比方法共享同一source checkpoint、target receiver、target-old TX集合、K-shot support索引、held-out query索引、seed、LEO视图和适应预算。

- ProtoNet CDA：support均值形成prototype，query不反传。
- MRIOR-SDA：保留GAD和DV-KL；target support真实标签进入target CE。真实标签优先于CPL伪标签。没有额外无标签训练池时关闭CPL并单列消融。
- DADDA-SDA：加入target CE；LMMD对support使用真实标签。若另有未标注目标训练池，必须标为半监督扩展并与纯K-shot监督结果分表。

K主曲线为`{1,2,5,10,20}`，`K=50`只作higher-shot/saturation诊断。Stage2-B不报告`seen_new_acc`。

## Phase2类增量/新类学习

统一比较CVS、CSIL、MoPC-HR和Orthogonal Incremental SEI。所有方法使用同一`R_t`上的target-old与target-new support/query；`Y_old`与`Y_new`互斥，query不参与更新或选模。

主指标为`old_acc`、`seen_new_acc`、`H_old_new`、old-to-new/new-to-old混淆、平均遗忘、参数更新量、prototype存储和适应延迟。unknown/open-set属于Phase3备用项，不进入本组主排序。

## 统计方案

- 正式主表使用5个预登记seed；单seed只作启动/机制检查。
- 每个seed复用完全相同的数据manifest，并在方法间形成paired comparison。
- 报告mean、standard deviation、95% bootstrap confidence interval和paired effect size。
- 多方法比较先做整体检验，再做预登记pairwise comparison并进行Holm校正。
- 同一方法的K-shot曲线同时报告area under adaptation curve；不得只挑最优K。
- 所有结论绑定完整candidate/run/seed行，不拼接不同实验的单项最值。
