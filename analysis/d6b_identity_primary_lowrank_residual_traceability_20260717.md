# D6b identity主分数＋有界低秩残差可追溯说明

日期：2026-07-17

## 机制

D6b不再用低秩投影替换288维identity表示。每个query先计算完整identity cosine分数`s_id`，再计算support-only低秩分支分数`s_lr`，最终分数固定为：

```text
s = s_id + alpha * (s_lr - s_id)
alpha in {0, 0.1, 0.2, 0.3}
```

因此`alpha=0`严格等于identity基线，低秩分支只能提供最大30%的有界残差。

## support-only选择门禁

rank、shrinkage与`alpha`只使用每scenario注册support的逐类leave-two-out；K不足4时退化为leave-one-out。每个候选必须在三个scenario分别满足：

1. 最低类support删除法准确率不低于对应identity基线；
2. support总体准确率相对identity基线下降不超过1pp。

固定网格必须包含`alpha=0`，保证不满足非退化约束时自动回退identity。三个scenario使用统一arm，但分别拟合状态，禁止support拼接。

## before/after

before拟合完整identity原型和support-only低秩分支。after保持before的identity旧类原型、低秩投影、低秩旧类原型、rank、shrinkage与`alpha`bitwise不变，只为registry中原先不存在的注册标签增加两条分支的原型。不存在old/new角色参数；是否新增仅由注册类registry成员关系判断。

## 协议和资源

- 三scenario物理support ID和接收IQ SHA必须两两不交。
- 无clean/source、query标签、role Oracle、类别quota、batch-global assignment或dense query graph接口。
- query只进行逐样本全注册类推理，不更新任何状态。
- 0epoch闭式适配；288维、rank32、三scenario投影参数27,648，低于80,000；持久状态运行时强制低于256KB。

## 当前证据边界

模块和直接测试已完成。当前已有`receiver=20-1,seed=713101,K10,new5`query在D4/D5阶段被评分，因此不再用该query声称D6b独立开发性能，也不据其结果选择`alpha`。D6b真实性能必须使用新的、此前未评分的development切片，或在预先锁定arm后进入独立确认。

## Fresh未评分holdout实测

后续从同一合法cache的每类每scenario 40个独立物理样本中构造fresh holdout：

- 原K10 support仍使用rank0–9；
- 既有已评分query不读取、不复用；
- enrollment pool中从未进入K10拟合且未被评分的rank10–19，离线拆分为每类每scenario 10个fresh query；
- 离线拆分先生成不带标签的predictor query artifact与独立truth sidecar；
- predictor只读取K10 support和无标签fresh query，先写入immutable prediction artifact、execution receipt和COMMIT；
- scorer在COMMIT SHA核验后才首次打开truth sidecar。

support删除法结果显示，48个组合中只有12个`alpha=0`组合通过三scenario最低类非退化和总体1pp容忍门禁；所有`alpha∈{0.1,0.2,0.3}`组合至少在一个scenario违反门禁。因此统一arm回退：

```text
rank=8
shrinkage=0.9
alpha=0
```

其中rank/shrinkage在`alpha=0`时不影响identity预测，仅保留低秩分支状态用于审计。fresh首次评分结果：

|指标|D6b fresh holdout|
|---|---:|
|注册前old_acc|82.78%|
|注册前旧类floor|53.33%|
|注册后old_acc|63.33%|
|注册后旧类floor|46.67%|
|seen-new_acc|66.67%|
|新类floor|26.67%|
|H_old_new|64.96%|
|旧类遗忘|19.44pp|

资源：三个scenario合计6,912个低秩投影参数，before状态53,640B，after状态72,420B，0epoch、query fit/update为0、dense query graph为0。

结论：D6b非退化门禁正确阻止了有害低秩残差，但`alpha=0`只解决“低秩替换伤害”，没有解决新旧identity原型碰撞；after即使bitwise锁定旧类状态，新增类仍可在全类argmax中夺走旧类query。因此D6b不晋升。下一机制若继续，必须用support-only的局部碰撞边界或类条件margin约束新类注册，而不能放宽本轮已经明确失败的残差门禁，也不能据fresh truth回选`alpha`。
