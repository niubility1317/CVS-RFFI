# D72物理rank留一联合头bagging追溯与预注册

## 机制定位

D62是当前同row联合最强，但A=82.22%、min-A=53.33%，错误集中于弱场景与少数类。D70证明生命周期整行替换几乎全部回退，D71证明top-2局部pair门只在clear场景启用。D72因此不再增加行、pair或score gate，而检验固定D62 metric下的完整联合头方差是否可由physical-rank leave-one平均降低。

## 追溯矩阵

|要求|D72实现约束|验证证据|状态|
|---|---|---|---|
|LEO_weak-only与K语义|只重用固定support特征；不生成view或物理样本|support multiplicity与Runner审计|PREREGISTERED|
|inner无泄漏|每类每折恰删同一rank；held/train交集0；K折exact-once|partition audit＋测试|PREREGISTERED|
|Stage2-B/C同等|before旧类与final全注册类分别执行同一bagging公式|geometry audit|PREREGISTERED|
|类身份无关|所有匿名类共享相同D62 fit、算术平均和中心化|类置换测试|PREREGISTERED|
|query边界|单query、全部注册类、单一仿射state、无query fit|API与artifact审计|PREREGISTERED|
|int8正式态|平均FP32头重新编译为两级residual-int8/FP16；FP32只作matched诊断|量化误差与零翻转测试|PREREGISTERED|
|地面组件|当前eligible=false，D72输入0且不可更新|geometry/resource字段|PREREGISTERED|
|资源|20个optimizer step不增加；额外闭式fit据实计数；query额外MAC0|resource verifier|PREREGISTERED|
|完整报告|总体、场景、类、fold、机制、训练、量化、资源、artifact、缺陷与比较|report完成门|PREREGISTERED|

## 与历史路线的非重复性

- 不同于D50–D54：不使用median、centroid residual、谱transport或score偏移。
- 不同于D63：不以leave-one证据选择匿名行，不使用TP/FP Pareto门或atomic fallback。
- 不同于D67/D68：不混合D62/D65专家，不做逐行标准化、方向翻转或连续alpha。
- 不同于D69/D70：不冻结/替换生命周期行；before与final分别由同一类对称bagging重新拟合。
- 不同于D71：不做top-2 pair重排，query图和额外MAC均为0。

## 停止边界

真实105行若相对D62有任何联合交换伤害，或遗忘下降仅来自before下降，则路线负向关闭；不扫描bag权重、trim比例、median、bootstrap次数、温度或场景/角色mask。D72结束后先完成D70–D72回顾，再设计D73。

## 实现状态

core、probe和两组测试已完成。专项11/11、D42–D72相邻37文件完整链均通过；K8资源公式锁定每top-level fit新增16次D62 leave-one fit和512次闭式component fit，optimizer step仍为20，query额外MAC/state为0。真实outer尚未执行，不能由测试推断性能。
