# D92 E0连续session追溯

| 需求 | 冻结实现/证据 |
|---|---|
| 一个或少数类分批注册 | `singleton_forward`、`singleton_reverse`、`chunk_2_2_1` |
| 不提前读取未来类 | SessionLedger仅接受当前delta包；测试使用future-open sentinel |
| 不增加K | 每类固定K10物理support token，累计只合并类，不重复样本 |
| 旧类域适应不重复 | `DA1_REG0`锚点只生成一次并在所有session复用 |
| 首个单类立即注册 | 同一auto-shrinkage原语的单类Ledoit-Wolf桥接，固定0.5/0.5 |
| 后续回到原始E0 | S2至S4直接前缀扩展；S5调用原始D92 E0构造器 |
| 终态无顺序效应 | 三种连续调度S5与`batch_5`state/prediction/指标严格相等 |
| query零更新 | 预测入口不接收truth/scorer；fit/update/selection/truth/role/quota/global均false |
| 推理资源不变 | 相同注册类数下D42 state bytes和C×288 MAC与一次性E0相同 |
| 注册资源 | v2：每sessionwall≤300ms；实测增量peak≤4MiB；累计wall另报；已验证：连续session focused 37 passed |
| 指标合法 | scorer只在预测封存后读取truth；未注册真类不计分 |
| 四态命名 | 固定DA1，输出`DA1_REG0`与`DA1_REG1_S*`；DA0=N/A |
| 证据边界 | development-only continuous-session screen，不是Target125推广声明 |
