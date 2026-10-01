# Margin 单样本 prior 复用实现与验证

新增接口已通过 22 项限定合成验证。它根据固定关系 `f_C(x)=P f_B(x)+r_C(x)` 复用同一样本刚完成的 B 分数，减少 C 内一次重复 B 求值。它不改变训练、模型状态或分类规则，没有接入当前远端基准，也不代表准确率改善或已测星载耗时收益。

数学条件见[静态设计](D92_MARGIN_SINGLE_QUERY_REUSE_DESIGN_20261001.md)，训练方法原理见[联合方法解释](D92_MARGIN_JOINT_USER_EXPLANATION_20261001.md)。本次实现是软件计算路径改进，没有从目标评分选参数、重新拟合或训练新方法。

## 使用接口

```python
scores_B, work_B, packet = b.score_single_for_reuse(**one_five_branch_sample)
scores_C, work_C = c.score_single_with_reused_prior(packet, **one_five_branch_sample)
```

两个接口位于[生产 core](../code/cvsrffi/d92_margin_joint_local_ridge.py#L924)。每次仅接受一条完整五分支输入；packet 捕获输入和完整原始 float64 B 分数，只在相同实际 B 对象、同一输入和完整注册类映射下有效。C 先求完整 residual，再按现有旧列次序相加；新增列保留 residual。new0 的 C 就是 B，不再调用 residual。

数组缓冲只读之外，还核对 shape、dtype、strides、数据地址及底层缓冲身份；B 的冻结数值标量按实际类型和值绑定。可写的固定字典保存小型数值快照，不复制完整支持集。对象替换、评分视图变化、字典变化、错误 B、空或多样本输入均拒绝复用。失败保留已消耗 residual 工作，不输出完整 C，也不回退为 B。

## 验证与资源口径

root 在实际激活的 `ssr-gpu` 中串行完成 22 项测试。合成验证涵盖普通 C 的完整分数和列映射逐位相等、输入与分数副本捕获、过期状态拒绝、new0、tau=0、gamma=None/0、自由截距、实际生产 K1 B→C/new0、失败计费和调用次数。状态使用真实 production core 构造的 K1 部分与人工合法评分展开明确区分；不冒充实际数据性能实验。[完整验证记录](D92_MARGIN_SINGLE_QUERY_REUSE_VALIDATION_20261001.json)保留最初生产标量兼容失败及修正后的证据。[测试源码](../tests/test_d92_margin_joint_single_query_reuse.py)可复核。

与 parent `9d5c06ac8113cf09999091b400a1995dcabc1af5` 比较，现有顶层函数及类在移除两个新增可选方法后，AST 完全一致。默认 `score_with_audit` 仍按原路径执行；冻结配置和当前 benchmark 入口未修改。限定检查通过后没有重复完整实验审查、数据重验或全套测试。

| 项目 | 本次口径 |
| --- | --- |
| 新执行工作 | B 计费一次；C 仅记录自己的 residual 和合并工作 |
| 复用工作 | C fresh prior 求值次数为 0，reused prior 样本数为 1；不把 B 历史工作再次相加 |
| 计时 | API 保存 packet 准备、复用核对和实际 C 调用时间；本次未做设备性能基准，节省秒数为 N/A |
| 训练 | 本次不更新模型参数；验证中的实际生产 K1 head 拟合只使用合成 support |
| 传输 | 本接口新增地面数据和聚合摘要载荷为 0 B；部署代码传输及实际线路字节数 N/A |
| 内存与星载 | 捕获单样本及小字典产生额外本地状态；整进程峰值、设备能耗和星载收益 N/A |

只有同时输出 B/C 的路径才有已完成 B 分数可复用。仅部署 C 时，仍须计算一次 B prior，不能据本次消除重复调用宣称单独 C 推理也减去一次必需 B。K1、rank0、tau0、gamma None/0 均不省略 C 的完整 residual 或自由常数。

当前唯一实际 query run 仍为 `20261001-phase2-d92-margin-joint-repeat-m2-r01`，immutable runtime `d86edc3235ec0ca0e1925223970e07373256fb9c`。该运行继续采用原接口，不热改、不重启、不重跑。全部四行 A/B/C 固定后才由独立 scorer 连接 truth；终态对照按[配对方案](D92_MARGIN_QUERY_BASELINE_PAIRING_PLAN_20261001.md)执行。
