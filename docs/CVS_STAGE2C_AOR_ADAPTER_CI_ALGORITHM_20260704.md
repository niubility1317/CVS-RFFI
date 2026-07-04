# CVS Stage2-C AOR-Adapter-CI算法说明

## 定位

AOR-Adapter-CI（Anchor-preserving Open-set Receiver Adapter Collaborative Inference）用于`ADV3B02_CORE90_SOFT_E200`冻结特征上的星地信道协同推理诊断。它不是全模型重训，而是在每个target receiver上用`target_old/target_new`的K-shot support拟合一个identity初始化的轻量特征适配代理，再通过旧类锚点优先门控和开放集拒识门控进行M=1..R协同融合。

## 协议边界

- `target_unknown`只用于最终评测，不进入adapter拟合、阈值校准、profile选择或可靠性估计。
- adapter拟合数据仅包含`target_old`和`target_new`support。
- 旧类锚点来自source old prototypes，用于限制旧类漂移。
- 伪未知只由support原型插值/外推构造，不使用真实unknown标签。
- `same_max_budget`下不同M的`event_count`可能变化，因此M曲线是覆盖率诊断，不是严格同分母因果提升证明。
- 当前资源字段是`resource_proxy`，不能声明真实星载端到端链路、调度、加密和重传预算通过。

## 方法

对每个receiver拟合对角适配代理：

```text
z' = normalize((1-alpha) * z + alpha * ((z - center) * scale + center))
```

其中`center/scale`由本receiver的known support估计。旧类support的leave-one-out floor若因adapter下降，则回滚到identity adapter。

每个receiver输出：

```text
top_label
known_score
prototype_score
old_anchor_score
margin
unknown_score
quality
```

协同融合顺序：

1. 旧类锚点门控：只要旧类候选满足`old_anchor/known/margin/vote`下限，优先接收旧类。
2. 开放集拒识门控：只有旧类门控未通过，且unknown evidence、低margin或跨receiver分歧满足预注册阈值，才拒识unknown。
3. known分类：对old/seen-new进行质量加权原型投票。
4. 资源门控：`target_pass=metric_pass and resource_proxy_pass`。

## 预期作用

该路线解决前序OPC/TCSR/APACE/RMD的共同问题：纯决策层阈值在旧类保护和unknown拒识之间出现强冲突。AOR只允许小幅特征几何校正，并用旧类锚点和回滚限制旧类下降。

## 验收口径

正式达标必须同一行同时满足：

```text
old_acc>=0.99
min_old>=0.95
seen_new_acc>=0.97
min_seen>=0.93
unknown_reject>=0.99
resource_proxy_pass=true
target_pass=true
```

若仅提升`unknown_reject`但`old_acc/min_old`下降，结论只能写为非部署诊断。
