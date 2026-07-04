# CVS Stage2-C KERA-CI算法说明

## 定位

KERA-CI（Known Enrollment Repair Adapter Collaborative Inference）是AOR-Adapter-CI后的修复路线。AOR证明旧类锚点优先可以提升旧类均值，但会把已注册seen-new和unknown都吸收到旧类，导致`seen_new_acc=0`、`unknown_reject=0`。KERA只改变融合顺序：先判断已注册seen-new是否满足known enrollment，再执行old-anchor guard，最后才做unknown拒识。

## 协议边界

- 仍使用`ADV3B02_CORE90_SOFT_E200`冻结特征和support-only轻量适配代理。
- adapter拟合只使用`target_old/target_new`support。
- `target_unknown`只进入最终评测，不参与adapter、阈值、profile选择、可靠性估计或伪未知构造。
- profile为预注册诊断组合，不能根据本轮unknown结果反复调参后声称sealed evaluation。
- `same_max_budget`下不同M可能对应不同`event_count`，不能写成严格同分母协同收益。
- 资源字段仍是`resource_proxy`，不能声明真实星载链路端到端预算通过。

## 融合顺序

每个receiver输出AOR同构证据：

```text
top_label
top_label_set in {old, seen_new}
known_score
prototype_score
old_anchor_score
margin
unknown_score
quality
```

事件级融合：

1. `seen_new_enrollment`：若seen-new候选有足够known score、margin和receiver vote，且不明显弱于old候选，则优先接收seen-new。
2. `old_anchor_guard`：若seen-new未通过，再用old anchor保护高置信旧类。
3. `unknown_reject`：只有seen-new和old均未通过，才根据unknown evidence、低margin和receiver disagreement拒识。
4. `known_accept/defer`：其余情况按known候选接收或defer。

## 与AOR的差异

| 项 | AOR | KERA |
|---|---|---|
| 第一门控 | old anchor | seen-new enrollment |
| 主要修复对象 | unknown拒识前保护旧类 | 修复seen-new被旧类吞并 |
| unknown门控 | old guard失败后触发 | known enrollment整体失败后触发 |
| 预期收益 | old mean提高 | seen-new不再被old-first系统性压制 |

## 验收口径

KERA本轮不是放宽最终目标。完整目标仍要求同一行同时满足：

```text
old_acc>=0.99
min_old>=0.95
seen_new_acc>=0.97
min_seen>=0.93
unknown_reject>=0.99
resource_proxy_pass=true
target_pass=true
```

若KERA只提升seen-new但导致old下降或unknown仍失败，只能写作`NON_DEPLOYMENT_DIAGNOSTIC`。
