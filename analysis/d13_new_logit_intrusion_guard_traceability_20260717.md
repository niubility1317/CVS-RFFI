# D13类条件new-logit侵入保护追踪

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D13-01 | `项目.md`§7.1/7.1.1 | 只读取sealed单一LEO_weak received-IQ；无clean/source/query；逐物理sample实际IQ SHA绑定 | D13 runner、artifact audit | verified | v3 pre-open、逐sample provenance、跨场景token/hash互斥 | 复用D12 runtime-authorized artifact安全模式，不新增LEO状态 |
| D13-02 | `项目.md`exact-K | Before旧6类、After旧6+新5类只读取严格K10 enrollment-only包 | D13 runner/tests | verified | 13项聚焦测试与v3真实包 | K5伪装K10反例fail closed；K1/K5未切片 |
| D13-03 | D12结论/主任务 | Before old cosine/prototype不变；After只追加new prototype并仅对new logit减类条件penalty | D13 module/tests | verified | old prototype与old score列逐位相等测试 | v3所有fold记录`old_score_columns_bitwise_unchanged=true` |
| D13-04 | 主任务公式 | `old_risk_c`与`new_room_c`仅由support fold-train margin分布计算，`delta_c`使用固定quantile/safety/cap/floor-margin规则 | D13 module/tests/log | verified-negative | 固定`linear`分位数、Before-correct old门、shortfall审计 | 正delta最多0.003656，机制实质不足 |
| D13-05 | 主任务joint K10 L2O | 每fold old/new各held2；held old/new均面对全部注册类；报告old/new/joint/H/floor/forgetting | D13 module/tests/audit | verified | 6候选×3场景×5fold；held2变异反例 | held变异不改变fold learned penalty、诊断或selection SHA |
| D13-06 | 主任务统一选择 | 三场景统一超参；硬门old逐类vs Before/base非退化、forgetting≤0，new overall/floor相对alpha0和D11同口径不退化 | D13 runner/audit | verified-negative | lock SHA`74feab0c...a9e5f`；全部正arm三场景门false | D11-v6 COMMIT/audit/report实读哈希绑定 |
| D13-07 | 主任务安全回退 | 全部正delta候选失败时真实回退`delta=0`并输出NO-GO | D13 module/runner/tests/COMMIT | verified | v3 selected=`d13_delta0_base` | `SUPPORT_ONLY_D13_NOT_SELECTED_NO_QUERY_OPEN` |
| D13-08 | `项目.md`§7.2 | formal预测逐sample面对全部注册类；无role/quota/query fit/global assignment/dense graph | D13 formal API/tests | verified-no-query | 单行全类API与runtime/operator反例 | v3 query/truth/prediction/score/scorer均false |
| D13-09 | D12安全模式 | state readonly、content SHA、runtime/code/checkpoint/operator/view/support selection绑定；普通Mapping/array拒绝 | D13 module/tests/runner | verified | readonly/tamper/mapping/actual-IQ SHA与COMMIT绑定 | 模块无`_select_artifact`/`_ARTIFACT_TOKEN`引用 |
| D13-10 | 资源硬门 | 0参数、0epoch、总state≤256KiB；new5 penalty增量优先几十B；报告MAC/延迟/显存/状态Pareto | D13 module/runner/audit | verified | After序列化18,452—18,481B；数组12,804B；guard增量60B | MAC与数组state相对qKNN约-90%；Python安全包装延迟不是正式query latency |
| D13-11 | 实验要求 | 使用D8b真实三场景strict K10包跑support-only；formal artifact/state/COMMIT完整；不打开query | D13 runner/artifact/report | verified-not-selected | v3 COMMIT file SHA`2d2d7874...b29860` | claim scope仅development support-only |
| D13-12 | AGENTS.md/Git | 独立新文件，`ssr-gpu`本地验证，检查diff；不stage/commit | tests/trace/handoff | verified-uncommitted | 13/13 pytest、py_compile、diff-check | 共享脏树中仅新增5个D13文件 |

## 预登记候选与选择口径

所有场景共享同一个`old_risk_quantile/new_room_quantile/safety/cap/new_floor_margin`组合。每个场景、每个fold允许依据其合法fold-train support独立计算每个新类的`delta_c`，但不得按场景切换超参数。

候选必须显式包含`delta=0`。正delta候选只有在三个场景同时满足以下条件时才可被选择：

1. After old逐类准确率不低于Before old和alpha0 base；
2. `old_forgetting<=0`；
3. After new overall和floor均不低于同fold alpha0以及D11-v6同口径参考；
4. `H_old_new`与joint accuracy不低于alpha0；
5. 状态、epoch、参数和逐sample全类决策资源边界通过。

全部正delta候选失败时，最终状态必须真实保存`delta=0`，状态为`SUPPORT_ONLY_D13_NOT_SELECTED_NO_QUERY_OPEN`。

## 公式红队与正式范围

早期探索过逐样本`hinge_margin`，但其可行性不能直接继承constant delta的`requested<=cap/room_bound`证明：当strength小于1时，在old-risk分位点的实际扣减不足，且constant new-room bound不等价于hinge不伤new。因此正式runner候选只保留constant penalty；hinge不进入真实选择、v3 COMMIT或任何promotion判断。

正式constant校准只使用Before-correct old fold-train样本：

```text
old_risk_c = Q_qo(s_c - max_old s_old)
new_room_c = Q_qn(s_c - max_other s_other)
requested_c = max(0, old_risk_c + safety)
room_bound_c = max(0, new_room_c - new_floor_margin)
delta_c = min(requested_c, cap, room_bound_c)
```

每类同时记录`protection_feasible`与`protection_shortfall=requested_c-delta_c`。任一fold-class存在正shortfall时，该正candidate的场景门失败。

## v3最终support-only证据

最终artifact：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d13_new_logit_intrusion_guard_v3_final`

- COMMIT file SHA：`2d2d7874a66660233189b0c3e5e66545d2f26c4f3b5c9dbf878ef7e923b29860`
- module SHA：`217d90572d263f47af9d442ba5347ec8839958097f514c5bc7e2add2b713ce3b`
- runner SHA：`f6aa801793e39e7ce4c7a7d0665093fd7f669a353cc7937a727e3d529ee28a14`
- unified lock SHA：`74feab0c409ffb2be2485307cf9cf17d851f8847d1b53179b3d68ab6770a9e5f`
- D11-v6参考绑定：COMMIT`d9f0a0af...58f2`、support audit`75bf5a61...f1e4`、report`eb58d6d9...d740`

|场景|Before old|After old/floor|seen-new/floor|joint/H|forgetting|结论|
|---|---:|---:|---:|---:|---:|---|
|`leo_clear_weak`|0.7167|0.6333/0.1000|0.5000/0.2000|0.5727/0.5588|0.0833|拒绝|
|`leo_low_elev_weak`|0.7333|0.6833/0.2000|0.4600/0.1000|0.5818/0.5499|0.0500|拒绝|
|`leo_rain_weak`|0.7333|0.6167/0.3000|0.6200/0.4000|0.6182/0.6183|0.1167|拒绝|

完整`training_log.jsonl`共276行：90行joint fold summary全部含candidate ID、mode和统一lock SHA；全数值finite。所有校准诊断中只有7个delta非零，最大值0.003656；98个fold-class存在正protection shortfall。常数new-logit penalty没有足够可行room恢复old遗忘，故真实回退delta0、不开放query。

## 最终追踪计数

- verified或verified-negative：12
- deferred：0
- rejected：1（hinge正式候选，公式门不成立）
- blocked：0

最高风险剩余项：D13只抑制new score，无法修复Before旧类内部混淆；真实support中old-risk大多为负且new-room接近0，常数penalty不能在保持new逐类非退化时恢复5.00—11.67pp旧类遗忘。下一机制必须在不修改old logits的条件下使用更精确的new-vs-old pairwise/local score shape，或先提升Before old floor；当前D13不得进入query或125确认矩阵。
