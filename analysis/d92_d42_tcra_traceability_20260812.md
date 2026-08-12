# D92 D42 TCRA设计追溯（2026-08-12）

## 决策

TPCE v3在真实K10三场景均回退E0：66个pair原子中选择35–39个后，仍有旧类或pooled-new tail落在`[-tol,0)`，registration wall为197–208ms。该路线同时未过科学激活门与150ms资源硬门，不进入Hard10。

候选冻结为`E0_FULL_D42_TAIL_CLASS_ROW_ASCENT`（TCRA）。它保留唯一E0 FULL fit和真实D42 state，只在support侧直接修改`coef2_qint8`；每个原子仅上调真实类的一格，不再同步压低竞争类。

## 可行性摘要（不超过20行）

1. old类`c`的固定tail为E0 support上`true-vs-all` margin的lower-Q20行集。
2. new固定tail为所有new support的`true-new-vs-old` margin lower-Q20；候选按该tail与真实类取交集。
3. tail只从E0计算一次，选择期间不重排、不重扫、不读取query。
4. 每个非空`class×D42 block`只生成一个原子；坐标最大化canonical-row-order下float64`abs(sum(x_ij))`，平局取最小坐标。
5. 原子只执行`q2[c,j]+=sign(sum(x_ij))`；不改competitor、code1、scale、bias或其他state字段。
6. 候选排序固定为新增达标tail覆盖数、最差gain、gain总和、语义handle；无强度或arm扫描。
7. 每个拟接受prefix均materialize真实D42 state并全头评分；不安全候选只被拒绝并移除，搜索有限终止。
8. 每个真实accepted prefix必须相对E0保持六旧tail、pooled-new cross/all和双向hinge不退化。
9. 最终六旧tail与pooled-new cross均严格`>tol`，pooled-new all不降，双向hinge不增；否则byte-exact E0 fallback。
10. K≤2严格D92 FULL alias；query fit/update/selection/truth/role/quota/global reassignment均为0。
11. synthetic C26/K10只读原型generated=78、selected=7、七组全过、双hinge=0，中位22.604ms；仅为可行性证据。
12. 独立监督者裁决`MERGE A`；support guard不能证明held-query八指标，性能只由后续冻结Hard11同排结果裁决。

## 需求到证据

|需求|实现/测试证据|G0证据|
|---|---|---|
|单FULL fit、TPCE后处理fit=0|slim inventory与receipt测试|真实fit audit 2/1|
|row/label置换等变|canonical row handle、semantic class handle测试|state/receipt闭包|
|真实prefix/final D42 guard|负向prefix、最终回退测试|三场景active且无fallback|
|query零访问|query审计负测|七项禁用访问全false|
|低资源|解析MAC/瞬时内存/完整wall收据|三场景registration wall P90≤150ms|

只有G0三场景均active、无fallback、state非E0、完整资源收据过门，才实现并发布Hard11；G0没有truth/scorer，不形成性能结论。

## v1真实G0与v2修订

v1固定K10 G0的三场景wall为121.518–137.051ms，满足150ms资源硬门；但严格“七组全部`>tol`”终门导致三场景均`support_guard_failed`。所有负gain仍在`[-tol,0)`，双向hinge均为0，六旧类gain总和分别为0.012791、0.012760、0.010555。该结果未读取truth或query性能。

因此v2冻结为`TCRA_SAFE_DIRECTIONAL_v2`，并明确这是看到support-side G0后的开发修订，不得宣称v1 G0已通过。保留原E0固定tail、原子、排序、每prefix真实D42守卫和全部协议边界；只将最终门改为：

1. 六旧tail与pooled-new cross的最小gain均`>=-tol`；pooled-new all gain`>=-tol`；双向hinge delta均`<=tol`。
2. 至少一个旧类tail gain严格`>tol`，且六旧类gain之和严格`>tol`。
3. final state非E0、selected atom count>0；否则exact E0 fallback。
4. 门公式、tol、tail、原子和排序在Hard9前冻结，不按scene/outer分支，也不再放宽。
5. 原G0 outer`rx_7_7__seed_713106__k_10__new_5`不得进入v2性能晋级统计。
6. v2先在同一outer执行新的truth-free G0只验证代码/状态/资源；通过后只运行剩余9个最难performance outer并保留K1 liveness。
7. Hard9八项同排均值任一平或负即`REJECT_ROUTE`；support within-tol不得表述为新类收益。
