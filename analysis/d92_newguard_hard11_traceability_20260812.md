# D92 NewGuard Hard11设计追溯

设计源：`E:\codex\home\attachments\a32ff2e7-5e54-4d07-9697-60c470abe165\pasted-text-1.txt`

目标候选：`E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN`

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| NG-01 | 一 | H、old BA、c_old_acc、floor、seen-new全升，forgetting、双向混淆全降 | analyzer/report | FAIL | v3严格复算八项delta全为0 | tie不算成功，裁决`REJECT_ROUTE` |
| NG-02 | 一、九 | query MAC和永久state不增，注册计算显著低于D92 | core/query/analyzer | PASS/FAIL | query/state逐row相等；fit reduction最小95.8333%；wall门失败 | FULL主fit=1 |
| NG-03 | 二 | 不再扫描FloorBoost标量，不用旧类整体加分作主机制 | core/method lock | PASS | 单候选method-lock与静态测试 | 2026-08-12终态纠偏移除20档回缩 |
| NG-10 | 三.1 | 复用E0_FULL_ONLY的288维、task-balanced covariance和单FULL头 | probe/slim | PASS | actual FULL inventory=1 | 禁用BLOCK/OCF/LOO/Fisher/Pareto |
| NG-11 | 三.2 | 增广新类support零空间内构造旧类内部残差 | probe | PASS | 核心零空间与真实probe测试 | 数值秩阈值为机器精度规则 |
| NG-12 | 三.2 | 内部残差旧类组零和，新类行byte-exact | probe | PASS | 置换、零和、字节测试 | 只区分old/new集合，不分具体TX |
| NG-13 | 三.3 | 每类基线最弱20%support固定后执行确定性max-min | probe | PASS | tail冻结、行/标签重排测试 | `method=lower`，无query选择 |
| NG-14 | 三.4 | 共享`tau<=0`降低旧类包络，联合优化双向margin | probe | PASS | raw/deployed margin与包络receipt | 单一固定强度 |
| NG-15 | 三.4、四 | 不可行/秩/有限值失败时精确回退E0 | probe/slim/query | PASS | raw/deployed故障注入与真实D42回退 | 回退不算性能成功 |
| NG-16 | 四 | K<=2保持D92 FULL精确alias | probe/slim/query | PASS | K1/K2回归与K1 liveness | liveness不入性能均值 |
| NG-17 | 四 | 正式int8/FP16部署头重新核对新类保护与tail约束 | core/probe/query | PASS | closure fault injection先RED后GREEN | Xnew、零和、包络、tail、new margin任一失败整头回退 |
| NG-20 | 五 | 输出完整NewGuard数学、资源、fallback和零访问receipt | probe/slim/query | PASS | 跨层receipt漂移负测 | 旧backtrack字段已改为single-candidate字段 |
| NG-21 | 五、六 | query truth/fit/update/selection/role/quota/global全部false | query/runner | PASS | 协议负测+v3真实smoke | scorer后连接truth |
| NG-30 | 七 | 冻结原Hard10+1个K1、3scene、1arm、8shard | config/hard11/runner | PASS | 11job、33scene-arm、8shard PASS | v3为开发矩阵 |
| NG-31 | 七 | 复用历史paired_rows且SHA=`6ebb37...de6a`，并冻结E0 raw score/per-old-class扩展证据 | config/analyzer | PASS | 11 raw score、paired/per-old SHA及truth四方绑定 | 不重跑D92/E0 |
| NG-32 | 七、十 | 唯一runner、K>2真实smoke先于shard、共享技术停派、不可覆盖 | runner/launch/report | PASS | v3 handoff与完整取回 | v1/v2停止，v3唯一launch闭合 |
| NG-40 | 八 | 八项均值严格Pareto，任一反向即REJECT | analyzer/config | PASS/REJECT | 八项delta全0，严格门FAIL | tie正确触发`REJECT_ROUTE` |
| NG-41 | 八.2 | 达成规定幅度才ADVANCE；全正但幅度不足仅REVISE_ONCE | analyzer/config | PASS | 三分支测试 | 未运行Target125 |
| NG-42 | 八.3 | outer、slice、旧类、receiver、scene稳定性门 | analyzer | PASS/FAIL | tie不再计seen-new/new-to-old方向正确 | stability FAIL |
| NG-43 | 九 | median wall<=1.5x E0、p90<=150ms、peak<=E0+512KiB | analyzer/config | FAIL | 30个same-outer/same-scene配对：median ratio=1.74784，p90=179.172ms，peak max delta=5,951,488字节 | 三项均失败 |
| NG-50 | 六、十 | 本地TDD、真实checkpoint smoke、独立P0/P1、Git、预注册报告 | tests/report/git | PASS | 146聚焦+51相邻回归；终态复审`APPROVE/P0=0/P1=0` | 历史v3审查P0结论被终态审计纠正 |
| NG-51 | 十一 | 完整同排/分组/资源/混淆报告与唯一裁决 | analyzer/report | PASS | `analysis_strict_spec_v4`+v3最终报告 | 四态边界明确，唯一裁决`REJECT_ROUTE` |

终态实现与分析纠偏已提交为`8ec37964`；NewGuard聚焦`146 passed`、相邻E0OCF/FloorBoost回归`51 passed`，独立终审`APPROVE / P0=0 / P1=0`。v4只读复算的`summary.json` SHA256为`bf8c963379089d0d5090ac06890386f33d4182e3bbd0547c4f99de205cc83552`，`gates.json` SHA256为`c6861b6792bc9b59d715757e2b19b204a45fb374456b056efe2dc04461d066e0`。

## 可行性摘要（冻结前，12行）

1. 新方法不再正向抬升旧类整体bias；共享截距只允许`tau<=0`，方向上直接抑制new→old。
2. K5/new20的新类增广support最多100行，289维零空间至少189维。
3. K10/new20最多200行，零空间至少89维；new5/new10的零空间更大。
4. 零空间用紧凑行空间算子和确定性机器精度阈值计算，不显式构造289×289投影矩阵。
5. 每个旧类从E0基线固定的bottom-20%support构造同公式投影方向。
6. 6个旧类方向先作组内零和，再由小型线性max-min确定6个强度和共享`tau`。
7. 约束同时要求旧类tail对所有竞争类的margin与新类new-vs-old margin共同提高。
8. 类置换只同步置换方向、变量和约束，算法形式不变。
9. 新类权重行不变；新类support上的内部旧类logit变化为零，只有统一非正`tau`。
10. 求解完成后同时核对FP32中间头和正式int8/FP16部署头；正式保护失败即回退E0。
11. 额外工作只发生在support端，主LDA仍仅一次FULL fit。
12. 数值、约束或闭包失败时整头逐字节回退E0；K1/K2继续D92 FULL exact alias。
