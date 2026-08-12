# D92 NewGuard Hard11设计追溯

设计源：`E:\codex\home\attachments\a32ff2e7-5e54-4d07-9697-60c470abe165\pasted-text-1.txt`

目标候选：`E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN`

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| NG-01 | 一 | H、old BA、c_old_acc、floor、seen-new全升，forgetting、双向混淆全降 | analyzer/report | pending | Hard10同排分析 | 不允许加权抵消 |
| NG-02 | 一、九 | query MAC和永久state不增，注册计算显著低于D92 | core/query/analyzer | pending | receipt+resource audit | FULL主fit=1 |
| NG-03 | 二 | 不再扫描FloorBoost标量，不用旧类整体加分作主机制 | core/method lock | pending | 静态身份测试 | FloorBoost仅作负证据 |
| NG-10 | 三.1 | 复用E0_FULL_ONLY的288维、task-balanced covariance和单FULL头 | probe/slim | pending | fit inventory测试 | 禁用BLOCK/OCF/LOO/Fisher/Pareto |
| NG-11 | 三.2 | 增广新类support零空间内构造旧类内部残差 | probe | pending | `X_new @ delta`测试 | 数值秩阈值为机器精度规则 |
| NG-12 | 三.2 | 内部残差旧类组零和，新类行byte-exact | probe | pending | 置换/字节测试 | 只区分old/new集合，不分具体TX |
| NG-13 | 三.3 | 每类基线最弱20%support固定后执行确定性max-min | probe | pending | tail冻结和重排测试 | `method=lower`，无query选择 |
| NG-14 | 三.4 | 共享`tau<=0`降低旧类包络，联合优化双向margin | probe | pending | support margin测试 | 新类support内部残差为零 |
| NG-15 | 三.4、四 | 不可行/秩/有限值失败时精确回退E0 | probe/slim/query | pending | 故障注入测试 | 回退不能算性能成功 |
| NG-16 | 四 | K<=2保持D92 FULL精确alias | probe/slim/query | pending | K1/K2回归 | liveness不入性能均值 |
| NG-20 | 五 | 输出完整NewGuard数学、资源、fallback和零访问receipt | probe/slim/query | pending | receipt漂移负测 | 只增加直接必需字段 |
| NG-21 | 五、六 | query truth/fit/update/selection/role/quota/global全部false | query/runner | pending | 协议负测+真实smoke | scorer后连接truth |
| NG-30 | 七 | 冻结原Hard10+1个K1、3scene、1arm、8shard | config/hard11/runner | pending | manifest测试 | 11job、33scene-arm |
| NG-31 | 七 | 复用历史paired_rows且SHA=`6ebb37...de6a` | config/analyzer | pending | 本地hash+analyzer测试 | 不重跑D92/E0 |
| NG-32 | 七、十 | 唯一runner、smoke先于shard、共享技术停派、不可覆盖 | runner/launch/report | pending | runner测试+N607 handoff | 不按性能停止 |
| NG-40 | 八 | 八项均值严格Pareto，任一反向即REJECT | analyzer/config | pending | gate分支测试 | tie不算成功 |
| NG-41 | 八.2 | 达成规定幅度才ADVANCE；全正但幅度不足仅REVISE_ONCE | analyzer/config | pending | 三分支测试 | 不自动跑125 |
| NG-42 | 八.3 | outer、slice、旧类、receiver、scene稳定性门 | analyzer | pending | 分组fixture | 不由单一切片承担收益 |
| NG-43 | 九 | median wall<=1.5x E0、p90<=150ms、peak<=E0+512KiB | analyzer/config | pending | resource gate测试 | 解析MAC不冒充时延 |
| NG-50 | 六、十 | 本地TDD、真实checkpoint smoke、独立P0/P1、Git、预注册报告 | tests/report/git | pending | command/commit/review | P2不阻塞 |
| NG-51 | 十一 | 完整同排/分组/资源/混淆报告与唯一裁决 | analyzer/report | pending | artifact reverse audit | 使用四态命名 |

## 可行性摘要（冻结前，12行）

1. 新方法不再正向抬升旧类整体bias；共享截距只允许`tau<=0`，方向上直接抑制new→old。
2. K5/new20的新类增广support最多100行，289维零空间至少189维。
3. K10/new20最多200行，零空间至少89维；new5/new10的零空间更大。
4. 零空间用确定性SVD机器精度阈值计算，不按outer或query调秩。
5. 每个旧类从E0基线固定的bottom-20%support构造同公式投影方向。
6. 6个旧类方向先作组内零和，再由小型线性max-min确定6个强度和共享`tau`。
7. 约束同时要求旧类tail对所有竞争类的margin与新类new-vs-old margin共同提高。
8. 类置换只同步置换方向、变量和约束，算法形式不变。
9. 新类权重行不变；新类support上的内部旧类logit变化为零，只有统一非正`tau`。
10. 求解完成后仅发布一个FP32仿射头，query MAC和state布局与E0完全相同。
11. 额外工作只发生在support端，主LDA仍仅一次FULL fit。
12. 数值、约束或闭包失败时整头逐字节回退E0；K1/K2继续D92 FULL exact alias。
