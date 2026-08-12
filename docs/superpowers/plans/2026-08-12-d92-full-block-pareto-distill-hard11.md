# E0 FULL/BLOCK Pareto蒸馏Hard11实施计划

> 设计：`docs/superpowers/specs/2026-08-12-d92-full-block-pareto-distill-design.md`
>
> 追溯：`analysis/d92_full_block_pareto_distill_traceability_20260812.md`

## Task 1：共享统计与科学核心

Owner：Terra科学实现。只拥有：

- `code/cvsrffi/stage2_d92_registration_balanced_covariance.py`
- 新增`code/cvsrffi/stage2_d92_full_block_pareto_distill.py`
- `code/scripts/probe_d92_registration_balanced_covariance.py`
- `code/cvsrffi/stage2_d92_e0d_slim.py`
- `code/cvsrffi/stage2_d92_e0d_query_evaluation.py`
- 对应科学测试文件。

步骤：先写RED，覆盖共享协方差只算一次、FULL/BLOCK各一次、词典序目标、并列tail、old/new组内置换、D42解码重检、K1/K2 alias、query零访问、E0 exact fallback和资源收据；再写最小实现并运行`ssr-gpu`聚焦回归。不得修改Hard11 runner/analyzer/report，不得使用query或调整冻结门。

## Task 2：Hard10机械闭环

Owner：Luna机械实现。只新增Pareto Distill专属config、Hard11 builder、runner、analyzer、CLI与对应测试；复用NewGuard已验证的10个performance outer、1个K1 liveness、E0 raw/per-old基线、8shard、truth-last scorer和共享distinct-outer技术停派。不得修改科学公式、阈值、矩阵或历史artifact。

analyzer必须输出八项同排均值、10行逐outer、3scene、receiver、K/new、6旧类、fallback和资源；严格三分支为`REJECT_ROUTE`、`REVISE_ONCE`、`ADVANCE_TO_TARGET125_CANDIDATE`。

## Task 3：集成、本地门与版本

Primary集成两项实现，补预注册报告和Git追溯。最小发布门：聚焦协议负测、真实checkpoint K>2 no-query smoke、K1 alias smoke、共享统计资源receipt、独立P0=0/P1=0、clean Git commit、不可覆盖run ID。P2只记录。

若共享路径本地实测已经超过wall硬门或部署头无效，则在本地`REJECT_ROUTE`，不浪费N607性能矩阵。

## Task 4：唯一N607 Runner

冻结run ID、commit、archive/config/launch SHA、exact command、普通N607账号、8卡映射、输出/log路径、expected counts和health-only stop后，交给唯一Luna runner。Runner完成preflight、同步、真实K>2 smoke、一次detached launch、短连接监控、11/11闭合、完整取回和结构化handoff；不读性能、不调方法、不重启同run ID。

## Task 5：truth-side分析与决策

Primary在prediction immutable后一次性连接truth，运行冻结analyzer并更新同一报告。任一核心指标未严格优于E0立即拒绝并更换方法族；只有全门通过才建议后续Target125，绝不自动启动。

