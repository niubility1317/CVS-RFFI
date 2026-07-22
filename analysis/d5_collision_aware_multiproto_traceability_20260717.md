# D5碰撞感知压缩多原型head Traceability

日期：2026-07-17

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D5-01 | 用户目标；`项目.md`7.1.1 | 每个scenario独立使用固定LEO_weak接收IQ对应的物理support，不跨scenario拼接，不新增LEO状态或K | `code/cvsrffi/stage2_collision_aware_multiproto.py`;开发脚本 | verified | 三场景独立receipt；package token互斥审计PASS | head只接收已提取support feature及scenario原子调用 |
| D5-02 | 用户目标 | 每类最多3个prototype，使用support-only class Gram、物理sample LOO混淆、紧致度和局部1NN结构 | 同上 | verified | 6项聚焦pytest；真实receipt逐类candidate diagnostics | 不使用计算view冒充物理LOO |
| D5-03 | 用户目标 | 由support-only稳定性自动选择1/2/3 prototype并保守回退，避免LOO过拟合 | 同上；`tests/test_stage2_collision_aware_multiproto.py` | verified | 合成多模态/紧致类回归；真实分布clear 8/3/0、low 9/1/1、rain 10/0/1 | 增加复杂度需达到预登记最小稳定性增益 |
| D5-04 | 用户目标；`项目.md`7.2 | 所有query逐样本对全部注册类打分，不访问query标签、角色、类别数量、配额、顺序或query图 | 同上 | verified | batch组合不变；score API仅`query_features/head`；receipt oracle字段false | query truth仅由独立scorer后连接 |
| D5-05 | 用户目标 | after逐位锁定before旧prototype；仅对新增注册prototype执行support-only有界去混淆 | 同上 | verified | 旧scale/prototype/mask/centroid/penalty逐位测试；真实support floor evidence | “旧/新”只来自注册生命周期，不来自query role Oracle |
| D5-06 | 用户目标；`项目.md`10.3.1 | 参数/状态/MAC审计；0epoch闭式适配；状态<=256KB、参数<=80k、无dense query图 | 同上 | verified | 0参数/0epoch；25,988bytes；7,846–8,136 MAC；dense graph 0 | FP16部署状态精确计数 |
| D5-07 | 用户目标 | 在合法`rx20-1/seed713101/K10/new5`开发row运行before/after并报告逐scenario、逐类和遗忘 | 开发报告目录 | verified | 既有margin扫描统一降级为`POST_SCORE_DIAGNOSTIC_PARETO_POINT` | 不得由已评分query选择0.008或任何margin |
| D5-08 | 交付纪律 | 聚焦测试、diff审计且不提交Git | 本记录 | verified | `git diff --check`；目标文件保持untracked | 用户明确要求不提交 |
| D6C-01 | 项目.md 7.2、10.3.1 | margin只能由support-only旧类非侵入、逐类floor下界、新类物理leave-one/two-out和复杂度选择 | `code/cvsrffi/stage2_collision_aware_multiproto.py` | verified | `select_support_only_margin*`；7项pytest | selector无query参数 |
| D6C-02 | 项目.md 7.2 | 三场景联合锁定唯一margin，打开query前写不可变COMMIT | D6c运行artifact | verified | margin=0；COMMIT 01:54:45；query flags false | 五个候选统一support-only排序 |
| D6C-03 | 用户纠正 | 预测COMMIT后才允许独立scorer打开truth | `run_d6c_fresh_holdout.py` | verified | prediction 01:55:51；score 01:55:56 | prediction artifact只含token/scenario/prediction |
| D6C-04 | 并行holdout冲突 | 同一rank10–19 holdout已被D6a先评分，不得称fresh/unseen或独立验证 | 开发报告 | verified | D6a score COMMIT 01:51:51 | D6c降级为`CONCURRENTLY_CONSUMED_HOLDOUT_METHOD_COMPARISON` |

## Verification

```text
conda run -n ssr-gpu python -m py_compile code/cvsrffi/stage2_collision_aware_multiproto.py
conda run -n ssr-gpu python -m pytest tests/test_stage2_collision_aware_multiproto.py -q
7 passed
git diff --check -- code/cvsrffi/stage2_collision_aware_multiproto.py tests/test_stage2_collision_aware_multiproto.py analysis/d5_collision_aware_multiproto_traceability_20260717.md
```

Reverse audit：12项均为`verified`，无`deferred`、`rejected`或`blocked`。既有margin扫描仅为post-score diagnostic；D6c margin=0是纯support-only锁定，但其holdout已被并行D6a提前消费，只能作为非独立同holdout方法对照。最高风险仍是严重旧类遗忘与floor不足。
