# D12联合注册残差logit head追踪

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D12-01 | 项目.md§7.1/7.1.1 | 只读取sealed单一LEO_weak received-IQ，内部逐物理sample提取base feature；无clean/source/query | D12 module、runner、audit | verified | v3 pre-open与COMMIT | 三场景物理ID互斥；actual received-IQ SHA逐sample核验；不新增LEO状态 |
| D12-02 | 项目.md exact-K | Before旧6类与After旧6+新5类都只读取严格K10 enrollment-only包 | D12 runner/tests | verified | v3 package root/seal与13项测试 | 每类恰好K10；K5传K10行反例fail closed |
| D12-03 | 主报告D12 | `cosine(q,p)+alpha*W2*activation(W1q)`，rank8、base cosine保留、小固定alpha | D12 module/tests | verified-negative | alpha∈{0,.05,.10,.15}真实比较 | d=288、C=11正alpha参数2392；全部正alpha未过门，最终alpha0回退 |
| D12-04 | 主报告D12 | Before旧类拟合；After old+new联合拟合并使用old logit distillation与identity正则 | D12 trainer | verified-negative | v3完整loss trace | 正alpha联合head能局部改善H，但不能阻止旧类遗忘 |
| D12-05 | 主报告D12/D11反例 | 每fold old/new各held2，输出old/new/joint/H/逐类floor/forgetting | D12 joint L2O/tests | verified | 5fold×3场景×4候选 | K8 fold由独立support selection SHA绑定；Before/After旧lineage按label×rank精确复用 |
| D12-06 | 主任务 | 三场景使用统一超参数，逐类old非退化且new同口径门不过则不开放query | D12 runner/audit | verified-negative | 三场景gate均false | 同时要求old逐类不低于Before/base、forgetting≤0、new不低于D11-v6 joint-new且H高于base；D10标NOT_COMPARABLE |
| D12-07 | 项目.md§7.2 | 每个query逐sample面对全部注册类；无role/quota/query fit/dense graph | D12 formal prediction API/tests | verified-no-query | 单行全注册类API与v3 audit | query/truth/prediction/score/scorer均未打开 |
| D12-08 | D11独立审计修复要求 | formal API只接受内部runtime-authorized artifact；拒绝普通Mapping/array；固定runner内部callback路径 | D12 module/tests/runner | verified | Mapping、IQ SHA、runtime、operator反例 | CLI不暴露callback；module+runner+runtime/checkpoint SHA由verified package入口闭合 |
| D12-09 | 资源硬门 | 参数≤12k、epoch≤15、state readonly+content SHA、状态≤256KiB | D12 state/tests/audit | verified | v3与反例测试 | content SHA覆盖完整hyper/resource/operator/view/selection；NPZ/metadata写后回读 |
| D12-10 | 实验要求 | 使用D8b真实三场景strict K10包跑support-only；不打开query | D12 runner/artifact/report | verified-not-selected | v3 COMMIT `2b1f0f33ff189972c38fc1942ad83593365b0c5dce341da9118f8a43b72456d8` | `SUPPORT_ONLY_D12_NOT_SELECTED_NO_QUERY_OPEN` |
| D12-11 | AGENTS.md/Git | 独立新文件、本地ssr-gpu验证、检查diff；不stage/commit | tests/trace/handoff | verified-uncommitted | 13/13 pytest；py_compile；diff-check | 共享脏树中仅新增4个D12文件，交主agent选择性提交 |

## 预登记候选

首轮候选统一固定`rank=8`、`activation=tanh`、`epochs<=15`。开发选择只使用三场景K10 support的联合leave-two-out证据，排序优先级为：全部场景旧类逐类非退化门、最差场景new floor、平均`H_old_new`、平均joint accuracy。若全部候选均未过门，则输出`SUPPORT_ONLY_D12_NOT_SELECTED_NO_QUERY_OPEN`并停止，不读取任何query或scorer。

## v1/v2/v3证据边界

- `d12_joint_residual_logit_head_v1`是早期NO-GO，尚未含alpha0真实回退、完整state hash与最终gate；仅保留旧代码哈希负结果，不追认。
- `d12_joint_residual_logit_head_v2_hardened`绑定module SHA`a24c0a0467f22f3798561720b44d8c3c77006ce6b861b4dabba5b032d6e67814`与runner SHA`feba43c4554e82e6590b95bde98df9ac8aa38232d7946e4e4909dfb5fdd822a1`。统一正alpha候选`alpha=.05`三场景均失败：旧类遗忘`6.67/5.00/11.67pp`，new overall`0.50/0.46/0.68`，不得打开query。
- `d12_joint_residual_logit_head_v3_alpha0_fallback`是最终support-only证据，module SHA仍为`a24c0a0467f22f3798561720b44d8c3c77006ce6b861b4dabba5b032d6e67814`，runner SHA为`eb7596bb1934232970b1fef9739121e06c7a2309ee1996c15af2b2f334caaf4b`。全部正alpha失败后真实选择`alpha=0`，明确是安全回退而非D12性能改进。

## v3最终结果

|场景|Before old overall|After old overall/floor|seen-new overall/floor|H|old forgetting|参数/epoch/state|
|---|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|0.7167|0.6333/0.1000|0.5000/0.2000|0.5588|0.0833|0/0/12672B|
|`leo_low_elev_weak`|0.7333|0.6833/0.2000|0.4600/0.1000|0.5499|0.0500|0/0/12672B|
|`leo_rain_weak`|0.7333|0.6167/0.3000|0.6200/0.4000|0.6183|0.1167|0/0/12672B|

alpha0最终状态相对identity-only单qKNN的head侧MAC与持久状态均下降90%（3168 vs 31680 MAC；12672B vs 126720B），但Python安全校验路径的support-row微基准延迟高约41—47倍；该微基准不是正式query延迟。run总wall time25.04s，CUDA峰值allocated/reserved为58,253,312/96,468,992B，Python tracemalloc峰值54,478,050B。

`training_log.jsonl`共1176行，覆盖`joint_l2o_before`555行、`joint_l2o_after`555行、fold summary60行、最终Before/After fit各3行；epoch范围0—12，数值全finite，训练行均含base/residual logit范数与最大logit修正。K1/K5未从K10切片，等待独立strict package。

## 最终追踪计数

- verified或verified-negative：11
- deferred：0（K1/K5属于本次D12 K10任务范围外的后续独立包验证）
- rejected：0
- blocked：0

最高风险剩余项：D12正alpha联合head在三场景均无法满足注册后旧类非遗忘，尤其rain仍有11.67pp遗忘；因此当前路线不能开放query，也不能进入125确认矩阵。
