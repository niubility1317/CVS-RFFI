# D37 B3-preserving residual-int8追踪表

## 范围与权威

- 科学/数据协议：`E:\type10-7\项目.md`与`p2_min_v1`。
- 研发目标：`E:\codex\home\attachments\128c9845-1eda-42ba-a430-dd11cb2a34a0\goal-objective.md`。
- 活动设计与三轮复盘：`automation_reports/CV-SincNet/d37_b3_preserving_int8_20260718/report.md`。
- 说明：本追踪表在D37 core首批实现后补建；这是流程时序偏差，不影响代码证据，但在反向审计中保留为已知限制。任何`verified`只表示对应实现/测试闭合，不表示support-held性能或正式query性能达标。

## 条款追踪

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D37-01|目标§2、§3|只读取固定单次LEO弱观测形成的合法support；不访问clean/source/query|`stage2_d37_b3_preserving_int8.py`、runner|implemented|core resource audit字段；runner集成测试待完成|数据复用现有D18 `VALIDATED_ONCE` cell，不重验|
|D37-02|目标§3|query不参与fit/选择；每个样本独立面对全部注册类|`base_score_d37_b3_preserving_int8`、`score_d37_b3_preserving_int8`|verified|`test_row_scoring_is_batch_split_and_order_invariant`、`test_scoring_never_fits_or_opens_query_state`|score API无labels/role/batch quota参数|
|D37-03|复盘§2.2、设计§3|旧类直接来自最终B3权重，不重建另一套旧头|`fit_d37_b3_preserving_int8`|verified|`test_direct_b3_compile_preserves_old_decisions_and_byte_prefix`|共享`log_diag`为FP32算子，不是FP32 target prototype|
|D37-04|目标§2.6、设计§3|target-old/new实际预测身份均为正式int8且不存FP32 target prototype|D37 state、residual compiler|verified|dtype/readonly/error单测；geometry audit三项布尔门|采用两级残差int8＋每块FP16 scale|
|D37-05|目标§7.8、设计§3|注册时旧类量化状态append-only，前缀逐bit不变|`old_prefix_bitwise_unchanged_d37`|verified|prefix单测覆盖未校准与校准后state|必要条件，不冒充outer held安全|
|D37-06|目标§5|全部类使用相同公式，标签置换同构，无class ID专属规则|D37 compiler/scorer、runner selection|implemented|geometry audit；runner置换集成测试待完成|固定块和margin与class ID无关|
|D37-07|设计§3|以support-OOF硬区间联合约束旧类零侵入与新类真实类胜出|`fit_oof_feasible_offset_d37`|verified|三个margin可行例与空区间fail-closed单测|`L<=U`才返回公共offset|
|D37-08|设计§3|A/B/C只改变固定margin`0/0.05/0.10`|`D37B3PreservingInt8Config`、runner candidate registry|implemented|core参数化单测；runner集成待完成|避免小参数盲扫和多机制混杂|
|D37-09|目标§6、设计§4|开发矩阵为7候选×3场景×5fold=105行，outer held不进入fit|runner、integration test|verified|真实rank-pair integration执行；D37候选锁与105行断言|候选Z0/C0/B3/D33-FAST+D37-A/B/C|
|D37-10|目标§5、设计§4|同row报告before-old、after-old、new、H、forgetting、全部逐类floor/混淆/侵入/不可达|runner artifacts、report|pending|待105行输出及完整日志解析|不得使用边际max/min拼接|
|D37-11|设计§4|晋级门含量化旧头逐scene/fold/class不弱于B3、全部旧行outer侵入0、所有新类physical LOSO正margin、matched identity/B3/D33 joint/floor非劣|runner selector|verified|selection反例、全旧行侵入计数、full-K10门测试|prefix不变只作为必要门|
|D37-12|目标§9|<=80k参数、<=30epoch、<=50step、<=256KB、无dense query graph/批优化|core resource audit、runner full audit|verified|`test_resource_caps_hold_for_registered_scale`覆盖new2/5/10/20|core当前0参数/epoch/step；最大注册规模低于状态上限|
|D37-13|目标§11|完整training trace、selection/resource/geometry/support audit、receipt与stdout|runner|pending|待真实support screen|完整解析要求105/105且哈希闭合|
|D37-14|AGENTS Version Management|本地修改在Git承载面、窄验证、diff审计、提交|repo/report|implemented|当前git diff/status待最终复核|根目录报告为镜像，不是独立Git仓库|
|D37-15|目标§10、AGENTS N607|本地验证后才同步N607；短连接、preflight、GPU/进程审计|launcher/report|deferred|当前未触碰N607|已知N607环境组合不兼容D36闭环；先完成本地同cell算法屏|
|D37-16|目标§13|只有完整独立确认矩阵全门达标才能完成goal|confirmation artifacts/report|deferred|development support screen尚未完成|D37窄验证绝不等于目标完成|

## 遗漏陷阱反向审计

1. `log_diag_fp32`是所有类共享的B3特征算子；target identity权重只存在`code1_qint8/code2_qint8`与FP16 scale，不将其误写为“全状态int8”。
2. OOF区间可行只证明inner support-held约束可同时满足；outer held侵入/可达性仍须独立实测。
3. runner中必须捕获单候选空区间并记录`fail_closed`，不能使其余candidate/fold丢行，也不能自动放宽margin。
4. 量化旧头与FP32 B3的总体相同不够；晋级必须逐类非劣。
5. local 105行完成不等于N607验证、query结果或正式确认矩阵。

## 当前验证记录

```text
conda run -n ssr-gpu python -m pytest tests\test_stage2_d37_b3_preserving_int8.py -q
12 passed
```

当前计数：`verified=9`、`implemented=4`、`pending=2`、`deferred=2`、`rejected=0`、`blocked=0`。最高风险是D37公共offset的OOF可行区间在真实D18 cell上为空，或虽可行但outer held仍出现侵入/不可达。另有明确K1方法缺口：单物理样本不能同时作为prototype输入与自身OOF held行；D37尚无合法预锁定K1校准规则。
