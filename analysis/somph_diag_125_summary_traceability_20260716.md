# SOMP-H D1 125稳定性聚合追踪

日期：2026-07-16

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| S125-01 | `项目.md`10.3.1 | 读取不可变125-job manifest及每个job的pipeline/score绑定，不从stdout拼指标 | `code/scripts/summarize_cvs_somph_diag_125_stability.py` | verified | 17项focused/adjacent pytest PASS | 先验证125个Cartesian键和job ID恰好一次，再读取job，重复evidence键fail closed |
| S125-02 | `项目.md`9.2、9.3、10.3.1 | 同row输出Stage2-B/C old、old floor、seen-new、H、forgetting/gain及逐TX结果 | 同上 | verified | actual scorer-format fixture及漏query失败测试PASS | 每state×scenario要求prediction token与密封query package exact equality；before/after old query一致 |
| S125-03 | `项目.md`10.3.1 | K10绝对门槛、K5相对K10不降超过3pp、K1总体与逐receiver增益非负 | 同上 | verified | performance PASS/FAIL双向测试PASS | 技术聚合成功不等于性能PASS；direct缺失时仍显式保留已运行门禁FAIL |
| S125-04 | `项目.md`10.3.1 | K1复用K10 rank0 support及完全相同query | 同上 | verified | post-channel IQ SHA重算、缺类/truth-gap失败测试PASS | support/query全数组等长非空且token唯一；after-new20每场景必须恰有26个rank0 |
| S125-05 | `项目.md`10.3.1 | strict direct ADV3B02 K1差值不得伪造 | 同上 | verified | gates JSON assertion PASS | 当前D1产物无direct stream，固定写`MISSING_NOT_RUN` |
| S125-06 | `AGENTS.md`Experiment Reporting | 输出summary、row/scenario/receiver/per-TX表和gates，严格no-overwrite | 同上；测试 | verified | 六个输出存在且重复运行拒绝覆盖 | 聚合器只读实验输入 |

最高风险项：当前D1 diag pipeline未保存strict direct ADV3B02预测流，因此本聚合器只能显式报告该正式K1门槛缺失，不能判为通过。

验证命令：

`python -m pytest -q tests/test_summarize_cvs_somph_diag_125_stability.py tests/test_run_cvs_somph_diag_125_stability.py tests/test_run_cvs_somph_diag_row_pipeline.py tests/test_stage2_diag_cosine_scorer.py`

结果：`17 passed`。实现对本125稳定性tranche是严格字段/绑定落地；对完整正式目标仍是不完整证据，因为direct ADV3B02 K1配对流和其95%置信区间尚未运行。
