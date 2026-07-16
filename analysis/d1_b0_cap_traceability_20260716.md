# D1-B0-Cap实现追踪

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| D1B0-01 | 用户目标；项目.md §10.3.1 | 新增独立D1-B0-Cap候选，不改变现有D1/D2默认行为 | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | `tests/test_stage2_diag_cosine_exploration.py`；9项focused pytest通过 | 默认候选仍为现有D1 |
| D1B0-02 | 用户目标 | D1-B0-Cap删除class bias，参数量和持久状态不计入class bias | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | 参数量、空bias state和receipt测试通过 | 保留D1的可训练class weights |
| D1B0-03 | 用户目标 | FFT96对应`log_scale`限制为`±ln(1.5)`，z_id160/RF32沿用`±1.5` | `code/cvsrffi/stage2_diag_cosine_exploration.py` | verified | 逐块clamp边界与训练后state边界测试通过 | 训练与预测使用同一逐维边界 |
| D1B0-04 | 项目.md §7.4、§10.3.1 | 保持LEO_weak-only、support-only、无query拟合/role/quota/global assignment，并在receipt记录机制字段 | `code/cvsrffi/stage2_diag_cosine_exploration.py`、`tests/test_stage2_diag_cosine_exploration.py` | verified | receipt协议字段及`query_rows_used_for_fit=0`测试通过 | 未新增准入artifact |
| D1B0-05 | 用户目标 | runner CLI和row pipeline支持候选选择，pipeline默认兼容旧D1 | `code/scripts/run_cvs_stage2_diag_cosine_exploration.py`、`code/scripts/run_cvs_somph_diag_row_pipeline.py`、`tests/test_run_cvs_somph_diag_row_pipeline.py` | verified | standalone parser、pipeline parser、predictor/scorer调用透传测试通过 | scorer尚未从prediction COMMIT反向验证candidate，列为后续证据增强 |
| D1B0-06 | 项目.md §10.3.1 | 固定20epoch、auxiliary weight 4、逐样本全注册类推理和资源上限 | implementation/tests | verified | receipt与resource断言；9项focused pytest通过 | 当前MAC字段仅为head/fit估算，端到端MAC与singleton延迟仍需实测 |

## Verification

- `conda run -n ssr-gpu python -m py_compile code/cvsrffi/stage2_diag_cosine_exploration.py code/scripts/run_cvs_stage2_diag_cosine_exploration.py code/scripts/run_cvs_somph_diag_row_pipeline.py tests/test_stage2_diag_cosine_exploration.py tests/test_run_cvs_somph_diag_row_pipeline.py`
- `conda run -n ssr-gpu python -m pytest -q tests/test_stage2_diag_cosine_exploration.py tests/test_stage2_diag_cosine_scorer.py tests/test_run_cvs_somph_diag_row_pipeline.py`→`9 passed`
- `git diff --check -- ...`→PASS，仅有仓库既有LF/CRLF提示。

结论：6项机制要求全部verified；实现为严格D1-B0-Cap机制落地，不是近似替代。真实N607 row、端到端MAC/singleton延迟，以及scorer对candidate/COMMIT的密码学反向绑定仍由主任务继续完成，不能用当前静态验证替代。
