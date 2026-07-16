# SOMP-H D1 125稳定性launcher追踪表

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| S125-01 | 用户目标；`项目.md`§10.3.1 | 固定5个target receiver×5个独立稳定性seed×5个切片，共125个Stage2-B/C pair job，不含开发seed713101 | `code/scripts/run_cvs_somph_diag_125_stability.py`；`tests/test_run_cvs_somph_diag_125_stability.py` | verified | 精确计数、receiver/seed/slice覆盖、无713101、25个K1 job测试通过；manifest区分375个scenario pair与750个注册前/后scenario-state指标 | 当前authority为false，因此claim scope固定为development-only，不冒充正式确认；切片固定为K10-new5/10/20、K5-new20、K1-new20 |
| S125-02 | `项目.md`§7.4、§10.3.1 | 复用既有row pipeline直接实验，不新增准入链；按receiver/seed定位cache、authority bundle和COMMIT SHA | 同上 | verified | manifest路径与COMMIT文件SHA断言通过 | COMMIT SHA为`COMMIT.json`文件SHA256 |
| S125-03 | `项目.md`§7.1、§7.2 | 每个job明确LEO_weak-only、无clean、无query truth/role/quota/global assignment，且含Stage2-B/C成对结果 | 同上 | verified | manifest完整contract与before/after stage断言通过 | 声明不替代row pipeline已有密封验证 |
| S125-04 | 用户子任务 | 候选锁定`d1_historical_diag_fftrf`，逐job调用`run_cvs_somph_diag_row_pipeline.py`且不可覆盖 | 同上 | verified | command构造断言；已有manifest差异、job输出或日志存在均fail closed | 不暴露candidate调参CLI |
| S125-05 | 用户子任务 | 固定8分片、device、完整events/summary；技术失败记录后继续，`--fail-fast`可选 | 同上 | verified | manifest预先绑定每个job的shard，8分片无重无漏；非8分片和空分片fail closed；模拟失败事件后继续并生成partial summary | 每个job独立stdout/stderr日志 |
| S125-06 | `项目.md`§10.3.1 | K1/K5 support嵌套由现有builder保证，不重新实现support选择 | 同上 | verified | manifest nesting policy与K1覆盖断言通过 | K1/K5只作为锁定candidate压力评估 |

## 验证记录

- `conda run -n ssr-gpu python -m py_compile code/scripts/run_cvs_somph_diag_125_stability.py tests/test_run_cvs_somph_diag_125_stability.py`
- `conda run -n ssr-gpu python -m pytest -q tests/test_run_cvs_somph_diag_125_stability.py tests/test_run_cvs_somph_diag_row_pipeline.py`→`8 passed`
- `conda run -n ssr-gpu python code/scripts/run_cvs_somph_diag_125_stability.py --help`

反向审计结论：6项launcher要求全部verified。实现是固定tranche的严格薄launcher，不改变`项目.md`协议，也不复制或扩建准入逻辑。该tranche只含K5/new20和K1/new20压力切片，尚未聚合K1逐receiver增益或直接ADV3B02差值，因此明确属于不完整的development-only独立稳定性证据，不是正式目标完成矩阵；真实N607同步、manifest生成与125个实验执行由主任务继续完成。
