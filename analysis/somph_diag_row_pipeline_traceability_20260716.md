# SOMP-H+D1真实LEO_weak开发row流水线追踪

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| P01 | `项目.md`§7.1、§10.3.1 | 只从authority绑定的已叠加`leo_*_weak`cache构建row，Phase2不接触clean或source附加工件 | `code/scripts/run_cvs_somph_diag_row_pipeline.py` | verified | CLI固定调用`build_somph_offline_row_pair`；聚焦测试通过 | 复用现有formal offline builder，不新增准入链 |
| P02 | `项目.md`§7.2、§10.3.1 | predictor请求不得包含truth、角色、类别配额或query标签，逐样本对全部注册类决策 | `code/scripts/run_cvs_somph_diag_row_pipeline.py`、`tests/test_run_cvs_somph_diag_row_pipeline.py` | verified | exact request mock断言；SOMP-H runtime request回归通过 | 请求只含现有exact runtime contract字段 |
| P03 | 用户子任务 | CLI参数化cache manifest、authority bundle/commit、Phase1 checkpoint、sealed runtime、method lock、output root、receiver、seed、K、new_count、device | `code/scripts/run_cvs_somph_diag_row_pipeline.py` | verified | `--help`成功；parser测试通过 | formal query-per-TX固定为20 |
| P04 | 用户子任务 | 先构建同row before/after，再只运行SOMP-H enrollment产生合法head并finalize apply包 | `code/scripts/run_cvs_somph_diag_row_pipeline.py` | verified | mock调用序列为build→两组enroll/finalize→D1 | before=`stage2b`，after=`stage2c` |
| P05 | 用户子任务、`项目.md`§7.2 | 两份D1不可变prediction均落盘后，独立diag scorer才可读取truth | `code/scripts/run_cvs_somph_diag_row_pipeline.py`、`tests/test_run_cvs_somph_diag_row_pipeline.py` | verified | scorer前检查两份预测存在且只读；truth只传最终scorer | predictor request及D1调用均无truth参数 |
| P06 | 用户子任务 | 全链路无覆盖写 | `code/scripts/run_cvs_somph_diag_row_pipeline.py`、`tests/test_run_cvs_somph_diag_row_pipeline.py` | verified | 预存在output root时在builder调用前抛`FileExistsError`；`git diff --check`通过 | 复用下游exclusive-create语义 |

## 验证记录

- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code/scripts/run_cvs_somph_diag_row_pipeline.py tests/test_run_cvs_somph_diag_row_pipeline.py`
- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code/scripts/run_cvs_somph_diag_row_pipeline.py --help`
- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q tests/test_run_cvs_somph_diag_row_pipeline.py`
- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q tests/test_run_cvs_somph_diag_row_pipeline.py tests/test_stage2_diag_cosine_exploration.py tests/test_stage2_diag_cosine_scorer.py tests/test_somph_runtime_request.py`
- `git diff --check -- code/scripts/run_cvs_somph_diag_row_pipeline.py tests/test_run_cvs_somph_diag_row_pipeline.py analysis/somph_diag_row_pipeline_traceability_20260716.md`

反向审计：6项均为`verified`；无deferred、rejected或blocked项。当前实现为现有formal offline builder、SOMP-H enrollment/finalizer与D1/scorer接口的严格薄编排，不是对数据协议或算法的近似重写。
