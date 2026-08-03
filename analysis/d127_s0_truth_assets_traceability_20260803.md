# D127 S0独立truth/formal-D92资产可追溯表

| ID | 来源要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|
| T1 | 预测闭合后才可读取truth，且truth不回流预测器 | `code/cvsrffi/stage2_d127_s0_truth_assets.py` | verified | `test_stage2_d127_s0_truth_assets.py` | 先校验paired prediction、plan、lock与已写truth-open event。 |
| T2 | 只从D92 scorer-side truth sidecar读取真实标签与role | `code/cvsrffi/stage2_d127_s0_truth_assets.py` | verified | `test_stage2_d127_s0_truth_assets.py` | 仅以opaque query ID连接`true_class_handle/evaluation_role`；不读predictor包或预测值。 |
| T3 | 18个D127行逐行绑定D92 retry2同receiver/K/scene/job，并记录来源hash | `code/cvsrffi/stage2_d127_s0_truth_assets.py` | verified | `test_stage2_d127_s0_truth_assets.py` | K5强制对应K10 source job；记录6个job的pipeline/row/pair/truth/score哈希。 |
| T4 | 输出匹配现行paired scorer契约，文件exclusive | `code/cvsrffi/stage2_d127_s0_truth_assets.py` | verified | `test_stage2_d127_s0_truth_assets.py` | 输出通过paired scorer的truth/formal打开校验。 |
| T5 | 最小CLI以固定hash输入和exclusive输出构建资产 | `code/scripts/build_d127_s0_truth_assets.py` | verified | `test_build_d127_s0_truth_assets.py` | 不评分、不调用模型。 |
| T6 | 覆盖hash漂移、18行覆盖、标签/role来源、K5前缀与不可覆盖 | `tests/test_stage2_d127_s0_truth_assets.py`、`tests/test_build_d127_s0_truth_assets.py` | verified | `pytest -q tests/test_stage2_d127_s0_scorer.py tests/test_stage2_d127_s0_truth_assets.py tests/test_build_d127_s0_truth_assets.py` | 不连接N607。 |
