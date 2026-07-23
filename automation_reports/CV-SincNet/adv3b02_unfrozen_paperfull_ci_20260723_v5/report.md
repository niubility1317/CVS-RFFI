# ADV3B02可训练骨干CSIL与MoPC-HR全量对比v5

- 类型：`FORMAL_PAPER_METHOD_COMPARISON_BASELINE`
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- comparison-only wrapper保持LEO逐样本哈希验证，并为远端旧`_assert_scenario_alignment`返回已验证的首场景reference arrays；本地新版物理独立门禁仍单独空置。
- builder SHA=`5bdb17d40d120ff3ea7c5a454b67137824a7f98a827a7d0c04770edbde6e0465`；30项focused test、`py_compile`、`git diff --check`PASS。
- 新远端root：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v5`；4cell smoke全链PASS后才授权100 package/800cell/2400行矩阵。
- v5写出partial bundle后被最终全场景validator的跨场景token复用门禁阻断；0 receipt/cell/prediction/scoring，无性能结果，矩阵未授权，远端只读封存。
