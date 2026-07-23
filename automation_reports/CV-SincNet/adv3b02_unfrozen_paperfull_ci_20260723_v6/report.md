# ADV3B02可训练骨干CSIL与MoPC-HR全量对比v6

- 状态：`PREREGISTERED_FINAL_BUNDLE_VALIDATOR_SCOPE_FIX_LOCAL_VERIFIED_SMOKE_PENDING`
- comparison-only最终validator逐场景调用原严格validator并合并，保留seal/member/结构/同场景support-query/LEO IQ SHA验证，仅豁免跨场景token唯一性；通用validator未修改。
- builder SHA=`1fe3e54e177ccc5a14ab5ae0dc06fefb776283f0f03652e6c898ea5ef97a3121`；31项focused test、`py_compile`、`git diff --check`PASS。
- 新远端root：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v6`；4cell smoke全链PASS后才授权100 package/800cell/2400行矩阵。
