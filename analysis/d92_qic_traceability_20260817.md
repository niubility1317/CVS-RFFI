# D92 QIC traceability

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|QIC-01|Frozen method 1-5|Support-only quantization self-consistent FP16 intercept closure|`code/cvsrffi/stage2_d92_d42_quantization_intercept_closure.py`|implemented|Focused RED then 7 passed|Exact E0 fallback;only`intercept_fp16`can publish|
|QIC-02|Hard constraints|Arm, lifecycle, K alias and fit inventory|`code/cvsrffi/stage2_d92_e0d_slim.py`|implemented|Focused Slim RED→GREEN and full Slim regression|`full_only`;K>2one actual FULL;K1/K2exact alias|
|QIC-03|Hard constraints|Reachable prediction path and fail-closed receipt audit|`code/cvsrffi/stage2_d92_e0d_query_evaluation.py`|implemented|Query RED→GREEN plus full Query regression|REG1only;raw→mirror whitelist;static resource relation and query-zero closure|
|QIC-04|G0 decision|Truth-free three-scene G0 launcher and receipt validator|G0 release files|TECHNICAL_G0_PASS|v1发布层自引用停止；v2三scene marker PASS，wall 63.808–64.912ms、peak 8–92KiB、query MAC/state exact E0|v2无truth/scorer/性能读取；允许进入Hard9+K1|
|QIC-05|Hard9+K1 decision|9 performance+1 K1、30 scene-arm、8-shard truth-last screen|QIC Hard9 matrix/runner/analyzer/release|v1_TECHNICAL_STOP；v2_LOCAL_VERIFIED|v1归档SHA漂移在prepare停止；v2 matrix/runner39项与精确归档锁检查通过，py_compile、CLI help、JSON、bash-n通过|v2等待sole runner；当前无Hard9性能结果|
