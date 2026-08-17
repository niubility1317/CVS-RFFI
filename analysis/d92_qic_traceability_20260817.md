# D92 QIC traceability

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|QIC-01|Frozen method 1-5|Support-only quantization self-consistent FP16 intercept closure|`code/cvsrffi/stage2_d92_d42_quantization_intercept_closure.py`|implemented|Focused RED then 7 passed|Exact E0 fallback;only`intercept_fp16`can publish|
|QIC-02|Hard constraints|Arm, lifecycle, K alias and fit inventory|`code/cvsrffi/stage2_d92_e0d_slim.py`|implemented|Focused Slim RED→GREEN and full Slim regression|`full_only`;K>2one actual FULL;K1/K2exact alias|
|QIC-03|Hard constraints|Reachable prediction path and fail-closed receipt audit|`code/cvsrffi/stage2_d92_e0d_query_evaluation.py`|implemented|Query RED→GREEN plus full Query regression|REG1only;raw→mirror whitelist;static resource relation and query-zero closure|
|QIC-04|G0 decision|Truth-free three-scene G0 launcher and receipt validator|G0 release files|TECHNICAL_G0_PASS|v1发布层自引用停止；v2三scene marker PASS，wall 63.808–64.912ms、peak 8–92KiB、query MAC/state exact E0|v2无truth/scorer/性能读取；允许进入Hard9+K1|
|QIC-05|Hard9+K1 decision|9 performance+1 K1、30 scene-arm、8-shard truth-last screen|QIC Hard9 matrix/runner/analyzer/release|ANALYZED_REJECT_ROUTE|v3完成10/10 jobs、8/8 shards；有效truth-last闭包10 paired/60 per-old/30 scene。八项均值全部反向：ΔH=-33.3614pp、Δold BA/`c_old`=-26.0185pp、Δfloor=-23.3333pp、Δseen-new=-34.6296pp、Δforgetting=+26.0185pp、Δnew→old=+18.5833pp、Δold→new=+15.6481pp；query MAC/state exact E0，wall P90=115.503ms，peak=2.84MiB|唯一裁决`REJECT_ROUTE`；不进入Target125，不继续QIC扫描或偏置修补|
