# CID-BPP K5 held feasibility run

## 1.身份与状态

- run ID：`cid_bpp_k5_held_r1_30c11b75_20260723`
- 时间：`2026-07-23 03:31:37 +08:00`
- operator：主agent；N607唯一runner待指派`gpt-5.6-terra high`
- candidate：`JOINT-CID-BPP/r0-spike`
- scope：`PHASE1_HELD_PROXY_NON_PROMOTABLE`
- 当前状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`
- 方法Git commit：`30c11b75f8c132b1655b7601dcda4eeb4337b301`
- 设计冻结commit：`52c4d9b42839309d9d2ef5c38701f197289324c6`

## 2.目标、假设与对照

复用GEOFF/r8不可变Phase1 dual archive及真实coverage，在同一K5 held receiver、6个pseudo-new类和3个`leo_*_weak`场景上输出18个slice、72个同row四臂结果。假设是support-only C-id的低秩类内nuisance软抑制与统一BPP头解决互补误差，使`M_JOINT`同时优于`M_DA`和`M_HEAD`并产生正`I_syn`。

四臂固定为：`M0=qKNN(identity)`、`M_DA=qKNN(C-id)`、`M_HEAD=BPP(identity)`、`M_JOINT=BPP(C-id)`。四臂共享同一support/query/K5/registry/Patch A锁；`M_DA`与`M_JOINT`共享metric receipt，`M_HEAD`与`M_JOINT`共享`a0/b0/T`及量化门。

## 3.数据与不可变输入

`p2_min_v1`数据状态继续复用`VALIDATED_ONCE`，本run不改变received IQ、物理ID、receiver/TX集合、场景、K、support/query划分或schema，不重复数据验证。

|输入|SHA256/状态|
|---|---|
|parity receipt|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|
|archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`，8400 rows|
|manifest|`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|coverage摘要|6类/7receiver/4day/3scene；168 cells；zero=0；min=32；K5余量27；`feature_arrays_read=[]`|

held receiver由coverage SHA确定为`1-1`。每个outer lock完整排除该receiver及当前pseudo-new类；只在剩余Phase1 inner held-day episode上选择8个冻结family、执行FP32 teacher和5-shot jackknife。outer query不参与任何fit、选择、teacher、回退或状态更新。

## 4.本地实现、测试与独立review

仅新增：

- `code/cvsrffi/cid_bpp_phase1_nested_lodo.py`
- `code/cvsrffi/cid_bpp_fixed_held_spike.py`
- `tests/test_cid_bpp_fixed_held_spike.py`

`ssr-gpu`主线复核：三文件`py_compile`通过；专项＋相邻C-id/BPP回归`18 passed`；`git diff --check`通过。全8-family合成烟测闭合18 row、72 metrics、每group 8 family，aggregate query MAC=`3,847,824`。独立Terra high复审最终裁决`P0=0，P1=0→MERGE`。

## 5.发布源

|artifact|值|
|---|---|
|Git archive|`E:/type10-7/code/snapshots/cid_bpp_k5_held_r1_30c11b75_20260723/source_30c11b75.zip`|
|ZIP SHA256|`f974e841075aa14fe37e94b1d7c6d0359da8b8e96dc4bb60372726a9ec0d54fe`|
|ZIP size/files|`33,260,749 bytes / 4,466 entries`|
|wrapper SHA256|`edd1d58249385087c33024846c67e7c87734ff2eae0d5fb803c353a5043f05fd`|
|selector raw SHA256|`05958df26be904884b19b2fbbcdcff5c61a78612a5e064f997dfd611901c9a59`|
|held module raw SHA256|`515927c376608a3c94247270c475c3c6b4ee137e694f88bea591b5cb7ed2aed7`|
|test raw SHA256|`a10d541fe2d40af84abf881db927ea0c2ace31ee5d626345478a17db12ffb67b`|

## 6.N607冻结执行合同

- 远端run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cid_bpp_k5_held_r1_30c11b75_20260723`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：上述run root；source：`source_30c11b75`
- 唯一启动命令：`CUDA_VISIBLE_DEVICES=<runner-selected> nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 &`
- PID：待runner回填；GPU：待preflight选择；retry：`NO`
- 预期输出：`packet.json`、`truth.json`、`query.npz`、`prediction.json`、`score.json`、`sha256sums.txt`、`complete.marker`、`pipeline.pid`、`pipeline.exit`、完整log。
- runner必须执行direct preflight、新run root不存在检查、GPU/进程/磁盘检查、ZIP/wrapper/input SHA、解包单根布局、模块`py_compile`/import、`bash -n`，然后单次启动并用短连接监控、回收全部artifact；不得远端编辑、调参、retry、kill/restart或启动125。

## 7.性能证据与停止条件

必须输出同row的old-before、old-after、old adaptation gain、seen-new、H、BA、floor、min-old、min-new、forgetting、old→new、new→old、逐类、receiver、scene、K、`I_syn`、coverage、量化、MAC、CPU mean/P95、VRAM和state bytes。

- 无完整prediction：`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- prediction=18且score=72后，若`M_DA-M0`零argmax变化、`M_JOINT=M_HEAD`、mean`I_syn<=0`，或联合臂损害old/new/min/floor并增加forgetting：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 只有联合正收益且全部冻结门通过，才允许125稳定性screen；125不得选family、rank、阈值、量化或fallback。

## 8.风险与完成后检查

主要风险是完整8-family×inner LODO计算较慢、C-id发生整row identity回退、C-id与BPP重复收缩、或量化/资源门fail-closed。完成后必须完整读取log与全部artifact，独立重算18×4同row指标、argmax、receipt和哈希；进程落地或exit0本身不构成性能成功。
