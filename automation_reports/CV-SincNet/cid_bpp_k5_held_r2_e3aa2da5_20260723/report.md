# CID-BPP K5 held技术修订性能run

## 1.身份与状态

- run ID：`cid_bpp_k5_held_r2_e3aa2da5_20260723`
- 时间：`2026-07-23 +08:00`
- operator：主agent；唯一N607 runner=`gpt-5.6-terra high`
- candidate：`JOINT-CID-BPP/r0-spike-tech1`
- scope：`PHASE1_HELD_PROXY_NON_PROMOTABLE`
- 当前状态：`PREREGISTERED / P1_REMOTE_PACKET_VERIFY_PENDING / NO_PERFORMANCE_RESULT`
- 方法Git commit：`e3aa2da5af520e493d40ec343b913ce24e7629dd`
- 原失败run：`cid_bpp_k5_held_r1_30c11b75_20260723`，direct/GPU0/PID458576/exit1/prediction0

## 2.目标与唯一delta

本run只验证fallback-aware verifier技术修订，然后立即完成原冻结的18个K5 held slice、72个同row四臂prediction/score。唯一delta是：非fallback C-id继续强制support provenance；`jackknife_no_direction/jackknife_overlap`强制canonical identity的null provenance、rank0及identity metric receipt。fit、family、metric、BPP、四臂公式、support/query、量化和资源门均不变。

为避免重复计算或重新选择，直接复用原run在失败前已封存的`packet.json/truth.json/query.npz`；旧run失败发生在predict前，因此这些artifact尚未产生任何性能反馈。本run不重建packet，不重选family，也不读取outer truth进行选择。

## 3.不可变输入

`p2_min_v1`及`VALIDATED_ONCE`继续复用；received IQ、物理ID、receiver/TX、场景、K、support/query和schema均未改变，不重复数据验证。

|输入|SHA256/状态|
|---|---|
|原run packet file|`e7132ea454927782acf976c5baa0ed960c37f722791ca45f6c1c7e40d6f1bcc8`|
|packet internal SHA|`b7e56a7b307d91bfb5062519d921e59c35c79ce5f372462218c1a8db5d8a0144`|
|原run truth|`87c37bc8af7d83400a641e87b81602ac7eeb3e8e12b2dc8e26e1f8b597a81a4e`|
|原run query|`be089f42be790a73cd7a95d68cb13956a64735019b10f6cd4ba32199c33c56c9`|
|r8 parity|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|
|r8 archive|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`，8400 rows|
|r8 manifest|`34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|r8 coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|coverage摘要|6类/7receiver/4day/3scene；168 cells；zero=0；min=32；K5余量27；`feature_arrays_read=[]`|

## 4.本地门与独立review

`ssr-gpu`下三文件`py_compile`、专项＋相邻回归`18 passed`、`git diff --check`通过。真实r8 archive的失败同型25条C4 support无query smoke复现overlap=`0.4833083158142645`、`jackknife_overlap`、rank0、null provenance和identity receipt一致。

独立review对代码delta裁决`P0=0，P1=1→REVISE`；唯一P1是N607生成的其他低秩metric receipt无法在本机数值环境复算，必须在N607同环境、launch前用当前Git提交完整执行原packet的`_verify_packet`。这不是方法或协议修改，不允许为此改代码、放宽receipt或重建数据。

## 5.P1远端封存门

runner同步并验证当前Git source后，必须在启动wrapper前对原packet执行support-only完整`_verify_packet`，以noclobber写出：

`preflight_packet_verify.json`：

```json
{"method_commit":"e3aa2da5af520e493d40ec343b913ce24e7629dd","packet_file_sha256":"e7132ea454927782acf976c5baa0ed960c37f722791ca45f6c1c7e40d6f1bcc8","packet_sha256":"b7e56a7b307d91bfb5062519d921e59c35c79ce5f372462218c1a8db5d8a0144","status":"PASS","verifier":"cvsrffi.cid_bpp_fixed_held_spike._verify_packet"}
```

预期文件SHA256=`b153049167629c4ccd1932934d5149d1868183dd4dcfb530a6d0df98f70113b3`。只有该SHA精确通过，P1才清零并允许同一runner启动一次；失败则停止为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重试。

## 6.发布源

|artifact|值|
|---|---|
|release contract commit|`77a41240a26851726092f6c5789e394cbe341055`|
|Git archive|`E:/type10-7/code/snapshots/cid_bpp_k5_held_r2_e3aa2da5_20260723/source_e3aa2da5.zip`|
|ZIP SHA256|`0638a944abc939ebdafcf2c57cd5e06e40676113d4a55eefe0c2b771f75166d8`|
|ZIP size/files|`33,268,292 bytes / 4,469 entries`|
|wrapper SHA256|`b132a1c2a929b10b07c0e3dee4d3ae9988dfa84886f20880e410265868da5605`|
|selector raw SHA256|`05958df26be904884b19b2fbbcdcff5c61a78612a5e064f997dfd611901c9a59`|
|held module raw SHA256|`81e397a58c4d63bdd29defd35a98c4843ac4774d16892df279824f14b896d7f1`|
|test raw SHA256|`728c07923fd851735aa5e15dd80539d304feaffc2a6516ab138edee62e336095`|

## 7.N607冻结执行合同

- 新run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cid_bpp_k5_held_r2_e3aa2da5_20260723`
- 原输入root：`/home/szu2070436088/2510044040/CV-SincNet/runs/cid_bpp_k5_held_r1_30c11b75_20260723/output`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：新run root；source：`source_e3aa2da5`
- 唯一启动命令：`CUDA_VISIBLE_DEVICES=<runner-selected> nohup bash ./run_pipeline.sh > logs/pipeline.log 2>&1 &`
- retry：`NO`；不得运行build、target或125。
- 预期输出：P1 receipt、packet、truth、query、prediction、score、sha256sums、marker、PID、exit和完整log。

runner必须direct preflight、新run root不存在/GPU/进程/磁盘检查、ZIP/wrapper/source/原run inputs/r8 inputs SHA、单根布局、`py_compile`/import、`bash -n`；执行第5节远端无query门并核验receipt SHA；P1清零后仅启动一次，短连接监控并完整回收。禁止远端编辑、调参、重建、retry、kill/restart或125。

## 8.性能与停止条件

完成后必须独立复核18个prediction row、72个score row、logits→argmax、同row四臂、逐类/scene/K/receiver、wrong→correct/correct→wrong、coverage、量化、MAC、CPU mean/P95、VRAM和state bytes。

- 无完整prediction：`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- prediction=18且score=72，但DA零决策变化、JOINT=HEAD、mean`I_syn<=0`或联合伤害old/new/floor/min并增加forgetting：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 只有DA、HEAD各自正收益、JOINT优于两者、`I_syn>0`且全部保护门通过，才允许125稳定性screen。
