# DSSC-ZDOM-JG-qKNN-R4-BCRR/r1f-techfix1完整125实验报告

## 身份与当前状态

- run ID：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix1_f02dd8b0_20260723_145208`
- 创建时间：`2026-07-23T14:52:08+08:00`
- candidate：`DSSC_ZDOM_JG_QKNN_R4_BCRR/design-r1f`；implementation tag=`techfix1`
- 状态：`BLOCKED_PRE_LAUNCH / NO_PERFORMANCE_RESULT`
- 科学方法提交：`849fa342cd46cb8294b5d9b4f5358cea630d0643`
- 技术修复提交：`f02dd8b063437c0916af8bb5e9b39416b3d13f17`
- parent技术失败run：`dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_849fa342_20260723_141937`
- sole launch owner：`/root/dssc_r1f_techfix1_full125_runner`（`gpt-5.6-terra high`）；retry=`false`，未启动。

## 唯一修复与验证

parent run的125个row在prediction前全部失败：119个为sealed TorchScript隐式CUDA0与row GPU混用，6个为N607 PyTorch2.1＋NumPy2的ndarray桥接失败。techfix1只做两项兼容修复：row最前执行并读回`torch.cuda.set_device(row GPU)`；runner的ndarray→Tensor改为contiguous FP32 buffer→reshape→clone→device，Tensor→NumPy改为`tolist`→FP32。候选、输入、五臂、125矩阵、loss、adapter、qKNN、BCRR、INT8、fallback和decision geometry均不变。

- 本地`py_compile`：PASS。
- 专项、协议负例和artifact篡改：`23/23 passed`。
- 真实ADV3B02 checkpoint＋sealed enrollment support无query smoke：`[2,160]`有限FP32，norm min/max=`1/1`，query rows=`0`。
- 独立终审：`MERGE / P0=0 / P1=0`。
- 新run启动前P0：在N607 GPU1设置并读回current device，加载同一sealed runtime并对零IQ前向；另对runner的FP32 buffer双向边界做byte-equivalent零IQ smoke。不得读取target package或生成prediction。

## 冻结完整125

- receiver：`20-1,3-19,7-14,7-7,8-8`
- seed：`713102,713103,713104,713105,713106`
- slice：`(K10,new5),(K10,new10),(K10,new20),(K5,new20),(K1,new20)`
- scene：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- 总数：`125 jobs / 375 prediction slices / 1875 score rows`
- 五臂：`M0/M_DA_NG/M_DA/M_OTHER/M_JOINT`
- 调度：安全时GPU0–7各1个LPT动态worker；部分GPU不可用只降低并发，不缩窄矩阵。
- `I_syn=H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)`。

## 不可变发布包

|artifact|SHA256|
|---|---|
|source ZIP|`f409c63be3e4f037ced68a7387acea529fdc7b51332783359bf7dd92bfff0e2c`|
|DSSC method lock|`7663bbc4b7b199d98caa85b7736547a6927a2c7eb8e6a4de636967edca1e9c10`|
|ground bundle|`109724913cac4f82ff58359b927a7f1e7f7e7d233c0bfd0d05d323f94b1b12da`|
|SOMPH lock|`0496594db4a82efbbf17ec3d67ebc3fb1f0c7ced41b542a5a0bde3482e704523`|
|coverage|`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`|
|checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|sealed runtime|`f119e8cb3f6beda95f0d545205e91b43e4a557af2fd1d025e95d2edf2b8e6e2a`|
|Phase1 archive/manifest|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0` / `34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4`|
|parity receipt|`b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b`|

报告预登记的工作树LF字节SHA为：techfix runner=`943e9d030ac85745067a9beb775ab3dae5078fa0cba6fc9f05288222ffcde754`；测试=`16d3860131033f9877537b8ef6910e89f00cbe0ef53c92003d1ccb73b849ca5d`。实际source ZIP条目使用CRLF字节，runner条目SHA=`1e4d1396cdd7a660a5417a905d021e2c3375ac54e8583fc4c755e5a98ebd5562`，测试条目SHA=`72d4c7c1ac7780ca3594b7ee5818d668dcc1ae0b8bdd78a3056432ec626d4756`；两份文件逐行归一化后与工作树完全一致，但预登记没有区分工作树和发布包字节，导致启动前P0 hash gate失败。本run不生成、不修改、不重验数据；archive、parity、coverage均为`PRESENT_REUSED`。

## N607发布合同

- preflight：`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- remote run：`/home/szu2070436088/2510044040/CV-SincNet/runs/dssc_zdom_jg_qknn_r4_bcrr_full125_r1f_techfix1_f02dd8b0_20260723_145208`
- cache：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix`
- authority：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- runtime：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt`
- output：`<run>/artifacts`，启动前必须不存在；日志=`<run>/logs`；PID=`<run>/launcher.pid`。

同步并重算source ZIP、4个input及固定远端资产SHA，解压`<run>/source`并`py_compile`，通过GPU1零IQ/noquery兼容smoke和实时GPU安全门后，以短连接detach执行：

```text
env PYTHONPATH=<run>/source/code OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python <run>/source/code/scripts/run_dssc_zdom_jg_qknn_r4_bcrr_125.py matrix --phase1-checkpoint <checkpoint> --sealed-runtime <runtime> --package-method-lock <run>/input/somph_method_lock.json --dssc-method-lock <run>/input/dssc_method_lock.json --ground-bundle <run>/input/phase1_dssc_zdom_jg_ground_bundle.npz --coverage-receipt <run>/input/coverage_receipt.json --cache-root <cache> --authority-root <authority> --run-root <run>/artifacts --gpu-ids 0,1,2,3,4,5,6,7
```

## 结果门与runner回填

技术完成必须有125 row receipt、375 prediction、1875 score、完整`matrix_exit/aggregate_index`及真实artifact/hash/row闭合；否则为`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。性能必须同row报告old-before/after、old gain、seen-new、H、BA、floor、min-old/new、forgetting、双向混淆、逐类/receiver/scene/K/seed、coverage、量化、MAC、时延、显存和state。M_DA/M_OTHER无独立正收益、JOINT不胜两者、mean`I_syn<=0`、正协同<188/375或<2/3 scene、任一保护指标退化、协议/INT8/资源失败，均判`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不得用125反向调参。

- N607 GPU1兼容smoke：未执行；P0 hash gate失败后停止
- landed/PID/GPU：source ZIP＋4项input已landed；无PID、未占用新增GPU槽
- exit：未执行启动命令；无`launcher.pid`、无`artifacts`、无本run进程
- archive/parity/coverage：archive/parity未继续核验；coverage输入SHA=`c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17`
- prediction/score：`0/0（未启动）`
- 最终裁决：`BLOCKED_PRE_LAUNCH / NO_PERFORMANCE_RESULT`；全新run修正发布包SHA声明，不复用本run
